// Copyright (c) 2017 Fabian Schuiki

#![allow(dead_code, unused_variables)]
//! The execution engine that advances the simulation step by step.

use crate::{
    state::{
        Event, Instance, InstanceKind, InstanceRef, InstanceState, Signal, SignalRef, State,
        TimedInstance, ValuePointer, ValueSelect, ValueSlot,
    },
    tracer::Tracer,
};
use llhd::*;
use num::{BigInt, BigUint, One, ToPrimitive};
use rayon::prelude::*;
use std::{
    borrow::BorrowMut,
    collections::{HashMap, HashSet},
};

pub struct Engine<'ts, 'tm: 'ts> {
    step: usize,
    state: &'ts mut State<'tm>,
    parallelize: bool,
}

impl<'ts, 'tm> Engine<'ts, 'tm> {
    /// Create a new engine to advance some simulation state.
    pub fn new(state: &'ts mut State<'tm>, parallelize: bool) -> Engine<'ts, 'tm> {
        Engine {
            step: 0,
            state,
            parallelize,
        }
    }

    /// Run the simulation to completion.
    pub fn run(&mut self, tracer: &mut Tracer) {
        while self.step(tracer) {}
    }

    /// Perform one simulation step. Returns true if there are remaining events
    /// in the queue, false otherwise. This can be used as an indication as to
    /// when the simulation is finished.
    pub fn step(&mut self, tracer: &mut Tracer) -> bool {
        println!("STEP {}: {}", self.step, self.state.time());
        self.step += 1;

        // Apply events at this time, note changed signals.
        let events = self.state.take_next_events();
        let mut changed_signals = HashSet::new();
        for Event { signal, value, .. } in events {
            let current = self.state.signal(signal.target).value();
            let modified = modify_pointed_value(&signal, current, value);
            let s = stringify_value(&modified);
            if self.state.signal_mut(signal.target).set_value(modified) {
                changed_signals.insert(signal.target);
            }
        }

        // Wake up units whose timed wait has run out.
        let timed = self.state.take_next_timed();
        for TimedInstance { inst, .. } in timed {
            // println!("timed wake up of {:?}", inst);
            self.state
                .instance(inst)
                .lock()
                .unwrap()
                .set_state(InstanceState::Ready);
        }

        // Wake up units that were sensitive to one of the changed signals.
        for inst in self.state.instances_mut() {
            let mut inst = inst.lock().unwrap();
            let trigger = if let InstanceState::Wait(_, ref signals) = *inst.state() {
                signals.iter().any(|s| changed_signals.contains(s))
            } else {
                false
            };
            if trigger {
                inst.set_state(InstanceState::Ready);
            }
        }

        // Call output hook to write simulation trace to disk.
        tracer.step(self.state, &changed_signals);

        // Execute the instances that are ready.
        let ready_insts: Vec<_> = self
            .state
            .instances()
            .iter()
            .enumerate()
            .filter(|&(_, u)| u.lock().unwrap().state() == &InstanceState::Ready)
            .map(|(i, _)| i)
            .collect();
        // println!("ready_insts: {:?}", ready_insts);
        let events = if self.parallelize {
            ready_insts
                .par_iter()
                .map(|&index| {
                    let mut lk = self.state.instances()[index].lock().unwrap();
                    self.step_instance(lk.borrow_mut())
                })
                .reduce(
                    || Vec::new(),
                    |mut a, b| {
                        a.extend(b);
                        a
                    },
                )
        } else {
            let mut events = Vec::new();
            for &index in &ready_insts {
                let mut lk = self.state.instances()[index].lock().unwrap();
                events.extend(self.step_instance(lk.borrow_mut()));
            }
            events
        };
        self.state.schedule_events(events.into_iter());

        // Gather a list of instances that perform a timed wait and schedule
        // them as to be woken up.
        let timed: Vec<_> = ready_insts
            .iter()
            .filter_map(
                |index| match *self.state.instances()[*index].lock().unwrap().state() {
                    InstanceState::Wait(Some(ref time), _) => Some(TimedInstance {
                        time: time.clone(),
                        inst: InstanceRef::new(*index),
                    }),
                    _ => None,
                },
            )
            .collect();
        self.state.schedule_timed(timed.into_iter());

        // Advance time to next event or process wake, or finish
        match self.state.next_time() {
            Some(t) => {
                self.state.set_time(t);
                true
            }
            None => false,
        }
    }

    /// Continue execution of one single process or entity instance, until it is
    /// suspended by an instruction.
    fn step_instance(&self, instance: &mut Instance) -> Vec<Event> {
        match *instance.kind() {
            InstanceKind::Process {
                prok,
                mut next_block,
            } => {
                let ctx = ProcessContext::new(self.state.context(), prok);
                let mut events = Vec::new();
                // println!("stepping process {}, block {:?}", prok.name(), next_block.map(|b| b.name()));
                while let Some(block) = next_block {
                    next_block = None;
                    for inst in block.insts(&ctx) {
                        let action =
                            self.execute_instruction(inst, instance.values(), self.state.signals());
                        match action {
                            Action::None => (),
                            Action::Value(vs) => instance.set_value(inst.as_ref().into(), vs),
                            Action::Store(ptr, v) => {
                                let current = match instance.value(ptr.target) {
                                    ValueSlot::Variable(k) => k,
                                    x => panic!("variable targeted by store action has value {:?} instead of Variable(...)", x),
                                };
                                let new = modify_pointed_value(&ptr, current, v);
                                instance.set_value(ptr.target, ValueSlot::Variable(new));
                            }
                            Action::Event(e) => events.push(e),
                            Action::Jump(blk) => {
                                let blk = prok.body().block(blk);
                                // println!("jump to block {:?}", blk.name());
                                next_block = Some(blk);
                                break;
                            }
                            Action::Suspend(blk, st) => {
                                let blk = blk.map(|br| prok.body().block(br));
                                instance.set_state(st);
                                match *instance.kind_mut() {
                                    InstanceKind::Process {
                                        ref mut next_block, ..
                                    } => *next_block = blk,
                                    _ => unreachable!(),
                                }
                                return events;
                            }
                        }
                    }
                }

                // We should never arrive here, since every block ends with a
                // terminator instruction that redirects control flow.
                panic!("process starved of instructions");
            }

            InstanceKind::Entity { entity } => {
                // let ctx = EntityContext::new(self.state.context(), entity);
                let mut events = Vec::new();
                for inst in entity.insts() {
                    let action =
                        self.execute_instruction(inst, instance.values(), self.state.signals());
                    match action {
                        Action::None => (),
                        Action::Value(vs) => instance.set_value(inst.as_ref().into(), vs),
                        Action::Store(ptr, vs) => panic!("cannot store in entity"),
                        Action::Event(e) => events.push(e),
                        Action::Jump(..) => panic!("cannot jump in entity"),
                        Action::Suspend(..) => panic!("cannot suspend entity"),
                    }
                }
                let inputs = instance.inputs().iter().cloned().collect();
                instance.set_state(InstanceState::Wait(None, inputs));
                events
            }
        }
    }

    /// Execute a single instruction. Returns an action to be taken in response
    /// to the instruction.
    fn execute_instruction(
        &self,
        inst: &Inst,
        values: &HashMap<ValueId, ValueSlot>,
        signals: &[Signal],
    ) -> Action {
        match std::panic::catch_unwind(|| self.execute_instruction_inner(inst, values, signals)) {
            Ok(x) => x,
            Err(_) => panic!("panic while executing {:#?}", inst),
        }
    }

    /// Execute a single instruction. Returns an action to be taken in response
    /// to the instruction.
    fn execute_instruction_inner(
        &self,
        inst: &Inst,
        values: &HashMap<ValueId, ValueSlot>,
        signals: &[Signal],
    ) -> Action {
        // Resolves a value ref to a constant time value.
        let resolve_delay = |vr: &ValueRef| -> ConstTime {
            match *vr {
                ValueRef::Const(ref k) => match **k {
                    ConstKind::Time(ref k) => k.clone(),
                    _ => panic!("constant value is not a time {:?}", k),
                },
                _ => panic!("value does not resolve to a delay"),
            }
        };

        // Resolves a value ref to a signal ref.
        let resolve_signal = |vr: &ValueRef| -> SignalRef {
            match values[&vr.id().unwrap()] {
                ValueSlot::Signal(r) => r,
                ref x => panic!(
                    "expected value {:?} to resolve to a signal, got {:?}",
                    vr, x
                ),
            }
        };

        // Resolves a value ref to a constant value. This is the main function
        // that allows the value of individual nodes in a process/entity to be
        // determined for processing.
        let resolve_value = |vr: &ValueRef| -> ValueRef {
            match *vr {
                ValueRef::Const(_) | ValueRef::Aggregate(_) => vr.clone(),
                _ => match values[&vr.id().unwrap()] {
                    ValueSlot::Const(ref k) => k.clone(),
                    ref x => panic!(
                        "expected value {:?} to resolve to a constant/aggregate, got {:?}",
                        vr, x
                    ),
                },
            }
        };

        // Resolves a value ref to a variable pointer.
        let resolve_variable_pointer = |vr: &ValueRef| -> ValuePointer<ValueId> {
            let id = vr.id().unwrap();
            match values[&id] {
                ValueSlot::Variable(_) => ValuePointer {
                    target: id,
                    select: vec![],
                    discard: (0, 0),
                },
                ValueSlot::VariablePointer(ref ptr) => ptr.clone(),
                ref x => panic!(
                    "expected value {:?} to resolve to a variable pointer, got {:?}",
                    vr, x
                ),
            }
        };

        // Resolves a value ref to a signal pointer.
        let resolve_signal_pointer = |vr: &ValueRef| -> ValuePointer<SignalRef> {
            match values[&vr.id().unwrap()] {
                ValueSlot::Signal(sig) => ValuePointer {
                    target: sig,
                    select: vec![],
                    discard: (0, 0),
                },
                ValueSlot::SignalPointer(ref ptr) => ptr.clone(),
                ref x => panic!(
                    "expected value {:?} to resolve to a signal pointer, got {:?}",
                    vr, x
                ),
            }
        };

        let action = match *inst.kind() {
            // For the drive instruction we assemble a new event for the queue.
            // The absolute time of the event is calculated either from the
            // delay given by the instruction, or as a single delta step. The
            // signal to be driven and the target value are resolved from the
            // instance's signal and value table.
            DriveInst(ref signal, ref value, ref delay) => Action::Event(Event {
                time: delay
                    .as_ref()
                    .map(|delay| self.time_after_delay(&resolve_delay(delay)))
                    .unwrap_or_else(|| self.time_after_delta()),
                signal: resolve_signal_pointer(signal),
                value: resolve_value(value),
            }),

            WaitInst(block, ref delay, ref sensitivity) => {
                Action::Suspend(
                    Some(block),
                    InstanceState::Wait(
                        // Calculate the absolute simulation time after which the
                        // unit should wake up again, if any.
                        delay
                            .as_ref()
                            .map(|d| self.time_after_delay(&resolve_delay(d))),
                        // Gather a list of signals that will trigger a wake up.
                        sensitivity.iter().map(|s| resolve_signal(s)).collect(),
                    ),
                )
            }

            // The probe instruction simply evaluates to the probed signal's
            // current value.
            ProbeInst(_, ref signal) => Action::Value(ValueSlot::Const(
                signals[resolve_signal(signal).as_usize()].value().clone(),
            )),

            UnaryInst(op, ref ty, ref arg) if ty.is_int() => Action::Value(ValueSlot::Const(
                Const::new(ConstKind::Int(execute_unary(
                    op,
                    resolve_value(arg).unwrap_const().unwrap_int(),
                )))
                .into(),
            )),

            BinaryInst(op, ref ty, ref lhs, ref rhs) if ty.is_int() => {
                Action::Value(ValueSlot::Const(
                    Const::new(ConstKind::Int(execute_binary(
                        op,
                        resolve_value(lhs).unwrap_const().unwrap_int(),
                        resolve_value(rhs).unwrap_const().unwrap_int(),
                    )))
                    .into(),
                ))
            }

            BranchInst(BranchKind::Uncond(block)) => Action::Jump(block),
            BranchInst(BranchKind::Cond(ref cond, if_true, if_false)) => {
                let v = resolve_value(cond);
                if v.unwrap_const() == &llhd::const_int(1, 1.into()) {
                    Action::Jump(if_true)
                } else {
                    Action::Jump(if_false)
                }
            }
            CompareInst(op, ref ty, ref lhs, ref rhs) if ty.is_int() => {
                Action::Value(ValueSlot::Const(
                    Const::new(ConstKind::Int(ConstInt::new(
                        1,
                        match execute_comparison(
                            op,
                            resolve_value(lhs).unwrap_const().unwrap_int(),
                            resolve_value(rhs).unwrap_const().unwrap_int(),
                        ) {
                            false => 0.into(),
                            true => 1.into(),
                        },
                    )))
                    .into(),
                ))
            }
            VariableInst(ref ty) => Action::Value(ValueSlot::Variable(const_zero(ty))),
            LoadInst(ref ty, ref ptr) => {
                let ptr = resolve_variable_pointer(ptr);
                if !ptr.select.is_empty() {
                    warn!("select {:?} ignored for load {:?}", ptr.select, inst);
                }
                Action::Value(match values[&ptr.target] {
                    ValueSlot::Variable(ref k) => ValueSlot::Const(k.clone()),
                    _ => panic!("load target {:?} did not resolve to a variable value", ptr),
                })
            }
            StoreInst(ref ty, ref ptr, ref value) => {
                Action::Store(resolve_variable_pointer(ptr), resolve_value(value))
            }
            ShiftInst(dir, ref ty, ref target, ref insert, ref amount) if ty.is_int() => {
                trace!(
                    "executing shift {:?} on ty {:?}, target {:?}, insert {:?}, amount {:?}",
                    dir,
                    ty,
                    target,
                    insert,
                    amount
                );
                Action::Value(ValueSlot::Const(
                    Const::new(ConstKind::Int(execute_shift_int(
                        dir,
                        resolve_value(target).unwrap_const().unwrap_int(),
                        resolve_value(insert).unwrap_const().unwrap_int(),
                        resolve_value(amount).unwrap_const().unwrap_int(),
                    )))
                    .into(),
                ))
            }

            // Signal and instance instructions are simply ignored, as they are
            // handled by the builder and only occur in entities.
            SignalInst(..) | InstanceInst(..) => Action::None,

            HaltInst => Action::Suspend(None, InstanceState::Done),
            _ => panic!("unsupported instruction {:#?}", inst),
        };

        trace!(
            "[{}] i{}: {} # {:?}",
            self.state.time(),
            llhd::ValueId::from(inst.as_ref().into()),
            action,
            inst.kind()
        );
        action
    }

    /// Calculate the time at which an event occurs, given an optional delay. If
    /// the delay is omitted, the next delta cycle is returned.
    fn time_after_delay(&self, delay: &ConstTime) -> ConstTime {
        use num::{zero, Zero};
        let mut time = self.state.time().time().clone();
        let mut delta = self.state.time().delta().clone();
        let mut epsilon = self.state.time().epsilon().clone();
        if !delay.time().is_zero() {
            time = time + delay.time();
            delta = zero();
            epsilon = zero();
        }
        if !delay.delta().is_zero() {
            delta = delta + delay.delta();
            epsilon = zero();
        }
        epsilon = epsilon + delay.epsilon();
        ConstTime::new(time, delta, epsilon)
    }

    /// Calculate the absolute time of the next delta step.
    fn time_after_delta(&self) -> ConstTime {
        ConstTime::new(
            self.state.time().time().clone(),
            self.state.time().delta() + 1,
            num::zero(),
        )
    }
}

/// An action to be taken as the result of an instruction's execution.
#[derive(Debug)]
enum Action {
    /// No action.
    None,
    /// Change the instruction's entry in the value table. Used by instructions
    /// that yield a value to change that value.
    Value(ValueSlot),
    /// Change another value's entry in the value table. Used by instructions
    /// to simulate writing to memory.
    Store(ValuePointer<ValueId>, ValueRef),
    /// Add an event to the event queue.
    Event(Event),
    /// Transfer control to a different block, executing that block's
    /// instructions.
    Jump(BlockRef),
    /// Suspend execution of the current instance and change the instance's
    /// state.
    Suspend(Option<BlockRef>, InstanceState),
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            Action::None => write!(f, "<no action>"),
            Action::Value(ref v) => write!(f, "= {:?}", v),
            Action::Store(ref ptr, ref v) => write!(f, "*i{} = {:?}", ptr.target, v),
            Action::Event(ref ev) => write!(f, "@{} {:?} <= {:?}", ev.time, ev.signal, ev.value),
            Action::Jump(..) | Action::Suspend(..) => write!(f, "{:?}", self),
        }
    }
}

fn execute_unary(op: UnaryOp, arg: &ConstInt) -> ConstInt {
    ConstInt::new(
        arg.width(),
        match op {
            UnaryOp::Not => bigint_bitwise_unary(arg.width(), arg.value(), |arg| !arg),
        },
    )
}

fn execute_binary(op: BinaryOp, lhs: &ConstInt, rhs: &ConstInt) -> ConstInt {
    use num::Integer;
    use std::ops::Rem;
    ConstInt::new(
        lhs.width(),
        match op {
            BinaryOp::Add => lhs.value() + rhs.value(),
            BinaryOp::Sub => lhs.value() - rhs.value(),
            BinaryOp::Mul => lhs.value() * rhs.value(),
            BinaryOp::Div => lhs.value() / rhs.value(),
            BinaryOp::Mod => lhs.value().mod_floor(rhs.value()),
            BinaryOp::Rem => lhs.value().rem(rhs.value()),
            BinaryOp::And => {
                bigint_bitwise_binary(lhs.width(), lhs.value(), rhs.value(), |lhs, rhs| lhs & rhs)
            }
            BinaryOp::Or => {
                bigint_bitwise_binary(lhs.width(), lhs.value(), rhs.value(), |lhs, rhs| lhs | rhs)
            }
            BinaryOp::Xor => {
                bigint_bitwise_binary(lhs.width(), lhs.value(), rhs.value(), |lhs, rhs| lhs ^ rhs)
            }
        },
    )
}

/// Evaluate a unary bitwise logic operation on a big integer.
fn bigint_bitwise_unary<F>(width: usize, arg: &BigInt, op: F) -> BigInt
where
    F: Fn(u8) -> u8,
{
    use num::bigint::Sign;
    use std::iter::repeat;
    let mut bytes: Vec<u8> = make_unsigned(width, arg)
        .to_bytes_le()
        .into_iter()
        .chain(repeat(0))
        .map(op)
        .take((width + 7) / 8)
        .collect();
    let unused_bits = bytes.len() * 8 - width;
    *bytes.last_mut().unwrap() &= 0xFF >> unused_bits;
    BigInt::from_bytes_le(Sign::Plus, &bytes)
}

/// Evaluate a binary bitwise logic operation between two big integers.
fn bigint_bitwise_binary<F>(width: usize, lhs: &BigInt, rhs: &BigInt, op: F) -> BigInt
where
    F: Fn(u8, u8) -> u8,
{
    use num::bigint::Sign;
    use std::iter::repeat;
    let mut bytes: Vec<u8> = make_unsigned(width, lhs)
        .to_bytes_le()
        .into_iter()
        .chain(repeat(0))
        .zip(
            make_unsigned(width, rhs)
                .to_bytes_le()
                .into_iter()
                .chain(repeat(0)),
        )
        .map(|(l, r)| op(l, r))
        .take((width + 7) / 8)
        .collect();
    let unused_bits = bytes.len() * 8 - width;
    *bytes.last_mut().unwrap() &= 0xFF >> unused_bits;
    BigInt::from_bytes_le(Sign::Plus, &bytes)
}

/// Convert the signed big integer into an unsigned big integer, taking the
/// two's complement if the argument is negative.
fn make_unsigned(width: usize, arg: &BigInt) -> BigUint {
    use num::bigint::Sign;
    use num::pow;
    match arg.sign() {
        Sign::Plus | Sign::NoSign => arg.to_biguint(),
        Sign::Minus => (pow(BigInt::from(2), width) - arg).to_biguint(),
    }
    .unwrap()
}

fn execute_comparison(op: CompareOp, lhs: &ConstInt, rhs: &ConstInt) -> bool {
    match op {
        CompareOp::Eq => lhs.value() == rhs.value(),
        CompareOp::Neq => lhs.value() != rhs.value(),
        CompareOp::Slt => lhs.value() < rhs.value(),
        CompareOp::Sle => lhs.value() <= rhs.value(),
        CompareOp::Sgt => lhs.value() > rhs.value(),
        CompareOp::Sge => lhs.value() >= rhs.value(),
        CompareOp::Ult => {
            make_unsigned(lhs.width(), lhs.value()) < make_unsigned(rhs.width(), rhs.value())
        }
        CompareOp::Ule => {
            make_unsigned(lhs.width(), lhs.value()) <= make_unsigned(rhs.width(), rhs.value())
        }
        CompareOp::Ugt => {
            make_unsigned(lhs.width(), lhs.value()) > make_unsigned(rhs.width(), rhs.value())
        }
        CompareOp::Uge => {
            make_unsigned(lhs.width(), lhs.value()) >= make_unsigned(rhs.width(), rhs.value())
        }
    }
}

/// Execute a shift instruction on an integer target.
fn execute_shift_int(
    dir: ShiftDir,
    target: &ConstInt,
    insert: &ConstInt,
    amount: &ConstInt,
) -> ConstInt {
    let shifted = match dir {
        ShiftDir::Left => {
            target.value() << amount.value().to_usize().expect("shift amount too large")
        }
        ShiftDir::Right => {
            target.value() >> amount.value().to_usize().expect("shift amount too large")
        }
    };
    ConstInt::new(target.width(), shifted)
}

fn value_type(v: &ValueRef) -> Type {
    match v {
        ValueRef::Const(k) => k.ty(),
        ValueRef::Aggregate(k) => k.ty(),
        _ => panic!("cannot determine type of {:?}", v),
    }
}

/// Modify a value according to a pointer.
fn modify_pointed_value<T>(ptr: &ValuePointer<T>, current: &ValueRef, new: ValueRef) -> ValueRef {
    let new = if ptr.discard != (0, 0) {
        let (left, right) = ptr.discard;
        // trace!("discard ({}, {}) of {:?}", left, right, new);
        match new {
            ValueRef::Const(k) => match *k {
                ConstKind::Int(ref ki) => {
                    let v = ki.value() % (BigInt::one() << (ki.width() - left));
                    let v = v >> right;
                    // trace!("{} => {}", ki.value(), v);
                    const_int(ki.width() - left - right, v).into()
                }
                _ => panic!("cannot discard {:?} on constant {:?}", ptr.discard, k),
            },
            // ValueRef::Aggregate(a) => new.clone(),
            _ => panic!("cannot discard {:?} on value {:?}", ptr.discard, new),
        }
    } else {
        new
    };
    modify_selected_value(&ptr.select, current, new)
}

/// Modify a value according to a sequence of selection operations.
fn modify_selected_value(select: &[ValueSelect], current: &ValueRef, new: ValueRef) -> ValueRef {
    if select.is_empty() {
        // assert_eq!(value_type(current), value_type(&new));
        return new;
    }
    match select[0] {
        ValueSelect::Element(index) => {
            // trace!("element {} of {:?}", index, current);
            match current {
                ValueRef::Const(k) => match **k {
                    ConstKind::Int(ref ki) => {
                        let current_bit: BigInt = (ki.value() >> index) % 2;
                        let modified_bit = {
                            modify_selected_value(
                                &select[1..],
                                &const_int(1, current_bit.clone()).into(),
                                new,
                            )
                            .unwrap_const()
                            .unwrap_int()
                            .value()
                                % 2
                        };
                        let modified =
                            ki.value() - (current_bit << index) + (modified_bit << index);
                        // trace!("{} => {}", ki.value(), modified);
                        const_int(ki.width(), modified).into()
                    }
                    _ => panic!("cannot select {:?} on constant {:?}", select[0], k),
                },
                ValueRef::Aggregate(a) => match **a {
                    AggregateKind::Struct(ref a) => {
                        let mut fields = a.fields().to_vec();
                        fields[index] =
                            modify_selected_value(&select[1..], &a.fields()[index], new);
                        const_struct(fields).into()
                    }
                    AggregateKind::Array(ref a) => {
                        let mut elements = a.elements().to_vec();
                        elements[index] =
                            modify_selected_value(&select[1..], &elements[index], new);
                        const_array(a.ty().clone(), elements)
                    }
                },
                _ => panic!("cannot select {:?} on value {:?}", select[0], current),
            }
        }
        ValueSelect::Slice(offset, length) => {
            // trace!("slice ({}, {}) of {:?}", offset, length, current);
            match current {
                ValueRef::Const(k) => match **k {
                    ConstKind::Int(ref ki) => {
                        let kui = &ki.value().to_biguint().unwrap();
                        let mod_len = BigUint::one() << length;
                        let mod_off = BigUint::one() << offset;
                        let lower = kui % mod_off;
                        let upper = (kui >> (length + offset)) << (length + offset);
                        let vc = (kui >> offset) % mod_len;
                        let vn = modify_selected_value(
                            &select[1..],
                            &const_int(length, vc.into()).into(),
                            new,
                        )
                        .unwrap_const()
                        .unwrap_int()
                        .value()
                        .to_biguint()
                        .unwrap();
                        let v = upper | (vn << offset) | lower;
                        // trace!("{} => {}", kui, v);
                        const_int(ki.width(), v.into()).into()
                    }
                    _ => panic!("cannot select {:?} on constant {:?}", select[0], k),
                },
                ValueRef::Aggregate(a) => match **a {
                    AggregateKind::Array(ref a) => {
                        let mut elements = a.elements().to_vec();
                        let modified = modify_selected_value(
                            &select[1..],
                            &const_array(
                                array_ty(length, a.ty().unwrap_array().1.clone()),
                                elements[offset..offset + length].to_vec(),
                            ),
                            new,
                        );
                        let modified = match **modified.unwrap_aggregate() {
                            AggregateKind::Array(ref a) => a.elements(),
                            _ => panic!("modified selected value {:?} is not an array", modified),
                        };
                        elements[offset..offset + length].clone_from_slice(modified);
                        const_array(a.ty().clone(), elements)
                    }
                    _ => panic!("cannot select {:?} on aggregate {:?}", select[0], a),
                },
                _ => panic!("cannot select {:?} on value {:?}", select[0], current),
            }
        }
    }
}

fn stringify_value(value: &ValueRef) -> String {
    match *value {
        ValueRef::Const(ref k) => match **k {
            ConstKind::Int(ref v) => format!("{}", v.value()),
            ConstKind::Time(ref t) => format!("{}", t),
        },
        ValueRef::Aggregate(ref a) => match **a {
            AggregateKind::Struct(ref a) => {
                let mut s = String::from("{");
                let mut first = true;
                for f in a.fields() {
                    if !first {
                        s.push_str(", ");
                    }
                    s.push_str(&stringify_value(f));
                    first = false;
                }
                s.push('}');
                s
            }
            AggregateKind::Array(ref a) => {
                let mut s = String::from("[");
                let mut first = true;
                for e in a.elements() {
                    if !first {
                        s.push_str(", ");
                    }
                    s.push_str(&stringify_value(e));
                    first = false;
                }
                s.push(']');
                s
            }
        },
        _ => format!("{:?}", value),
    }
}
