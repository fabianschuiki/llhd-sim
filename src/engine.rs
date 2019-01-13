// Copyright (c) 2017 Fabian Schuiki

#![allow(dead_code, unused_variables)]
//! The execution engine that advances the simulation step by step.

use llhd::inst::*;
use llhd::value::BlockRef;
use llhd::{Const, ConstInt, ConstKind, ConstTime, ProcessContext, ValueId, ValueRef};
use num::{BigInt, BigUint};
use rayon::prelude::*;
use crate::state::{
    Event, Instance, InstanceKind, InstanceRef, InstanceState, Signal, SignalRef, State,
    TimedInstance, ValueSlot,
};
use std::borrow::BorrowMut;
use std::collections::{HashMap, HashSet};
use crate::tracer::Tracer;

pub struct Engine<'ts, 'tm: 'ts> {
    step: usize,
    state: &'ts mut State<'tm>,
}

impl<'ts, 'tm> Engine<'ts, 'tm> {
    /// Create a new engine to advance some simulation state.
    pub fn new(state: &'ts mut State<'tm>) -> Engine<'ts, 'tm> {
        Engine {
            step: 0,
            state: state,
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
            if self.state.signal_mut(signal).set_value(value) {
                changed_signals.insert(signal);
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
        let events = ready_insts
            .par_iter()
            .map(|index| {
                let mut lk = self.state.instances()[*index].lock().unwrap();
                self.step_instance(lk.borrow_mut())
            })
            .reduce(
                || Vec::new(),
                |mut a, b| {
                    a.extend(b);
                    a
                },
            );
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
        // println!("executing instruction {:?}", inst);

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
                ref x => panic!("expected value to resolve to a signal, got {:?}", x),
            }
        };

        // Resolves a value ref to a constant value. This is the main function
        // that allows the value of individual nodes in a process/entity to be
        // determined for processing.
        let resolve_value = |vr: &ValueRef| -> Const {
            if let ValueRef::Const(ref k) = *vr {
                k.clone()
            } else {
                match values[&vr.id().unwrap()] {
                    ValueSlot::Const(ref k) => k.clone(),
                    ref x => panic!("expected value to resolve to a constant, got {:?}", x),
                }
            }
        };

        match *inst.kind() {
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
                signal: resolve_signal(signal),
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

            UnaryInst(op, _, ref arg) => Action::Value(ValueSlot::Const(Const::new(
                ConstKind::Int(execute_unary(op, resolve_value(arg).as_int())),
            ))),

            BinaryInst(op, _, ref lhs, ref rhs) => {
                Action::Value(ValueSlot::Const(Const::new(ConstKind::Int(
                    execute_binary(op, resolve_value(lhs).as_int(), resolve_value(rhs).as_int()),
                ))))
            }

            BranchInst(BranchKind::Uncond(block)) => Action::Jump(block),
            BranchInst(BranchKind::Cond(ref cond, if_true, if_false)) => {
                let v = resolve_value(cond);
                if v == llhd::const_int(1, 1.into()) {
                    Action::Jump(if_true)
                } else {
                    Action::Jump(if_false)
                }
            }

            // Signal and instance instructions are simply ignored, as they are
            // handled by the builder and only occur in entities.
            SignalInst(..) | InstanceInst(..) => Action::None,

            HaltInst => Action::Suspend(None, InstanceState::Done),

            _ => panic!("unsupported instruction {:#?}", inst),
        }
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
        use num::{zero, BigInt};
        ConstTime::new(
            self.state.time().time().clone(),
            self.state.time().delta() + BigInt::from(1),
            zero(),
        )
    }
}

/// An action to be taken as the result of an instruction's execution.
enum Action {
    /// No action.
    None,
    /// Change the instruction's entry in the value table. Used by instructions
    /// that yield a value to change that value.
    Value(ValueSlot),
    /// Add an event to the event queue.
    Event(Event),
    /// Transfer control to a different block, executing that block's
    /// instructions.
    Jump(BlockRef),
    /// Suspend execution of the current instance and change the instance's
    /// state.
    Suspend(Option<BlockRef>, InstanceState),
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
    use num::ToPrimitive;
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
            BinaryOp::Shl => lhs.value() << rhs.value().to_usize().expect("shift amount too large"),
            BinaryOp::Shr => lhs.value() >> rhs.value().to_usize().expect("shift amount too large"),
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
