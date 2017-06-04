// Copyright (c) 2017 Fabian Schuiki

//! The simulation state.

use std::cmp::Ordering;
use std::sync::Mutex;
use std::collections::{HashMap, BinaryHeap};
use num::zero;
use llhd::{Module, ValueId, Process, Entity, Block, Const, ConstTime, ModuleContext, Type};


pub struct State<'tm> {
	module: &'tm Module,
	context: ModuleContext<'tm>,
	time: ConstTime,
	signals: Vec<Signal>,
	probes: HashMap<SignalRef, Vec<String>>,
	insts: Vec<Mutex<Instance<'tm>>>,
	events: BinaryHeap<Event>,
	timed: BinaryHeap<TimedInstance>,
}

impl<'tm> State<'tm> {
	/// Create a new simulation state.
	pub fn new(module: &'tm Module, signals: Vec<Signal>, probes: HashMap<SignalRef, Vec<String>>, insts: Vec<Mutex<Instance<'tm>>>) -> State<'tm> {
		State {
			module: module,
			context: ModuleContext::new(module),
			time: ConstTime::new(zero(), zero(), zero()),
			signals: signals,
			probes: probes,
			insts: insts,
			events: BinaryHeap::new(),
			timed: BinaryHeap::new(),
		}
	}

	/// Get the module whose state this object holds.
	pub fn module(&self) -> &'tm Module {
		self.module
	}

	/// Get the module context for the module whose state this object holds
	pub fn context(&self) -> &ModuleContext {
		&self.context
	}

	/// Get the current simulation time.
	pub fn time(&self) -> &ConstTime {
		&self.time
	}

	/// Change the current simulation time.
	pub fn set_time(&mut self, time: ConstTime) {
		self.time = time
	}

	/// Get a slice of instances in the state.
	pub fn instances(&self) -> &[Mutex<Instance<'tm>>] {
		&self.insts
	}

	/// Get a mutable slice of instances in the state.
	pub fn instances_mut(&mut self) -> &mut [Mutex<Instance<'tm>>] {
		&mut self.insts
	}

	/// Get a reference to an instance in the state.
	pub fn instance(&self, ir: InstanceRef) -> &Mutex<Instance<'tm>> {
		&self.insts[ir.0]
	}

	/// Obtain a reference to one of the state's signals.
	pub fn signal(&self, sr: SignalRef) -> &Signal {
		&self.signals[sr.0]
	}

	/// Obtain a mutable reference to one of the state's signals.
	pub fn signal_mut(&mut self, sr: SignalRef) -> &mut Signal {
		&mut self.signals[sr.0]
	}

	/// Get a reference to all signals of this state.
	pub fn signals(&self) -> &[Signal] {
		&self.signals
	}

	/// Get a slice of all probe signals and the corresponding names.
	pub fn probes(&self) -> &HashMap<SignalRef, Vec<String>> {
		&self.probes
	}

	/// Add a set of events to the schedule.
	pub fn schedule_events<I>(&mut self, iter: I)
	where I: Iterator<Item = Event> {
		let time = self.time.clone();
		self.events.extend(iter.map(|i|{ assert!(i.time > time); i }));
	}

	/// Add a set of timed instances to the schedule.
	pub fn schedule_timed<I>(&mut self, iter: I)
	where I: Iterator<Item = TimedInstance> {
		let time = self.time.clone();
		self.timed.extend(iter.map(|i|{ assert!(i.time > time); i }));
	}

	/// Dequeue all events due at the current time.
	pub fn take_next_events(&mut self) -> Vec<Event> {
		let mut v = Vec::new();
		while self.events.peek().map(|x| x.time == self.time).unwrap_or(false) {
			v.push(self.events.pop().unwrap());
		}
		v
	}

	/// Dequeue all timed instances due at the current time.
	pub fn take_next_timed(&mut self) -> Vec<TimedInstance> {
		let mut v = Vec::new();
		while self.timed.peek().map(|x| x.time == self.time).unwrap_or(false) {
			v.push(self.timed.pop().unwrap());
		}
		v
	}

	/// Determine the time of the next simulation step. This is the lowest time
	/// value of any event or wake up request in the schedule. If both the event
	/// and timed instances queue are empty, None is returned.
	pub fn next_time(&self) -> Option<ConstTime> {
		use std::cmp::min;
		match (self.events.peek().map(|e| &e.time), self.timed.peek().map(|t| &t.time)) {
			(Some(e), Some(t)) => Some(min(e,t).clone()),
			(Some(e), None) => Some(e.clone()),
			(None, Some(t)) => Some(t.clone()),
			(None, None) => None,
		}
	}
}


#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct SignalRef(usize);

impl SignalRef {
	/// Create a new signal reference.
	pub fn new(id: usize) -> SignalRef {
		SignalRef(id)
	}
}

pub struct Signal {
	ty: Type,
	value: Const,
}

impl Signal {
	/// Create a new signal.
	pub fn new(ty: Type, value: Const) -> Signal {
		Signal {
			ty: ty,
			value: value,
		}
	}

	/// Get the signal's type.
	pub fn ty(&self) -> &Type {
		&self.ty
	}

	/// Get the signal's current value.
	pub fn value(&self) -> &Const {
		&self.value
	}

	/// Change the signal's current value. Returns whether the values were
	/// identical.
	pub fn set_value(&mut self, value: Const) -> bool {
		if self.value != value {
			self.value = value;
			true
		} else {
			false
		}
	}
}


pub struct Instance<'tm> {
	values: HashMap<ValueId, ValueSlot>,
	kind: InstanceKind<'tm>,
	state: InstanceState,
}

impl<'tm> Instance<'tm> {
	pub fn new(values: HashMap<ValueId, ValueSlot>, kind: InstanceKind<'tm>) -> Instance<'tm> {
		Instance {
			values: values,
			kind: kind,
			state: InstanceState::Ready,
		}
	}

	/// Get the instance's current state.
	pub fn state(&self) -> &InstanceState {
		&self.state
	}

	/// Change the instance's current state.
	pub fn set_state(&mut self, state: InstanceState) {
		self.state = state;
	}

	pub fn kind(&self) -> &InstanceKind<'tm> {
		&self.kind
	}

	pub fn kind_mut(&mut self) -> &mut InstanceKind<'tm> {
		&mut self.kind
	}

	/// Get a reference to the value table of this instance.
	pub fn values(&self) -> &HashMap<ValueId, ValueSlot> {
		&self.values
	}

	/// Access an entry in this instance's value table.
	pub fn value(&self, id: ValueId) -> &ValueSlot {
		self.values.get(&id).unwrap()
	}

	/// Change an entry in this instance's value table.
	pub fn set_value(&mut self, id: ValueId, value: ValueSlot) {
		self.values.insert(id, value);
	}
}

#[derive(Debug)]
pub enum ValueSlot {
	Signal(SignalRef),
	Const(Const),
}

pub enum InstanceKind<'tm> {
	Process {
		prok: &'tm Process,
		next_block: Option<&'tm Block>,
	},
	Entity {
		entity: &'tm Entity,
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstanceState {
	Ready,
	Wait(Option<ConstTime>, Vec<SignalRef>),
	Done,
}

#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct InstanceRef(usize);

impl InstanceRef {
	/// Create a new instance reference.
	pub fn new(id: usize) -> InstanceRef {
		InstanceRef(id)
	}
}


/// An event that can be scheduled in a binary heap, forming an event queue. The
/// largest element, i.e. the one at the top of the heap, is the one with the
/// lowest time value.
#[derive(Debug, Eq, PartialEq)]
pub struct Event {
	pub time: ConstTime,
	pub signal: SignalRef,
	pub value: Const,
}

impl Ord for Event {
	fn cmp(&self, rhs: &Event) -> Ordering {
		match self.time.cmp(&rhs.time) {
			Ordering::Equal => self.signal.cmp(&rhs.signal),
			Ordering::Greater => Ordering::Less,
			Ordering::Less => Ordering::Greater,
		}
	}
}

impl PartialOrd for Event {
	fn partial_cmp(&self, rhs: &Event) -> Option<Ordering> {
		Some(self.cmp(rhs))
	}
}


/// A notice that an instance is in a wait state and wants to be resumed once a
/// certain simulation time has been reached. TimedInstance objects can be
/// scheduled in a binary heap, which forms a wake up queue. The largest
/// element, i.e. the one at the top of the heap, is the one with the lowest
/// time value.
#[derive(Debug, Eq, PartialEq)]
pub struct TimedInstance {
	pub time: ConstTime,
	pub inst: InstanceRef,
}

impl Ord for TimedInstance {
	fn cmp(&self, rhs: &TimedInstance) -> Ordering {
		match self.time.cmp(&rhs.time) {
			Ordering::Equal => self.inst.cmp(&rhs.inst),
			Ordering::Greater => Ordering::Less,
			Ordering::Less => Ordering::Greater,
		}
	}
}

impl PartialOrd for TimedInstance {
	fn partial_cmp(&self, rhs: &TimedInstance) -> Option<Ordering> {
		Some(self.cmp(rhs))
	}
}
