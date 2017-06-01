// Copyright (c) 2017 Fabian Schuiki

//! The simulation builder creates the structure necessary for simulating a
//! design.

use std::collections::HashMap;

use llhd::Module;
use llhd::ModuleContext;
use llhd::Argument;
use llhd::{Value, ValueRef, ValueId};
use llhd::Type;
use llhd::Process;
use llhd::Entity;
use llhd::Block;


pub struct Builder<'tm> {
	module: &'tm Module,
	ctx: ModuleContext<'tm>,
	signals: Vec<Signal>,
	signal_probes: HashMap<SignalRef, Vec<String>>,
	// name_stack: Vec<String>,
}

impl<'tm> Builder<'tm> {
	/// Create a new builder for the given module.
	pub fn new(module: &Module) -> Builder {
		Builder {
			module: module,
			ctx: ModuleContext::new(module),
			signals: Vec::new(),
			signal_probes: HashMap::new(),
			// name_stack: Vec::new(),
		}
	}

	/// Allocate a new signal in the simulation and return a reference to it.
	pub fn alloc_signal(&mut self, _: Type) -> SignalRef {
		let r = SignalRef(self.signals.len());
		self.signals.push(Signal);
		r
	}

	/// Allocate a new signal probe in the simulation. This essentially assigns
	/// a name to a signal which is also known to the user.
	pub fn alloc_signal_probe(&mut self, signal: SignalRef, name: String) {
		println!("probe \"{}\" on signal {:?}", name, signal);
		self.signal_probes.entry(signal).or_insert(Vec::new()).push(name);
	}

	/// Instantiate a process or entity for simulation. This recursively builds
	/// the simulation structure for all subunits as necessary.
	pub fn instantiate(&mut self, unit: &ValueRef, inputs: Vec<SignalRef>, outputs: Vec<SignalRef>) {
		// Create signal probes for the input and output arguments of the unit.
		let values =
			unit_inputs(self.module, unit).iter().zip(inputs.iter()).chain(
			unit_outputs(self.module, unit).iter().zip(outputs.iter())).map(|(arg, &sig)|{
				if let Some(name) = arg.name() {
					self.alloc_signal_probe(sig, name.into());
				}
				(arg.as_ref().into(), ValueSlot::Signal(sig))
			}).collect();

		// Gather the process-/entity-specific information.
		let kind = match *unit {

			ValueRef::Process(r) => {
				let process = self.module.process(r);
				InstanceKind::Process {
					prok: process,
					next_block: process.body().blocks().next(),
				}
			}

			ValueRef::Entity(r) => {
				let entity = self.module.entity(r);
				InstanceKind::Entity {
					entity: entity,
				}
			}

			_ => panic!("instantiate called on non-unit value {:?}", unit)
		};

		println!("values: {:#?}", values);

		// Create the unit instance.
		let inst = Instance {
			values: values,
			kind: kind,
		};

		// TODO: Do something with the instance. Add it to "self" somewhere for
		// example :)
	}
}


/// Build the simulation for a module.
pub fn build(module: &Module) {
	let mut builder = Builder::new(module);

	// Find the first process or entity in the module, which we will use as the
	// simulation's root unit.
	let root = module.values().find(|v| match **v {
		ValueRef::Process(_) => true,
		ValueRef::Entity(_) => true,
		_ => false,
	}).expect("no process or entity found that can be simulated");

	// Allocate the input and output signals for the top-level module.
	let inputs: Vec<_> = unit_inputs(&module, root).iter().map(|arg| builder.alloc_signal(arg.ty())).collect();
	let outputs: Vec<_> = unit_outputs(&module, root).iter().map(|arg| builder.alloc_signal(arg.ty())).collect();

	// Instantiate the top-level module.
	builder.instantiate(root, inputs, outputs);
}


/// Obtain a slice of the input arguments to a unit.
pub fn unit_inputs<'a>(module: &'a Module, unit: &ValueRef) -> &'a [Argument] {
	match *unit {
		ValueRef::Process(r) => module.process(r).inputs(),
		ValueRef::Entity(r) => module.entity(r).inputs(),
		_ => panic!("unit_inputs called on non-unit {:?}", unit),
	}
}


/// Obtain a slice of the output arguments from a unit.
pub fn unit_outputs<'a>(module: &'a Module, unit: &ValueRef) -> &'a [Argument] {
	match *unit {
		ValueRef::Process(r) => module.process(r).outputs(),
		ValueRef::Entity(r) => module.entity(r).outputs(),
		_ => panic!("unit_outputs called on non-unit {:?}", unit),
	}
}


#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct SignalRef(usize);

pub struct Signal;

pub struct Instance<'tm> {
	values: HashMap<ValueId, ValueSlot>,
	kind: InstanceKind<'tm>,
}

#[derive(Debug)]
pub enum ValueSlot {
	Signal(SignalRef),
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
