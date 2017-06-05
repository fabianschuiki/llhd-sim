// Copyright (c) 2017 Fabian Schuiki

//! The simulation builder creates the structure necessary for simulating a
//! design.

use std::collections::HashMap;
use std::sync::Mutex;
use llhd::Module;
use llhd::Argument;
use llhd::Value;
use llhd::ValueRef;
use llhd::ValueId;
use llhd::Type;
use llhd::const_zero;
use llhd::inst::*;
use state::*;


pub struct Builder<'tm> {
	module: &'tm Module,
	signals: Vec<Signal>,
	probes: HashMap<SignalRef, Vec<String>>,
	insts: Vec<Instance<'tm>>,
	// name_stack: Vec<String>,
}

impl<'tm> Builder<'tm> {
	/// Create a new builder for the given module.
	pub fn new(module: &Module) -> Builder {
		Builder {
			module: module,
			signals: Vec::new(),
			probes: HashMap::new(),
			insts: Vec::new(),
			// name_stack: Vec::new(),
		}
	}

	/// Allocate a new signal in the simulation and return a reference to it.
	pub fn alloc_signal(&mut self, ty: &Type, init: Option<&ValueRef>) -> SignalRef {
		let r = SignalRef::new(self.signals.len());
		self.signals.push(Signal::new(
			ty.clone(),
			init.map(|v| v.as_const().clone()).unwrap_or_else(|| const_zero(ty))
		));
		r
	}

	/// Allocate a new signal probe in the simulation. This essentially assigns
	/// a name to a signal which is also known to the user.
	pub fn alloc_signal_probe(&mut self, signal: SignalRef, name: String) {
		self.probes.entry(signal).or_insert(Vec::new()).push(name);
	}

	/// Instantiate a process or entity for simulation. This recursively builds
	/// the simulation structure for all subunits as necessary.
	pub fn instantiate(&mut self, unit: &ValueRef, mut inputs: Vec<SignalRef>, outputs: Vec<SignalRef>) {
		// Create signal probes for the input and output arguments of the unit.
		let mut values: HashMap<ValueId, ValueSlot> =
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

				// Allocate signals and instantiate subunits.
				for inst in entity.insts() {
					match *inst.kind() {
						SignalInst(ref ty, ref init) => {
							let sig = self.alloc_signal(ty, init.as_ref());
							inputs.push(sig); // entity is re-evaluated when this signal changes
							if let Some(name) = inst.name() {
								self.alloc_signal_probe(sig, name.into());
							}
							values.insert(inst.as_ref().into(), ValueSlot::Signal(sig));
						}
						InstanceInst(_, ref unit, ref ins, ref outs) => {
							let resolve_signal = |value: &ValueRef|{
								match values[&value.id().unwrap()] {
									ValueSlot::Signal(sig) => sig,
									_ => panic!("value does not resolve to a signal"),
								}
							};
							self.instantiate(
								unit,
								ins.iter().map(&resolve_signal).collect(),
								outs.iter().map(&resolve_signal).collect()
							);
						}
						_ => ()
					}
				}

				InstanceKind::Entity {
					entity: entity,
				}
			}

			_ => panic!("instantiate called on non-unit value {:?}", unit)
		};

		// Create the unit instance.
		self.insts.push(Instance::new(values, kind, inputs, outputs))
	}

	/// Consume the builder and assemble the simulation state.
	pub fn finish(self) -> State<'tm> {
		State::new(
			self.module,
			self.signals,
			self.probes,
			self.insts.into_iter().map(|i| Mutex::new(i)).collect(),
		)
	}
}


/// Build the simulation for a module.
pub fn build(module: &Module) -> State {
	let mut builder = Builder::new(module);

	// Find the first process or entity in the module, which we will use as the
	// simulation's root unit.
	let root = module.values().rev().find(|v| match **v {
		ValueRef::Process(_) => true,
		ValueRef::Entity(_) => true,
		_ => false,
	}).expect("no process or entity found that can be simulated");

	// Allocate the input and output signals for the top-level module.
	let inputs: Vec<_> = unit_inputs(&module, root)
		.iter()
		.map(|arg| builder.alloc_signal(arg.ty().as_signal(), None))
		.collect();
	let outputs: Vec<_> = unit_outputs(&module, root)
		.iter()
		.map(|arg| builder.alloc_signal(arg.ty().as_signal(), None))
		.collect();

	// Instantiate the top-level module.
	builder.instantiate(root, inputs, outputs);

	// Build the simulation state.
	builder.finish()
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
