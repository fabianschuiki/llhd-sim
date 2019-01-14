// Copyright (c) 2017 Fabian Schuiki

//! A simulation tracer that can store the generated waveform to disk.

use crate::state::{Scope, SignalRef, State};
use llhd::{Const, ConstKind};
use num::{BigInt, BigRational};
use std;
use std::collections::{HashMap, HashSet};

/// A simulation tracer that can operate on the simulation trace as it is being
/// generated.
pub trait Tracer {
    /// Called once at the beginning of the simulation.
    fn init(&mut self, state: &State);

    /// Called by the simulation engine after each time step.
    fn step(&mut self, state: &State, changed: &HashSet<SignalRef>);

    /// Called once at the end of the simulation.
    fn finish(&mut self, state: &State);
}

/// A no-op tracer that does nothing.
pub struct NoopTracer;

impl Tracer for NoopTracer {
    fn init(&mut self, _: &State) {}
    fn step(&mut self, _: &State, _: &HashSet<SignalRef>) {}
    fn finish(&mut self, _: &State) {}
}

/// A tracer that emits the simulation trace as VCD.
pub struct VcdTracer<'tw> {
    writer: &'tw mut std::io::Write,
    abbrevs: HashMap<SignalRef, Vec<(String, String)>>,
    time: BigRational,
    pending: HashMap<SignalRef, Const>,
    precision: BigRational,
}

impl<'tw> VcdTracer<'tw> {
    /// Create a new VCD tracer which will write its VCD to `writer`.
    pub fn new(writer: &'tw mut std::io::Write) -> VcdTracer {
        use num::zero;
        VcdTracer {
            writer: writer,
            abbrevs: HashMap::new(),
            time: zero(),
            pending: HashMap::new(),
            // Hard-code the precision to ps for now. Later on, we might want to
            // make this configurable or automatically determined by the module.
            precision: BigInt::parse_bytes(b"1000000000000", 10).unwrap().into(), // ps
        }
    }

    /// Write the value of all signals that have changed since the last flush.
    /// Clears the `pending` set.
    fn flush(&mut self) {
        let time = (&self.time * &self.precision).trunc();
        write!(self.writer, "#{}\n", time).unwrap();
        for (signal, value) in std::mem::replace(&mut self.pending, HashMap::new()) {
            self.flush_signal(signal, &value);
        }
    }

    /// Write the value of a signal. Called at the beginning of the simulation
    /// to perform a variable dump, and during flush once for each signal that
    /// changed.
    fn flush_signal(&mut self, signal: SignalRef, value: &Const) {
        let value = match **value {
            ConstKind::Int(ref k) => format!("b{:b}", k.value()),
            ConstKind::Time(_) => panic!("time not supported in VCD"),
        };
        for &(ref abbrev, _) in &self.abbrevs[&signal] {
            write!(self.writer, "{} {}\n", value, abbrev).unwrap();
        }
    }

    /// Allocate short names and emit `$scope` statement.
    fn prepare_scope(&mut self, state: &State, scope: &Scope, index: &mut usize) {
        write!(self.writer, "$scope module {} $end\n", scope.name).unwrap();
        let mut probed_signals: Vec<_> = scope.probes.keys().cloned().collect();
        probed_signals.sort();
        for sigref in probed_signals {
            // Allocate short names for the probed signals.
            let probes = &scope.probes[&sigref];
            let names = probes.iter().map(|name| {
                let mut idx = *index;
                let mut id = String::new();
                loop {
                    id.push((33 + idx % 94) as u8 as char);
                    idx /= 94;
                    if idx == 0 {
                        break;
                    }
                }
                *index += 1;
                (id, name.clone())
            });

            // Determine the VCD type of the signal.
            let signal = state.signal(sigref);
            let ty = match **signal.ty() {
                llhd::IntType(width) => format!("wire {}", width),
                ref x => panic!("signal of type {} not supported in VCD", x),
            };

            // Write the abbreviations for this signal.
            let abbrevs_for_signal = self.abbrevs.entry(sigref).or_insert_with(Vec::new);
            for (abbrev, probe) in names {
                write!(self.writer, "$var {} {} {} $end\n", ty, abbrev, probe).unwrap();
                abbrevs_for_signal.push((abbrev, probe));
            }
        }
        for subscope in scope.subscopes.iter() {
            self.prepare_scope(state, subscope, index);
        }
        write!(self.writer, "$upscope $end\n").unwrap();
    }
}

impl<'tw> Tracer for VcdTracer<'tw> {
    fn init(&mut self, state: &State) {
        // Dump the VCD header.
        write!(self.writer, "$version\nllhd-sim\n$end\n").unwrap();
        write!(self.writer, "$timescale 1ps $end\n").unwrap();
        self.prepare_scope(state, state.scope(), &mut 0);
        write!(self.writer, "$enddefinitions $end\n").unwrap();

        // Dump the variables.
        write!(self.writer, "$dumpvars\n").unwrap();
        for &signal in state.probes().keys() {
            self.flush_signal(signal, state.signal(signal).value());
        }
        write!(self.writer, "$end\n").unwrap();
    }

    fn step(&mut self, state: &State, changed: &HashSet<SignalRef>) {
        // If the physical time in seconds of the simulation changed, flush the
        // aggregated pending changes and update the time.
        if self.time != *state.time().time() {
            self.flush();
            self.time = state.time().time().clone();
        }

        // Mark the changed signals for consideration during the next flush.
        self.pending.extend(
            changed
                .iter()
                .map(|&signal| (signal, state.signal(signal).value().clone())),
        );
    }

    fn finish(&mut self, _: &State) {
        self.flush();
    }
}
