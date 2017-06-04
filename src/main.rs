// Copyright (c) 2017 Fabian Schuiki
extern crate llhd;
extern crate clap;
extern crate num;
extern crate rayon;

pub mod builder;
pub mod state;
// pub mod worker;
pub mod engine;
pub mod tracer;

use std::fs::File;
use std::io::prelude::*;
use clap::{Arg, App};
use engine::Engine;
use tracer::{Tracer, VcdTracer};


fn main() {
	// Parse the command line arguments.
	let matches = App::new("llhd-sim")
		.about("Simulates low level hardware description files.")
		.arg(Arg::with_name("INPUT")
			.help("The input file to simulate")
			.required(true)
			.index(1))
		.get_matches();

	// Load the input file.
	let module = {
		let mut contents = String::new();
		File::open(matches.value_of("INPUT").unwrap())
			.unwrap()
			.read_to_string(&mut contents)
			.unwrap();
		match llhd::assembly::parse_str(&contents) {
			Ok(v) => v,
			Err(e) => {
				print!("{}", e);
				std::process::exit(1);
			}
		}
	};

	// Dump the input file back to the console. Just for reference.
	{
		use llhd::visit::Visitor;
		let stdout = std::io::stdout();
		llhd::assembly::Writer::new(&mut stdout.lock()).visit_module(&module);
	}

	// Build the simulation state for this module.
	let mut state = builder::build(&module);

	// Create a new tracer for this state that will generate some waveforms.
	let mut file = File::create("/tmp/output.vcd").unwrap();
	let mut tracer = VcdTracer::new(&mut file);
	tracer.init(&state);

	// Create the simulation engine and run the simulation to completion.
	{
		let mut engine = Engine::new(&mut state);
		engine.run(&mut tracer);
	}

	// Flush the tracer.
	tracer.finish(&state);
}
