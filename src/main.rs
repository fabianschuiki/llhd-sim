// Copyright (c) 2017 Fabian Schuiki
extern crate llhd;
extern crate clap;

use std::fs::File;
use std::io::prelude::*;
use clap::{Arg, App};


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
}
