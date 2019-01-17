// Copyright (c) 2017 Fabian Schuiki
#[macro_use]
extern crate log;

pub mod builder;
pub mod state;
// pub mod worker;
pub mod engine;
pub mod tracer;

use crate::engine::Engine;
use crate::tracer::{Tracer, VcdTracer};
use clap::{App, Arg};
use std::fs::File;
use std::io::prelude::*;

fn main() {
    // Parse the command line arguments.
    let matches = App::new("llhd-sim")
        .version(clap::crate_version!())
        .author(clap::crate_authors!())
        .about(clap::crate_description!())
        .arg(
            Arg::with_name("verbosity")
                .short("v")
                .multiple(true)
                .help("Increase message verbosity"),
        )
        .arg(
            Arg::with_name("INPUT")
                .help("The input file to simulate")
                .required(true)
                .index(1),
        )
        .get_matches();

    // Configure the logger.
    stderrlog::new()
        .quiet(!matches.is_present("verbosity"))
        .verbosity(matches.occurrences_of("verbosity") as usize + 1)
        .init()
        .unwrap();

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
    llhd::assembly::write(&mut std::io::stdout().lock(), &module);

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
