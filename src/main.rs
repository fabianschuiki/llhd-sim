// Copyright (c) 2017-2019 Fabian Schuiki
#[macro_use]
extern crate log;

use anyhow::{anyhow, Context, Result};
use clap::{App, Arg};
use std::{fs::File, io::prelude::*};

// pub mod builder;
// pub mod state;
// pub mod worker;
// pub mod engine;
// pub mod tracer;

// use crate::engine::Engine;
// use crate::tracer::{Tracer, VcdTracer};

fn main() -> Result<()> {
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
            Arg::with_name("sequential")
                .short("s")
                .long("sequential")
                .help("Disable parallelization"),
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
        // Open the input file.
        let path = matches.value_of("INPUT").unwrap();
        let mut contents = String::new();
        File::open(path)
            .and_then(|mut f| f.read_to_string(&mut contents))
            .with_context(|| format!("failed to read input from {}", path))?;

        // Parse the input file.
        let module = llhd::assembly::parse_module(&contents)
            .map_err(|e| anyhow!("{}", e))
            .with_context(|| format!("failed to parse input from {}", path))?;

        // Verify the file for integrity.
        let mut verifier = llhd::verifier::Verifier::new();
        verifier.verify_module(&module);
        verifier
            .finish()
            .map_err(|e| anyhow!("{}", e))
            .with_context(|| format!("failed to verify input from {}", path))?;
    };

    // // Build the simulation state for this module.
    // let mut state = builder::build(&module);

    // // Create a new tracer for this state that will generate some waveforms.
    // let mut file = File::create("/tmp/output.vcd").unwrap();
    // let mut tracer = VcdTracer::new(&mut file);
    // tracer.init(&state);

    // // Create the simulation engine and run the simulation to completion.
    // {
    //     let mut engine = Engine::new(&mut state, !matches.is_present("sequential"));
    //     engine.run(&mut tracer);
    // }

    // // Flush the tracer.
    // tracer.finish(&state);

    Ok(())
}
