# llhd-sim

[![Build Status](https://travis-ci.org/fabianschuiki/llhd-sim.svg?branch=master)](https://travis-ci.org/fabianschuiki/llhd-sim)
[![Crates.io](https://img.shields.io/crates/v/llhd-sim.svg)](https://crates.io/crates/llhd-sim)

This is the reference simulator for [llhd], striving to be complete but as minimal as possible. Its goal is to serve as a starting point for developing more sophisticated simulators for hardware written in llhd. As a secondary goal it acts as an application example of llhd.

## Usage

### Installation

You need a working [Rust installation](https://rustup.rs/). Use cargo to install llhd-sim:

    cargo install llhd-sim

### Example

Given the following input file:

    // foo.llhd
    proc @foo () (i32$ %out) {
    %entry:
        drv %out 0 1ns
        drv %out 42 2ns
        %0 = add i32 9000 1
        drv %out %0 3ns
        halt
    }

Use llhd-sim to simulate the described hardware and produce a VCD file:

    llhd-sim foo.llhd
    gtkwave /tmp/output.vcd

[llhd]: https://github.com/fabianschuiki/llhd
