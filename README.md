# llhd-sim

[![Build Status](https://travis-ci.org/fabianschuiki/llhd-sim.svg?branch=master)](https://travis-ci.org/fabianschuiki/llhd-sim)

This is the reference simulator for [LLHD], striving to be complete but as minimal as possible. Its goal is to serve as a starting point for developing more sophisticated simulators for hardware written in LLHD. As a secondary goal it acts as an application example of LLHD.

[LLHD]: https://github.com/fabianschuiki/llhd


## Roadmap and Milestones

- [x] execute processes
- [x] execute entities
- [ ] execute functions
- [x] unary, binary instructions
- [x] probe, drive instructions
- [ ] call instruction
- [x] sig, inst instructions
- [ ] compare instruction
- [ ] return, branch instructions
- [ ] preserve hierarchy in VCD
- **Milestone:** basic simulator
- [ ] run simulation for limited time
- [ ] set breakpoints in processes/functions
- [ ] set breakpoints in entities
- [ ] step over, step into
- [ ] list source code extract, print values/signals
- [ ] interactive prompt
- **Milestone:** interactive debugger
