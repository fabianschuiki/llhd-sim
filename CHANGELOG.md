# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## 0.4.0 - 2020-08-26
### Added
- Add human-readable change dump.
- Add `-t` option to simulate until a fixed number of steps have elapsed.
- Make output file optional.

### Changed
- Moved the entire codebase into the [llhd](https://github.com/fabianschuiki/llhd) repository.
- Update llhd to v0.9.0.

## 0.3.1 - 2019-03-17
### Fixed
- Make entities sensitive to output signals. ([#4](https://github.com/fabianschuiki/llhd-sim/issues/4))

## 0.3.0 - 2019-03-12
### Added
- Add support for `var`, `load`, `store`, `extract`, `insert`, `shl`, and `shr` instructions.
- Add `--sequential` option to disable parallelization.

### Changed
- Update llhd to v0.5.0.
- Improve command line help page.

### Fixed
- Structs and arrays are now properly unrolled in VCD output. ([#3](https://github.com/fabianschuiki/llhd-sim/issues/3))

## 0.2.0 - 2019-01-13
### Added
- Add `-v` verbosity option.
- Add `stderrlog` and `log` dependencies.
- Support branch instructions.
- Support comparison instructions.

### Fixed
- Preserve process/entity hierarchy in VCD output.

## 0.1.0 - 2019-01-05
### Added
- Initial release.
