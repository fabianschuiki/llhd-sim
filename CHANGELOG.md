# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
### Added
- Add support for `var`, `load`, `store`, `extract`, `shl`, and `shr` instructions.
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
