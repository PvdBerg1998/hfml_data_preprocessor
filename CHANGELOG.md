# Changelog
Version `2.0.0` is considered as the first "ready for use" release and changes before this point are not documented.

# `2.1.8`
### Changed
- The preliminary plots now use both the column name as well as the variable name in the y label.

# `2.1.7`
### Added
- Additional output stage called `unprocessed`, which is a step before `raw`. This is before the step which makes the data monotonic. This can help avoid artifacts in case your data does not have a 1:1 correspondence between x and y.

# `2.1.6`
Dependencies updated, fixes potential segfault in `chrono`

# `2.1.5`
### Changed
- Identity header replace rules, such as "x" = "x", are now ignored

# `2.1.4`
### Fixed
- Crash when x variable was constant for the entire data file

## `2.1.3`
### Changed
- Updated `gsl_rust` wrapper. Impulse filtering now truncates its window near the domain boundaries. This implies that a polluted zero-field bin, which often contains some random data points caused by acquisition errors, can be filtered away.

## `2.1.2`
### Added
- Extra trace-level logging with FFT frequency and sampling information

### Fixed
- Crash when FFT trim was above Nyquist frequency

## `2.1.1`
### Added
- Separate error messages for missing column or missing header in data files

## `2.1.0`
### Changed
- Renamed `post_interpolation` to `processed` for consistency with `preprocessed` 
- Removed Cargo compilation flags

## `2.0.2`
### Changed
- Use natural string sorting for metadata tags

### Fixed
- Reverted data sorting to heapsort for good worst case performance

## `2.0.1`
### Added
- Best-effort tag sorting based on downcasting
