# Changelog
Version `2.0.0` is considered as the first "ready for use" release and changes before this point are not documented.

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