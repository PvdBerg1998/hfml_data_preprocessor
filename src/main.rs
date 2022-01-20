/*
    main.rs : hfml_data_preprocessor. A data preprocessor for use with data
    generated during transport measurements at the HFML, Radboud University.
    Copyright (C) 2021 Pim van den Berg

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

//mod maxpol;
mod data;
mod output;
mod settings;

use crate::settings::Processing;
use crate::settings::ProcessingKind;
use anyhow::anyhow;
use anyhow::bail;
use anyhow::Result;
use cufft_rust as cufft;
use data::Data;
use data::XY;
use gsl_rust::fft;
use gsl_rust::interpolation::interpolate_monotonic;
use gsl_rust::interpolation::Derivative;
use gsl_rust::stats;
use log::*;
use rayon::prelude::*;
use settings::{Interpolation, Output, Settings};
use simplelog::*;
use std::{
    fs::File,
    time::{SystemTime, UNIX_EPOCH},
};

fn main() {
    gsl_rust::disable_error_handler(); // We handle errors ourselves and this aborts the program.
    if let Err(e) = _main() {
        error!("{e}");
    }
}

fn _main() -> Result<()> {
    // Register global logger
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let _ = std::fs::create_dir("log");
    CombinedLogger::init(vec![
        TermLogger::new(
            LevelFilter::Debug,
            ConfigBuilder::default()
                .set_time_format_str("%H:%M:%S.%f")
                .set_thread_mode(ThreadLogMode::Names)
                .set_thread_level(LevelFilter::Error)
                .build(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        ),
        WriteLogger::new(
            LevelFilter::Info,
            ConfigBuilder::default()
                .set_time_format_str("%H:%M:%S.%f")
                .set_thread_mode(ThreadLogMode::Names)
                .set_thread_level(LevelFilter::Error)
                .build(),
            File::create(format!("log/hfml_data_preprocessor_{unix}.log"))?,
        ),
    ])?;

    // Parse settings
    let settings =
        settings::load("settings.toml").map_err(|e| anyhow!("Failed to load settings: {e}"))?;
    debug!("Settings: {settings:#?}");
    if settings.project.output.is_empty() {
        warn!("Output list is empty: no data will be saved");
    }

    // Setup global thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(if settings.project.threading { 0 } else { 1 })
        .thread_name(|i| format!("worker {}", i + 1))
        .build_global()?;

    // Process files in parallel
    settings
        .project
        .files
        .par_iter()
        .try_for_each::<_, Result<_>>(|file| {
            info!("Loading file {}", file.source);
            let s = std::fs::read_to_string(&file.source)?;
            process_file(&settings, file, &s)?;
            Ok(())
        })?;

    Ok(())
}

fn process_file(settings: &Settings, file: &settings::File, s: &str) -> Result<()> {
    let mut data: Data = s.parse()?;
    debug!("Columns: {:#?}", data.columns());

    // Rename columns
    for (old, new) in settings.rename.columns.iter() {
        if !data.contains(old) {
            continue;
        }
        if data.contains(new) {
            warn!("Cannot rename {old} to existing column {new}");
            continue;
        }
        debug!("Renaming column {old} to {new}");
        data.rename_column(&*old, &*new);
    }
    debug!("Columns (after replacing): {:#?}", data.columns());

    // Extract everything required to process data pairs
    let parameters = settings
        .extract
        .pairs
        .iter()
        .map(|(name, xy)| {
            info!("Processing file '{}': dataset '{name}'", file.source);

            // Extract x/y names of data pair
            let x_name = xy
                .get("x")
                .ok_or_else(|| anyhow!("Missing x column specification for dataset '{name}'"))?;
            let x_name = x_name.as_str();
            let y_name = xy
                .get("y")
                .ok_or_else(|| anyhow!("Missing y column specification for dataset '{name}'"))?;
            let y_name = y_name.as_str();

            // Check if the given names actually correspond to existing data columns
            if !data.contains(x_name) {
                bail!("specified x column '{x_name}' for dataset '{name}' does not exist");
            }
            if !data.contains(y_name) {
                bail!("specified y column '{y_name}' for dataset '{name}' does not exist");
            }

            // Extract the columns
            info!("Extracting dataset '{name}' (x='{x_name}', y='{y_name}')");
            let xy = data.clone_xy(x_name, y_name);

            // Prepare x/y labels
            let xlabel = if settings.preprocessing.invert_x {
                format!("1/{x_name}")
            } else {
                x_name.to_owned()
            };
            let ylabel = y_name.to_owned();

            Ok((file.to_owned(), name.to_owned(), xlabel, ylabel, xy))
        })
        .collect::<Result<Vec<_>>>()?;

    // Process data pairs in parallel
    // It doesn't matter that we may already be parallelized,
    // as the threadpool uses a work stealing technique
    parameters
        .into_par_iter()
        .try_for_each(|(file, name, xlabel, ylabel, xy)| {
            process_pair(settings, &file, &name, &xlabel, &ylabel, xy)
        })?;

    Ok(())
}

fn process_pair(
    settings: &Settings,
    file: &settings::File,
    name: &str,
    xlabel: &str,
    ylabel: &str,
    mut xy: XY,
) -> Result<()> {
    // Check for NaN and infinities
    if !xy.is_finite() {
        warn!("Dataset '{name}' contains non-finite values");
    }

    // Output raw data
    if settings.project.output.contains(&Output::Raw) {
        info!("Storing raw data");
        save(name, "raw", &file.dest, xlabel, ylabel, xy.x(), xy.y())?;
    }

    // Premultiplication
    let mx = settings.preprocessing.prefactor.x;
    let my = settings.preprocessing.prefactor.y;
    if mx != 1.0 {
        info!("Multiplying x by {mx}");
        xy.multiply_x(mx);
    }
    if my != 1.0 {
        info!("Multiplying y by {my}");
        xy.multiply_y(my);
    }

    // 1/x
    if settings.preprocessing.invert_x {
        info!("Inverting x");
        xy.invert_x();
    }

    // Monotonicity
    info!("Making dataset monotonic");
    let mut xy = xy.into_monotonic();

    // Trimming
    if settings.preprocessing.trim_left >= settings.preprocessing.trim_right {
        bail!("Trimmed domain is empty or negative");
    }
    let (trim_a, trim_b) = if settings.preprocessing.invert_x {
        if settings.preprocessing.trim_left == 0.0 || settings.preprocessing.trim_right == 0.0 {
            bail!("Can't use zero as trim boundary when inverting x");
        }
        let a = 1.0 / settings.preprocessing.trim_right;
        let b = 1.0 / settings.preprocessing.trim_left;
        (a, b)
    } else {
        let a = settings.preprocessing.trim_left;
        let b = settings.preprocessing.trim_right;
        (a, b)
    };
    if xy.min_x() > trim_a {
        bail!(
            "Left trim {trim_a} is outside domain {:?} for dataset '{name}'",
            xy.min_x()
        );
    }
    if xy.max_x() < trim_b {
        bail!(
            "Right trim {trim_b} is outside domain {:?} for dataset '{name}'",
            xy.max_x()
        );
    }
    info!("Trimming domain to [{trim_a},{trim_b}]");
    xy.trim(trim_a, trim_b);

    // Output preprocessed data
    if settings.project.output.contains(&Output::PreInterpolation) {
        info!("Storing pre-interpolation data");
        save(
            name,
            "pre interpolation",
            &file.dest,
            xlabel,
            ylabel,
            xy.x(),
            xy.y(),
        )?;
    }

    // Interpolation
    let (x, mut y) = match &settings.preprocessing.interpolation {
        Some(Interpolation { n, algorithm }) => {
            // Parse and process one of:
            // - "n"
            // - log2("2^n")
            // - "min": use minimum dx in dataset
            let n_interp = if n == "min" {
                debug!("Calculating minimum delta x");

                // Loop over x, compare neighbours, find smallest interval
                match xy
                    .x()
                    .windows(2)
                    .map(|window| {
                        let (a, b) = (window[0], window[1]);
                        (b - a).abs()
                    })
                    .min_by_key(|&dx| float_ord::FloatOrd(dx))
                {
                    Some(dx) => {
                        let n = xy.domain_len() / dx;
                        n.ceil() as _
                    }
                    None => bail!("Unable to determine minimum delta x"),
                }
            } else {
                match n.parse::<u64>() {
                    Ok(n) => n,
                    Err(_) => 2u64.pow(parse_log2(n)?),
                }
            };

            let deriv = match settings.preprocessing.derivative {
                0 => Derivative::None,
                1 => Derivative::First,
                2 => Derivative::Second,
                _ => bail!("Only the 0th, 1st and 2nd derivative are supported"),
            };
            let deriv_str = match deriv {
                Derivative::None => "function",
                Derivative::First => "1st derivative",
                Derivative::Second => "2nd derivative",
            };

            info!("Interpolating {deriv_str} at {n_interp} points using {algorithm} interpolation");
            let dx = xy.domain_len() / n_interp as f64;
            let x_eval = (0..n_interp)
                .map(|i| (i as f64) * dx + xy.min_x())
                .collect::<Box<[_]>>();
            let y_eval =
                interpolate_monotonic((*algorithm).into(), deriv, xy.x(), xy.y(), &x_eval)?;
            drop(xy);
            (Vec::from(x_eval), Vec::from(y_eval))
        }
        None => xy.take_xy(),
    };

    // Output preprocessed data
    if settings.project.output.contains(&Output::Preprocessed) {
        info!("Storing preprocessed data");
        save(name, "preprocessed", &file.dest, xlabel, ylabel, &x, &y)?;
    }

    // Processing
    match &settings.processing {
        Some(Processing {
            kind: ProcessingKind::Fft,
            fft: Some(fft),
        }) => {
            // Parse and process one of:
            // - "n"
            // - log2("2^n")
            // - "min": use minimum padding to make the length a power of 2
            let desired_n_log2 = if fft.zero_pad == "min" {
                (y.len() as f64).log2().ceil() as u32
            } else {
                match fft.zero_pad.parse::<u32>() {
                    Ok(n) => n,
                    Err(_) => parse_log2(&fft.zero_pad)?,
                }
            };
            let desired_n = 2usize.pow(desired_n_log2);

            // Amount of nonzero points
            let n_data = y.len();
            // Amount of padded zero points
            let n_pad = desired_n.saturating_sub(n_data);

            // FFT preprocessing: centering, windowing, padding
            if fft.center {
                let mean = stats::mean(&y);

                info!("Removing mean value {mean}");
                y.iter_mut().for_each(|y| {
                    *y -= mean;
                });
            }
            if fft.hann {
                info!("Applying Hann window");
                y.iter_mut().enumerate().for_each(|(i, y)| {
                    *y *= (i as f64 * std::f64::consts::PI / n_data as f64)
                        .sin()
                        .powi(2);
                });
            }
            if n_pad > 0 {
                info!(
                    "Zero padding by {n_pad} to reach 2^{desired_n_log2} points total length ({} MB)",
                    desired_n * std::mem::size_of::<f64>() / 1024usize.pow(2)
                );
                y.resize(desired_n, 0.0);
            }

            // Check power of 2
            // This could be false if we set a padding length smaller than the data length
            let n = y.len();
            if !n.is_power_of_two() {
                bail!("FFT is only supported for data length 2^n");
            }

            // Execute FFT
            let y = if fft.cuda {
                if desired_n_log2 < 10 {
                    warn!("Small FFT sizes may be faster when run on a CPU");
                }

                if cufft::gpu_count() == 0 {
                    bail!("No CUDA capable GPUs available");
                }
                info!("Computing FFT on GPU '{}'", cufft::first_gpu_name()?);
                cufft::fft64_norm(&y)?
            } else {
                info!("Computing FFT on CPU");
                fft::fft64_packed(&mut y)?;
                fft::fft64_unpack_norm(&y)
            };

            // Prepare frequency space cutoffs
            let lower_cutoff = fft.truncate_lower.unwrap_or(0.0);
            let upper_cutoff = fft.truncate_upper.unwrap_or(f64::INFINITY);
            if lower_cutoff != 0.0 || upper_cutoff.is_finite() {
                info!("Truncating FFT to {lower_cutoff}..{upper_cutoff} 'Hz'");
            }

            // Sampled frequencies : k/(N dt)
            // Note: this includes zero padding.
            let dt = x[1] - x[0];
            let x = (0..y.len())
                .map(|i| i as f64 / (dt * n as f64))
                .collect::<Box<[f64]>>();
            let start_idx = ((lower_cutoff * dt * n as f64).ceil() as usize).min(y.len());
            let end_idx = ((upper_cutoff * dt * n as f64).ceil() as usize).min(y.len());

            // Output FFT
            if settings.project.output.contains(&Output::Processed) {
                info!("Storing FFT");
                save(
                    name,
                    "fft",
                    &file.dest,
                    "Frequency",
                    "FFT Amplitude",
                    &x[start_idx..end_idx],
                    &y[start_idx..end_idx],
                )?;
            }
        }
        Some(Processing {
            kind: ProcessingKind::Fft,
            fft: None,
        }) => bail!("Missing FFT settings"),
        None => {}
    }

    Ok(())
}

fn save(
    name: &str,
    title: &str,
    dst: &str,
    xlabel: &str,
    ylabel: &str,
    x: &[f64],
    y: &[f64],
) -> Result<()> {
    let sanitized_name = name.replace(' ', "_");
    let sanitized_title = title.replace(' ', "_");
    let sanitized_dst = dst.replace(' ', "_");
    let csv_path = format!("output/{sanitized_name}/{sanitized_title}/{sanitized_dst}.csv");
    let png_path = format!("output/{sanitized_name}/{sanitized_title}/{sanitized_dst}.png");
    let _ = std::fs::create_dir(format!("output/{sanitized_name}/{sanitized_title}"));
    output::store_csv(x, y, &csv_path)?;
    output::plot_csv(&csv_path, title, xlabel, ylabel, &png_path)?;
    Ok(())
}

/// Parses `log2("2^n")`.
///
/// Rounds up to the nearest power of 2.
fn parse_log2(n: &str) -> Result<u32> {
    if let Some(stripped) = n.strip_prefix("2^") {
        match stripped.parse::<u32>() {
            Ok(n_log2) => Ok(n_log2),
            Err(_) => {
                bail!("Invalid power of 2: {n}");
            }
        }
    } else {
        bail!("Invalid length: {n}");
    }
}
