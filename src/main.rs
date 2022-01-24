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
use anyhow::ensure;
use anyhow::Result;
use clap::Parser;
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
use std::path::PathBuf;
use std::time::Instant;
use std::{
    fs::File,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug, PartialEq, Eq, Parser)]
struct Args {
    settings: PathBuf,
    #[clap(short = 'v')]
    verbose: bool,
}

fn main() {
    gsl_rust::disable_error_handler(); // We handle errors ourselves and this aborts the program.
    if let Err(e) = _main() {
        error!("{e}");
    }
}

fn _main() -> Result<()> {
    let start = Instant::now();

    let args = Args::parse();

    // Register global logger
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let _ = std::fs::create_dir("log");
    let log_level = if args.verbose {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    };
    CombinedLogger::init(vec![
        TermLogger::new(
            log_level,
            ConfigBuilder::default()
                .set_time_format_str("%H:%M:%S.%f")
                .set_thread_mode(ThreadLogMode::Names)
                .set_thread_level(LevelFilter::Error)
                .build(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        ),
        WriteLogger::new(
            LevelFilter::Debug,
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
        settings::load(args.settings).map_err(|e| anyhow!("Failed to load settings: {e}"))?;
    debug!("Settings: {settings:#?}");
    if settings.project.output.is_empty() {
        warn!("Output list is empty: no data will be saved");
    }

    info!("Running project '{}'", settings.project.title);

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

    let end = Instant::now();
    info!("Finished in {} ms", (end - start).as_millis());

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
            ensure!(
                data.contains(x_name),
                "specified x column '{x_name}' for dataset '{name}' does not exist in file {}",
                file.source
            );
            ensure!(
                data.contains(y_name),
                "specified y column '{y_name}' for dataset '{name}' does not exist in file {}",
                file.source
            );

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
    let src = &file.source;
    if !xy.is_finite() {
        warn!("Dataset {src}:'{name}' contains non-finite values");
    }
    let n_log2 = (xy.len() as f64).log2();
    debug!("Dataset {src}:'{name}' length: ~2^{n_log2:.2}");

    // Output raw data
    if settings.project.output.contains(&Output::Raw) {
        debug!("Storing raw data for {src}:'{name}'");
        save(
            &settings.project.title,
            name,
            "raw",
            &file.dest,
            xlabel,
            ylabel,
            settings.project.gnuplot,
            xy.x(),
            xy.y(),
        )?;
    }

    // Premultiplication
    let mx = settings.preprocessing.prefactor.x;
    let my = settings.preprocessing.prefactor.y;
    if mx != 1.0 {
        debug!("Multiplying {src}:'{name}' x by {mx}");
        xy.multiply_x(mx);
    }
    if my != 1.0 {
        debug!("Multiplying {src}:'{name}' y by {my}");
        xy.multiply_y(my);
    }

    // 1/x
    if settings.preprocessing.invert_x {
        debug!("Inverting {src}:'{name}' x");
        xy.invert_x();
    }

    // Monotonicity
    debug!("Making {src}:'{name}' monotonic");
    let mut xy = xy.into_monotonic();

    // Trimming
    let trim_a = if settings.preprocessing.invert_x {
        settings.preprocessing.trim_right.map(|x| 1.0 / x)
    } else {
        settings.preprocessing.trim_left
    }
    .unwrap_or(xy.min_x());
    let trim_b = if settings.preprocessing.invert_x {
        settings.preprocessing.trim_left.map(|x| 1.0 / x)
    } else {
        settings.preprocessing.trim_right
    }
    .unwrap_or(xy.max_x());
    ensure!(
        settings.preprocessing.trim_left < settings.preprocessing.trim_right,
        "Trimmed domain is empty or negative"
    );
    ensure!(
        xy.min_x() <= trim_a,
        "Left trim {trim_a} is outside domain {:?} for {src}:'{name}'",
        xy.min_x()
    );
    ensure!(
        xy.max_x() >= trim_b,
        "Right trim {trim_b} is outside domain {:?} for {src}:'{name}'",
        xy.max_x()
    );
    debug!("Data domain: [{trim_a},{trim_b}]");
    xy.trim(trim_a, trim_b);

    // Output preprocessed data
    if settings.project.output.contains(&Output::PreInterpolation) {
        debug!("Storing pre-interpolation data for {src}:'{name}'");
        save(
            &settings.project.title,
            name,
            "pre interpolation",
            &file.dest,
            xlabel,
            ylabel,
            settings.project.gnuplot,
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
                debug!("Calculating minimum delta x for {src}:'{name}'");

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
                    None => bail!("Unable to determine minimum delta x for {src}:'{name}'"),
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

            info!("Interpolating {src}:'{name}' using {deriv_str} at {n_interp} points using {algorithm} interpolation");
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

    // Output post interpolation data
    if settings.project.output.contains(&Output::PostInterpolation) {
        debug!("Storing post-interpolation data for {src}:'{name}'");
        save(
            &settings.project.title,
            name,
            "post interpolation",
            &file.dest,
            xlabel,
            ylabel,
            settings.project.gnuplot,
            &x,
            &y,
        )?;
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

                debug!("Removing mean value {mean} from {src}:'{name}'");
                y.iter_mut().for_each(|y| {
                    *y -= mean;
                });
            }
            if fft.hann {
                debug!("Applying Hann window to {src}:'{name}'");
                y.iter_mut().enumerate().for_each(|(i, y)| {
                    *y *= (i as f64 * std::f64::consts::PI / (n_data - 1) as f64)
                        .sin()
                        .powi(2);
                });

                // Avoid numerical error and leaking at the edges
                *y.first_mut().unwrap() = 0.0;
                *y.last_mut().unwrap() = 0.0;
            }

            if n_pad > 0 {
                debug!(
                    "Zero padding {src}:'{name}' by {n_pad} to reach 2^{desired_n_log2} points total length ({} MB)",
                    desired_n * std::mem::size_of::<f64>() / 1024usize.pow(2)
                );
                y.resize(desired_n, 0.0);
            }

            // Check power of 2
            // This could be false if we set a padding length smaller than the data length
            let n = y.len();
            ensure!(
                n.is_power_of_two(),
                "FFT is only supported for data length 2^n"
            );

            // Execute FFT
            let y = if fft.cuda {
                if desired_n_log2 < 10 {
                    warn!("Small FFT sizes may be faster when run on a CPU");
                }

                if cufft::gpu_count() == 0 {
                    bail!("No CUDA capable GPUs available");
                }
                info!(
                    "Computing FFT on GPU '{}' for {src}:'{name}'",
                    cufft::first_gpu_name()?
                );
                cufft::fft64_norm(&y)?
            } else {
                info!("Computing FFT on CPU for {src}:'{name}'");
                fft::fft64_packed(&mut y)?;
                fft::fft64_unpack_norm(&y)
            };

            // Prepare frequency space cutoffs
            let lower_cutoff = fft.truncate_lower.unwrap_or(0.0);
            let upper_cutoff = fft.truncate_upper.unwrap_or(f64::INFINITY);
            if lower_cutoff != 0.0 || upper_cutoff.is_finite() {
                debug!("Truncating FFT to {lower_cutoff}..{upper_cutoff} 'Hz' for {src}:'{name}'");
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
                info!("Storing FFT for {src}:'{name}'");
                save(
                    &settings.project.title,
                    name,
                    "fft",
                    &file.dest,
                    "Frequency",
                    "FFT Amplitude",
                    settings.project.gnuplot,
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
    project: &str,
    name: &str,
    title: &str,
    dst: &str,
    xlabel: &str,
    ylabel: &str,
    plot: bool,
    x: &[f64],
    y: &[f64],
) -> Result<()> {
    let sanitized_project = project.replace(' ', "_");
    let sanitized_name = name.replace(' ', "_");
    let sanitized_title = title.replace(' ', "_");
    let sanitized_dst = dst.replace(' ', "_");
    let csv_path = format!(
        "output/{sanitized_project}/{sanitized_name}/{sanitized_title}/{sanitized_dst}.csv"
    );
    let png_path = format!(
        "output/{sanitized_project}/{sanitized_name}/{sanitized_title}/{sanitized_dst}.png"
    );
    let _ = std::fs::create_dir_all(format!(
        "output/{sanitized_project}/{sanitized_name}/{sanitized_title}"
    ));
    output::store_csv(x, y, &csv_path)?;
    if plot {
        output::plot_csv(&csv_path, title, xlabel, ylabel, &png_path)?;
    }
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
