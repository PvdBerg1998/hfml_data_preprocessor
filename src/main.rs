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

use crate::settings::*;
use anyhow::anyhow;
use anyhow::bail;
use anyhow::ensure;
use anyhow::Result;
use clap::Parser;
use cufft_rust as cufft;
use data::Data;
use data::MonotonicXY;
use data::XY;
use gsl_rust::fft;
use gsl_rust::interpolation::interpolate_monotonic;
use gsl_rust::interpolation::Derivative;
use gsl_rust::stats;
use itertools::Itertools;
use log::*;
use rayon::prelude::*;
use settings::{Settings, Template};
use simplelog::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
use std::{
    fs::File,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug, PartialEq, Eq, Parser)]
struct Args {
    template: Option<PathBuf>,
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

    // Parse template
    let template = Template::load(
        args.template
            .unwrap_or_else(|| PathBuf::from("settings.toml")),
    )
    .map_err(|e| anyhow!("Failed to load template: {e}"))?;

    debug!("Template: {template:#?}");
    info!("Running project '{}'", template.project.title);

    // Setup global thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(if template.project.threading { 0 } else { 1 })
        .thread_name(|i| format!("worker {}", i + 1))
        .build_global()?;

    // Load files
    let data = template
        .files
        .par_iter()
        .map(|file| {
            info!("Loading file {}", file.source);
            let s = std::fs::read_to_string(&file.source)?;
            let mut data = s.parse::<Data>()?;

            // Rename columns
            debug!("Columns: {:#?}", data.columns());
            for Rename { from, to } in template.rename.iter() {
                if !data.contains(from) {
                    continue;
                }
                ensure!(
                    !data.contains(to),
                    "cannot rename {from} to existing column {to}"
                );

                debug!("Renaming column {from} to {to}");
                data.rename_column(&*from, &*to);
            }
            debug!("Columns (after replacing): {:#?}", data.columns());

            Ok((file.source.as_str(), data))
        })
        .collect::<Result<HashMap<_, _>>>()?;

    let Template {
        project,
        fft,
        settings,
        ..
    } = template;

    // Prepare and preprocess data in parallel
    let preprocessed = settings
        .par_iter()
        .map(|settings| {
            let data = &data[settings.file.source.as_str()];
            settings.prepare(data, &project)?.preprocess()
        })
        .collect::<Result<Vec<_>>>()?;

    // Calculate minimum dx based interpolation amounts per variable per file
    let n_per_var = preprocessed
        .par_iter()
        .filter(|preprocessed| {
            // If we are not interpolating this data pair,
            // it shouldn't contribute to the stats.
            // It also saves time to skip this step if it's not required
            preprocessed.settings.processing.interpolation.is_some()
        })
        .filter(
            // See filter above
            |preprocessed| match preprocessed.settings.processing.interpolation_n.as_deref() {
                Some(INTERP_OPTION_MIN_VAR) | Some(INTERP_OPTION_MIN) => true,
                _ => false,
            },
        )
        .map(|preprocessed| {
            let name = &preprocessed.settings.extract.name;
            let src = preprocessed.settings.file.source.as_str();
            debug!("Calculating minimum delta x for all {src}:'{name}'");

            // Loop over x, compare neighbours, find smallest interval
            let n = match preprocessed
                .xy
                .x()
                .windows(2)
                .map(|window| {
                    let (a, b) = (window[0], window[1]);
                    (b - a).abs()
                })
                .min_by_key(|&dx| float_ord::FloatOrd(dx))
            {
                Some(dx) => {
                    let n = preprocessed.xy.domain_len() / dx;
                    n.ceil() as u64
                }
                None => bail!("Unable to determine minimum delta x for {src}:'{name}'"),
            };
            Ok((name.as_str(), n))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .into_group_map();

    // Calculate statistics and find the highest interpolation amount
    let max_n_per_var = n_per_var.into_iter().map(|(var, ns)| {
        let float_ns = ns.iter().map(|&n| n as f64).collect::<Vec<_>>();
        let n_mean = stats::mean(&float_ns);
        let n_stddev = stats::variance_mean(&float_ns, n_mean).sqrt();
        let n_mean_log2 = n_mean.log2().ceil() as u64;
        debug!("Minimum dx interpolation stats for {var}: mean = {n_mean} ~ 2^{n_mean_log2}, stddev = {n_stddev:.2}");

        // Store the highest amount
        (var, ns.into_iter().max().unwrap())
    }).collect::<HashMap<_, _>>();
    let max_n = max_n_per_var.values().max().copied();
    debug!("Final dx interpolation maxima: {max_n_per_var:#?}. Maximum: {max_n:#?}");

    // Process data
    let processed = preprocessed
        .into_par_iter()
        .map(|preprocessed| {
            let n_interp_global = max_n;
            let n_interp_var = max_n_per_var
                .get(preprocessed.settings.extract.name.as_str())
                .copied();
            preprocessed.process(n_interp_global, n_interp_var)
        })
        .collect::<Result<Vec<_>>>()?;

    // FFT
    if let Some(fft) = fft {
        let prepared = processed
            .into_par_iter()
            .map(|processed| processed.prepare_fft(&fft))
            .collect::<Result<Vec<_>>>()?;

        let use_cuda = if fft.cuda {
            if cufft::gpu_count() == 0 {
                warn!("No CUDA capable GPUs available, using CPU instead");
                false
            } else {
                info!(
                    "Computing batched FFTs on GPU '{}'",
                    cufft::first_gpu_name()?
                );
                true
            }
        } else {
            false
        };

        if use_cuda {
            let batch = prepared
                .iter()
                .map(|prepared| prepared.y.as_slice())
                .collect::<Vec<_>>();
            let fft = cufft::fft64_norm_batch(&batch)?;
            prepared
                .into_par_iter()
                .zip(fft)
                .try_for_each(|(prepared, fft)| prepared.finish(fft))?;
        } else {
            prepared
                .into_par_iter()
                .try_for_each(|mut prepared| -> Result<()> {
                    let name = &prepared.settings.extract.name;
                    let src = prepared.settings.file.source.as_str();

                    info!("Computing FFT on CPU for {src}:'{name}'");
                    fft::fft64_packed(&mut prepared.y)?;
                    let fft = fft::fft64_unpack_norm(&prepared.y);

                    prepared.finish(fft)
                })?;
        };
    }

    let end = Instant::now();
    info!("Finished in {} ms", (end - start).as_millis());

    Ok(())
}

impl Settings {
    fn prepare(&self, data: &Data, project: &Project) -> Result<Prepared<'_>> {
        let settings = self;
        let Extract { name, x, y } = &settings.extract;
        let src = settings.file.source.as_str();

        info!("Preparing file '{src}': dataset '{name}'");

        // Check if the given names actually correspond to existing data columns
        ensure!(
            data.contains(x),
            "specified x column '{x}' for dataset '{name}' does not exist in file '{src}'"
        );
        ensure!(
            data.contains(y),
            "specified y column '{y}' for dataset '{name}' does not exist in file '{src}'"
        );

        // Extract the columns
        info!("Extracting dataset '{name}' (x='{x}', y='{y}')");
        let xy = data.clone_xy(x, y);

        // Check for NaN and infinities
        if !xy.is_finite() {
            bail!("File '{src}':'{name}' contains non-finite values");
        }

        // Prepare labels
        let title = project.title.clone();
        let x_label = x.to_owned();
        let y_label = y.to_owned();

        Ok(Prepared {
            settings,
            title,
            x_label,
            y_label,
            xy,
        })
    }
}

impl<'a> Prepared<'a> {
    fn preprocess(self) -> Result<Preprocessed<'a>> {
        let Self {
            settings,
            title,
            x_label,
            y_label,
            xy,
        } = self;

        let Extract { name, x, y: _ } = &settings.extract;
        let src = settings.file.source.as_str();

        info!("Preprocessing file '{src}': dataset '{name}'");

        let n_log2 = (xy.len() as f64).log2();
        debug!("Dataset {src}:'{name}' length: ~2^{n_log2:.2}");

        // Monotonicity
        debug!("Making {src}:'{name}' monotonic");
        let mut xy = xy.into_monotonic();

        // Output raw data
        debug!("Storing raw data for {src}:'{name}'");
        save(
            &title,
            name,
            "raw",
            &settings.file.dest,
            &x_label,
            &y_label,
            xy.x(),
            xy.y(),
        )?;

        // Impulse filtering
        // Do this before trimming, such that edge artifacts may be cut off afterwards
        if settings.preprocessing.impulse_filter > 0 {
            let width = settings.preprocessing.impulse_filter;
            let tuning = settings.preprocessing.impulse_tuning;

            ensure!(
                tuning >= 0.0,
                "Impulse filter with negative tuning makes no sense"
            );

            debug!(
                "Applying impulse filter of width {width} and tuning {tuning} to {src}:'{name}'"
            );
            xy.impulse_filter(width as usize, tuning);
        }

        // Masking
        for &Mask { left, right } in &settings.preprocessing.masks {
            ensure!(
                left >= xy.left_x(),
                "Mask {left} is outside domain {:?} for {src}:'{name}'",
                xy.left_x()
            );
            ensure!(
                right <= xy.right_x(),
                "Mask {right} is outside domain {:?} for {src}:'{name}'",
                xy.right_x()
            );

            debug!("Masking {src}:'{name}' from {} to {}", left, right);
            xy.mask(left, right);
        }

        // Trimming
        let mut trim_left = settings
            .preprocessing
            .trim_left
            .unwrap_or_else(|| xy.left_x());
        let mut trim_right = settings
            .preprocessing
            .trim_right
            .unwrap_or_else(|| xy.right_x());
        ensure!(
            trim_left < trim_right,
            "Trimmed domain is empty or negative"
        );

        // Automagically swap trim sign if needed to deal with negative x data
        if xy.right_x() <= 0.0 && trim_right >= 0.0 {
            debug!("Flipping trim domain around zero for {src}:'{name}'");
            std::mem::swap(&mut trim_left, &mut trim_right);
            trim_left *= -1.0;
            trim_right *= -1.0;
        }

        ensure!(
            xy.left_x() <= trim_left,
            "Left trim {trim_left} is outside domain {:?} for {src}:'{name}'",
            xy.left_x()
        );
        ensure!(
            xy.right_x() >= trim_right,
            "Right trim {trim_right} is outside domain {:?} for {src}:'{name}'",
            xy.right_x()
        );
        debug!("Data domain: [{trim_left},{trim_right}]");
        xy.trim(trim_left, trim_right);

        // Premultiplication
        // Use local override prefactor if set
        let mx = settings.preprocessing.prefactor_x;
        let my = settings.preprocessing.prefactor_y;
        if mx != 1.0 {
            debug!("Multiplying {src}:'{name}' x by {mx}");
            xy.multiply_x(mx);
        }
        if my != 1.0 {
            debug!("Multiplying {src}:'{name}' y by {my}");
            xy.multiply_y(my);
        }

        // Output preprocessed data
        debug!("Storing preprocessed data for {src}:'{name}'");
        save(
            &title,
            name,
            "preprocessed",
            &settings.file.dest,
            &x_label,
            &y_label,
            xy.x(),
            xy.y(),
        )?;

        // 1/x
        if settings.preprocessing.invert_x {
            debug!("Inverting {src}:'{name}' x");
            xy.invert_x();

            // Output inverted data
            debug!("Storing inverted data for {src}:'{name}'");
            save(
                &title,
                name,
                "inverted",
                &settings.file.dest,
                &x_label,
                &y_label,
                xy.x(),
                xy.y(),
            )?;
        }

        let x_label = if settings.preprocessing.invert_x {
            format!("1/{x}")
        } else {
            x_label
        };

        Ok(Preprocessed {
            settings,
            title,
            x_label,
            y_label,
            xy,
        })
    }
}

impl<'a> Preprocessed<'a> {
    fn process(
        self,
        n_interp_global: Option<u64>,
        n_interp_var: Option<u64>,
    ) -> Result<Processed<'a>> {
        let Self {
            settings,
            title,
            x_label,
            y_label,
            xy,
        } = self;
        let name = &settings.extract.name;
        let src = settings.file.source.as_str();

        info!("Processing file '{src}': dataset '{name}'");

        // Interpolation
        let (x, y) = match (
            &settings.processing.interpolation,
            &settings.processing.interpolation_n,
        ) {
            (Some(algorithm), Some(n)) => {
                // Parse and process one of:
                // - "n"
                // - log2("2^n")
                // - "minvar"
                // - "min"
                let n_interp = match n.as_str() {
                    INTERP_OPTION_MIN_VAR => {
                        n_interp_var.expect("should have calculated n_interp_var")
                    }
                    INTERP_OPTION_MIN => {
                        n_interp_global.expect("should have calculated n_interp_global")
                    }
                    n => match n.parse::<u64>() {
                        Ok(n) => n,
                        Err(_) => 2u64.pow(parse_log2(n)?),
                    },
                };

                let deriv = match settings.processing.derivative {
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
                    .map(|i| (i as f64) * dx + xy.left_x())
                    .collect::<Box<[_]>>();
                let y_eval =
                    interpolate_monotonic((*algorithm).into(), deriv, xy.x(), xy.y(), &x_eval)?;

                (Vec::from(x_eval), y_eval)
            }
            (Some(_), None) => {
                bail!("Missing interpolation length specification");
            }
            (None, Some(_)) | (None, None) => xy.take_xy(),
        };

        // Output post interpolation data
        debug!("Storing post-interpolation data for {src}:'{name}'");
        save(
            &title,
            name,
            "post interpolation",
            &settings.file.dest,
            &x_label,
            &y_label,
            &x,
            &y,
        )?;

        Ok(Processed {
            settings,
            title,
            x_label,
            y_label,
            x,
            y,
        })
    }
}

impl<'a> Processed<'a> {
    fn prepare_fft(self, fft_settings: &'a Fft) -> Result<PreparedFft<'a>> {
        let Self {
            settings,
            title,
            x_label,
            y_label,
            x,
            mut y,
        } = self;
        let name = &settings.extract.name;
        let src = settings.file.source.as_str();

        info!("Preparing file '{src}': dataset '{name}' for FFT");

        // FFT
        // Parse and process one of:
        // - "n"
        // - log2("2^n")
        // - "min": use minimum padding to make the length a power of 2
        let desired_n_log2 = if fft_settings.zero_pad == "min" {
            (y.len() as f64).log2().ceil() as u32
        } else {
            match fft_settings.zero_pad.parse::<u32>() {
                Ok(n) => n,
                Err(_) => parse_log2(&fft_settings.zero_pad)?,
            }
        };
        let desired_n = 2usize.pow(desired_n_log2);

        // Amount of nonzero points
        let n_data = y.len();
        // Amount of padded zero points
        let n_pad = desired_n.saturating_sub(n_data);
        debug_assert_eq!(n_data + n_pad, desired_n);

        // FFT preprocessing: centering, windowing, padding
        if fft_settings.center {
            let mean = stats::mean(&y);

            debug!("Removing mean value {mean} from {src}:'{name}'");
            y.iter_mut().for_each(|y| {
                *y -= mean;
            });
        }
        if fft_settings.hann {
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

        Ok(PreparedFft {
            settings,
            fft_settings,
            title,
            x_label,
            y_label,
            n_data,
            n_pad,
            x,
            y,
        })
    }
}

impl<'a> PreparedFft<'a> {
    fn finish(self, mut fft: Vec<f64>) -> Result<()> {
        let Self {
            settings,
            fft_settings,
            title,
            x_label: _,
            y_label: _,
            n_data,
            n_pad,
            x,
            y: _,
        } = self;
        let name = &settings.extract.name;
        let src = settings.file.source.as_str();

        info!("FFT postprocessing file '{src}': dataset '{name}'");

        // Normalize
        debug!("Normalizing FFT by 1/{n_data}");
        let normalisation = (1.0 / n_data as f64).powi(2);
        fft.iter_mut().for_each(|y| {
            *y *= normalisation;
        });

        // Prepare frequency space cutoffs
        let lower_cutoff = fft_settings.truncate_lower.unwrap_or(0.0);
        let upper_cutoff = fft_settings.truncate_upper.unwrap_or(f64::INFINITY);
        if lower_cutoff != 0.0 || upper_cutoff.is_finite() {
            debug!("Truncating FFT to {lower_cutoff}..{upper_cutoff} 'Hz' for {src}:'{name}'");
        }

        // Sampled frequencies : k/(N dt)
        // Note: this includes zero padding.
        let n = n_data + n_pad;
        let dt = x[1] - x[0];
        let x = (0..fft.len())
            .map(|i| i as f64 / (dt * n as f64))
            .collect::<Box<[f64]>>();

        let start_idx = ((lower_cutoff * dt * n as f64).ceil() as usize).min(fft.len());
        let end_idx = ((upper_cutoff * dt * n as f64).ceil() as usize).min(fft.len());

        // Output FFT
        info!("Storing FFT for {src}:'{name}'");
        save(
            &title,
            name,
            "fft",
            &settings.file.dest,
            "Frequency",
            "FFT Amplitude",
            &x[start_idx..end_idx],
            &fft[start_idx..end_idx],
        )?;

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Prepared<'a> {
    settings: &'a Settings,
    title: String,
    x_label: String,
    y_label: String,
    xy: XY,
}

#[derive(Clone, Debug, PartialEq)]
struct Preprocessed<'a> {
    settings: &'a Settings,
    title: String,
    x_label: String,
    y_label: String,
    xy: MonotonicXY,
}

#[derive(Clone, Debug, PartialEq)]
struct Processed<'a> {
    settings: &'a Settings,
    title: String,
    x_label: String,
    y_label: String,
    x: Vec<f64>,
    y: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
struct PreparedFft<'a> {
    settings: &'a Settings,
    fft_settings: &'a Fft,
    title: String,
    x_label: String,
    y_label: String,
    n_data: usize,
    n_pad: usize,
    x: Vec<f64>,
    y: Vec<f64>,
}

fn save(
    project: &str,
    name: &str,
    title: &str,
    dst: &str,
    x_label: &str,
    y_label: &str,
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
    output::plot_csv(&csv_path, title, x_label, y_label, &png_path)?;
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

pub fn has_dup<T: PartialEq>(slice: &[T]) -> bool {
    for i in 1..slice.len() {
        if slice[i..].contains(&slice[i - 1]) {
            return true;
        }
    }
    false
}
