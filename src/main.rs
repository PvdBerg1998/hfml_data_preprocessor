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

mod data;
mod output;
mod settings;

mod built_info {
    include!(concat!(env!("OUT_DIR"), "/built.rs"));
}

use crate::settings::*;
use anyhow::bail;
use anyhow::ensure;
use anyhow::Context;
use anyhow::Result;
use clap::Parser;
#[cfg(feature = "cuda")]
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
use num_complex::Complex;
use num_traits::AsPrimitive;
use num_traits::Float;
use rayon::prelude::*;
use serde::Serialize;
use serde_json as json;
use settings::{Settings, Template};
use simplelog::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use std::{
    fs::File,
    time::{SystemTime, UNIX_EPOCH},
};

// The amount of dx samples we use to calculate the minimum dx of a dataset
const INTERP_MIN_DX_SAMPLES: usize = 5;

#[derive(Clone, Debug, PartialEq, Eq, Parser)]
struct Args {
    template: PathBuf,
    #[clap(short, long, parse(from_occurrences))]
    verbose: usize,
    #[clap(short, long)]
    quiet: bool,
}

fn main() {
    gsl_rust::disable_error_handler(); // We handle errors ourselves and this aborts the program.

    if let Err(e) = _main() {
        error!("{e:?}");
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

    let log_level = if args.quiet {
        LevelFilter::Warn
    } else {
        match args.verbose {
            0 => LevelFilter::Info,
            1 => LevelFilter::Debug,
            _ => LevelFilter::Trace,
        }
    };

    let log_path = format!("log/hfml_data_preprocessor_{unix}.log");
    CombinedLogger::init(vec![
        TermLogger::new(
            log_level,
            ConfigBuilder::default()
                .set_time_format_str("%H:%M:%S.%f")
                .set_thread_mode(ThreadLogMode::Names)
                .set_thread_level(LevelFilter::Error)
                .set_location_level(LevelFilter::Off)
                .build(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        ),
        WriteLogger::new(
            LevelFilter::Trace,
            ConfigBuilder::default()
                .set_time_format_str("%H:%M:%S.%f")
                .set_thread_mode(ThreadLogMode::Names)
                .set_thread_level(LevelFilter::Error)
                .set_location_level(LevelFilter::Off)
                .build(),
            File::create(&log_path).context("Failed to open log file")?,
        ),
    ])
    .unwrap();

    let version = format!(
        "{} {}, built at {} for {} by {}",
        built_info::PKG_VERSION,
        if cfg!(feature = "cuda") {
            "with CUDA support"
        } else {
            "without CUDA support"
        },
        built_info::BUILT_TIME_UTC,
        built_info::TARGET,
        built_info::RUSTC_VERSION
    );
    info!("Running hfml_data_preprocessor version {version}");

    // Parse template
    let template = Template::load(args.template).context("Failed to load template")?;

    trace!("Template: {template:#?}");
    info!("Running project '{}'", template.project.title);

    // Setup global thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(if template.project.threading { 0 } else { 1 })
        .thread_name(|i| format!("worker {}", i + 1))
        .build_global()
        .context("Failed to start threadpool")?;

    if template.files.is_empty() {
        bail!("No files listed in template");
    }

    // Load files
    let data = template
        .files
        .clone()
        .into_par_iter()
        .map(|file| {
            debug!("Loading file {}", file.source);
            let s = std::fs::read_to_string(&file.source)
                .context(format!("Failed to load file {}", file.source))?;
            let mut data = s
                .parse::<Data>()
                .context(format!("Failed to parse file {}", file.source))?;

            // Rename columns
            trace!("Columns: {:#?}", data.columns());
            for Rename { from, to } in template.rename.iter() {
                if !data.contains(from) {
                    continue;
                }
                ensure!(
                    !data.contains(to),
                    "Cannot rename {from} to existing column {to}"
                );

                trace!("Renaming column {from} to {to}");
                data.rename_column(&*from, &*to);
            }
            trace!("Columns (after replacing): {:#?}", data.columns());

            Ok((file.source, data))
        })
        .collect::<Result<HashMap<_, _>>>()
        .context("Failed to load data files")?;

    let Template {
        project,
        fft,
        settings,
        files,
        extract,
        rename: _,
    } = template;

    // Dumping area for all save records
    // We just guesstimate the storage required: 5 types of output for a normal FFT
    let mut save_records = Vec::with_capacity(settings.len() * 5);

    // Prepare and preprocess data in parallel
    let preprocessed = settings
        .clone()
        .into_par_iter()
        .map(Arc::new)
        .filter_map(|settings| {
            let data = &data[settings.file.source.as_str()];
            // None's are ignored: they correspond to missing columns which is treated as a soft error
            settings.prepare(data, project.clone()).transpose()
        })
        .map(|prepared| prepared.and_then(|prepared| prepared.preprocess()))
        .collect::<Result<Vec<_>>>()
        .context("Failed to preprocess data")?
        .into_iter()
        .map(|(preprocessed, saves)| {
            save_records.extend(saves);
            preprocessed
        })
        .collect::<Vec<_>>();

    // Calculate minimum dx based interpolation amounts per variable per file
    let n_per_var = preprocessed
        .iter()
        .filter(|preprocessed| {
            // If we are not interpolating this data pair,
            // it shouldn't contribute to the stats.
            // It also saves time to skip this step if it's not required
            preprocessed.settings.processing.interpolation.is_some()
        })
        .filter(
            // See filter above
            // Count only the ones participating in automatic dx interpolation
            |preprocessed| match preprocessed
                .settings
                .processing
                .interpolation
                .map(|algorithm| algorithm.length())
            {
                Some(InterpolationLength::Minimum)
                | Some(InterpolationLength::MinimumPerVariable) => true,
                _ => false,
            },
        )
        .map(|preprocessed| {
            let name = &preprocessed.settings.extract.name;
            let src = preprocessed.settings.file.source.as_str();
            trace!("Calculating minimum delta x for all '{src}':'{name}'");

            // Loop over x, compare neighbours, sort
            let mut dx = preprocessed
                .xy
                .x()
                .windows(2)
                .map(|window| {
                    let (a, b) = (window[0], window[1]);
                    (b - a).abs()
                })
                .map(float_ord::FloatOrd)
                .collect::<Vec<_>>();
            dx.sort();

            // Take the median of the smallest N intervals
            // This is done to avoid using a spurious extremely small interval
            let median_dx = dx
                .get(INTERP_MIN_DX_SAMPLES / 2)
                .unwrap_or_else(|| {
                    // Invariant: x/y have a length of >= 2 so at least 1 dx should exist
                    dx.last().unwrap()
                })
                .0;

            let n = (preprocessed.xy.domain_len() / median_dx).ceil() as u64;
            (name.to_owned(), n)
        })
        .into_group_map();

    // Calculate statistics and find the mean interpolation amount
    let max_n_per_var = n_per_var.into_iter().map(|(var, ns)| {
        let float_ns = ns.iter().map(|&n| n as f64).collect::<Vec<_>>();
        let n_mean = stats::mean(&float_ns);
        let n_stddev = stats::variance_mean(&float_ns, n_mean).sqrt();
        let n_mean_log2 = n_mean.log2().ceil() as u64;
        debug!("Minimum dx interpolation for {var}: mean = {n_mean:.2} ~ 2^{n_mean_log2}, stddev = {n_stddev:.2}");
        (var, n_mean as u64)
    }).collect::<HashMap<_, _>>();

    // May be None if the template does not use dx based interpolation
    let max_n = max_n_per_var.values().max().copied();
    info!("Local dx interpolation maxima: {max_n_per_var:?}");
    info!("Global dx interpolation maximum: {max_n:?}");

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
        .collect::<Result<Vec<_>>>()
        .context("Failed to process data")?
        .into_iter()
        .map(|(processed, saves)| {
            save_records.extend(saves);
            processed
        })
        .collect::<Vec<_>>();

    // FFT
    if let Some(fft) = fft {
        // We go through all this trouble of creating a "generator" to reduce memory load
        // Multiple files with relatively large sweep lengths can quickly fill your RAM
        // NB. The intermediate generators must also be lazy so we use dynamically dispatched iterators
        // Currently this is single threaded because making it parallel is a pain due to rayon not being object safe

        let fft_amount = processed.len()
            * match fft.sweep {
                FftSweep::Full => 1,
                FftSweep::Lower { sweep_steps }
                | FftSweep::Upper { sweep_steps }
                | FftSweep::Windows { sweep_steps } => sweep_steps,
            };

        let prepared_generator = {
            let fft = fft.clone();
            processed.into_iter().flat_map(move |processed| {
                // Extract boundaries
                let invert_x = processed.settings.preprocessing.invert_x;
                let left = processed.xy.left_x();
                let right = processed.xy.right_x();

                // Left and right boundaries in units of x
                let x_left = if invert_x { 1.0 / right } else { left };
                let x_right = if invert_x { 1.0 / left } else { right };

                type Iter = Box<dyn Iterator<Item = Result<PreparedFft>> + Send>;

                match fft.sweep {
                    FftSweep::Full => {
                        let iter =
                            std::iter::once(processed.prepare_fft(fft.clone(), left, right, None));
                        Box::new(iter) as Iter
                    }
                    FftSweep::Lower { sweep_steps } => {
                        // Sweep lower boundary while the upper boundary stays fixed
                        // Sweep uniformly in x
                        let dx = (x_right - x_left) / sweep_steps as f64;

                        // Iterate left boundary down, starting 1 tick left from the right side
                        let fft = fft.clone();
                        let iter = (1..=sweep_steps)
                            .map(move |i| {
                                let x_left = x_right - i as f64 * dx;
                                if invert_x {
                                    (i, 1.0 / x_right, 1.0 / x_left)
                                } else {
                                    (i, x_left, x_right)
                                }
                            })
                            .map(move |(i, left, right)| {
                                processed
                                    .clone()
                                    .prepare_fft(fft.clone(), left, right, Some(i))
                            });
                        Box::new(iter) as Iter
                    }
                    FftSweep::Upper { sweep_steps } => {
                        // Sweep upper boundary while the lower boundary stays fixed
                        // Sweep uniformly in x
                        let dx = (x_right - x_left) / sweep_steps as f64;

                        // Iterate right boundary up, starting 1 tick right from the left side
                        let fft = fft.clone();
                        let iter = (1..=sweep_steps)
                            .map(move |i| {
                                let x_right = x_left + i as f64 * dx;
                                if invert_x {
                                    (i, 1.0 / x_right, 1.0 / x_left)
                                } else {
                                    (i, x_left, x_right)
                                }
                            })
                            .map(move |(i, left, right)| {
                                processed
                                    .clone()
                                    .prepare_fft(fft.clone(), left, right, Some(i))
                            });
                        Box::new(iter) as Iter
                    }
                    FftSweep::Windows { sweep_steps } => {
                        // Sweep center of window
                        // Use 50% overlap between the windows
                        let dx = (right - left) / ((sweep_steps + 1) as f64);

                        // Move the window along the domain
                        let fft = fft.clone();
                        let iter = (0..sweep_steps)
                            .map(move |i| {
                                // NB. Careful with shadowing!
                                let right = left + (i + 2) as f64 * dx;
                                let left = left + i as f64 * dx;
                                (i, left, right)
                            })
                            .map(move |(i, left, right)| {
                                processed
                                    .clone()
                                    .prepare_fft(fft.clone(), left, right, Some(i))
                            });
                        Box::new(iter) as Iter
                    }
                }
            })
        };

        let use_cuda = {
            #[cfg(feature = "cuda")]
            if fft.cuda {
                if cufft::gpu_count() == 0 {
                    warn!("No CUDA capable GPUs detected, using CPU instead");
                    false
                } else {
                    true
                }
            } else {
                false
            }

            #[cfg(not(feature = "cuda"))]
            {
                warn!("Requested GPU FFT but CUDA support was disabled during build");
                false
            }
        };

        if use_cuda {
            #[cfg(feature = "cuda")]
            {
                info!(
                    "Computing batched FFTs on GPU '{}'",
                    cufft::gpu_name().context("Failed to retrieve GPU name")?
                );

                let gpu_memory_bytes =
                    cufft::gpu_memory().context("Failed to retrieve GPU memory")?;
                debug!("GPU has {} MB of RAM", gpu_memory_bytes / 10u64.pow(6));

                // Get total FFT length
                let fft_len = fft.zero_pad as usize;

                // Check if we need to split the batches
                // We conservatively do not allocate more than 1/4th of the available memory,
                // as we expect to need around 3N floats + some workspace.
                let gpu_memory_bytes = gpu_memory_bytes as usize / 4;
                let single_fft_bytes = fft_len * std::mem::size_of::<f32>();
                let ffts_per_batch = gpu_memory_bytes / single_fft_bytes;

                ensure!(
                    ffts_per_batch > 0,
                    "The GPU does not have enough RAM for a single FFT"
                );

                let n_subbatches = (fft_amount as f64 / ffts_per_batch as f64).ceil() as usize;
                info!(
                    "GPU will run {n_subbatches} batches of {} MB each",
                    fft_amount.min(ffts_per_batch) * single_fft_bytes / 10usize.pow(6)
                );

                // Now we only generate the required amount of FFTs,
                // minimizing memory load.
                for (i, prepared) in prepared_generator
                    .chunks(ffts_per_batch)
                    .into_iter()
                    .enumerate()
                {
                    info!("Preparing FFT batch #{i}");

                    // Take some prepared FFTs and split them into settings and data
                    let (prepared, data): (Vec<_>, Vec<_>) = prepared
                        .collect::<Result<Vec<_>>>()
                        .context("Failed to prepare FFT batch")?
                        .into_iter()
                        .map(|prepared| prepared.split_data())
                        .unzip();

                    // Convert data to a contiguous buffer of f32
                    let ffts = data.len(); // NB. Not necessarily equal to ffts_per_batch!
                    let subbatch = data
                        .into_iter()
                        .flatten()
                        .map(|x| x as f32)
                        .collect::<Vec<_>>();

                    assert_eq!(subbatch.len(), ffts * fft_len);

                    // Compute FFT on GPU
                    info!("Running GPU FFT batch #{i}");
                    let fft = cufft::fft32_batch_contiguous(&subbatch, ffts)
                        .context("Failed to execute GPU FFT")?;

                    info!("FFT postprocessing batch #{i}");
                    prepared
                        .into_par_iter()
                        .zip(fft)
                        .map(|(prepared, fft)| prepared.finish(fft))
                        .collect::<Result<Vec<_>>>()
                        .context("Failed to postprocess FFT batch")?
                        .into_iter()
                        .for_each(|saves| {
                            save_records.extend(saves);
                        });
                }

                #[cfg(not(feature = "cuda"))]
                unreachable!()
            }
        } else {
            info!("Computing FFTs on CPU");
            prepared_generator
                .par_bridge()
                .map(|prepared| {
                    let (prepared, mut data) =
                        prepared.context("Failed to prepare FFT")?.split_data();

                    let name = &prepared.settings.extract.name;
                    let src = prepared.settings.file.source.as_str();

                    debug!("Computing CPU FFT for '{src}':'{name}'");
                    fft::fft64_packed(&mut data).context("Failed to execute CPU FFT")?;
                    let fft = fft::fft64_unpack(&data);
                    let saves = prepared.finish(fft).context("Failed to postprocess FFT")?;

                    Ok(saves)
                })
                .collect::<Result<Vec<_>>>()
                .context("Failed to compute FFTs on CPU")?
                .into_iter()
                .for_each(|saves| {
                    save_records.extend(saves);
                });
        };
    }

    let end = Instant::now();
    let runtime_ms = (end - start).as_millis();
    let runtime_sec = (end - start).as_secs_f64();
    info!("Finished in {runtime_ms} ms");

    // Store metadata json for postprocessing convenience
    info!("Storing metadata json");
    let metadata = json::json!({
        "version": version,
        "log": log_path,
        "runtime_sec": runtime_sec,
        "interpolation_stats": {
            "max_per_variable": max_n_per_var,
            "max": max_n
        },
        "variables": extract.into_iter().map(|extract| extract.name).collect::<Vec<_>>(),
        "output": save_records,
        "settings": settings,
        "processed_files": files.into_iter().map(|file| file.source).collect::<Vec<_>>(),
    });
    std::fs::write(
        format!("output/{}/metadata.json", project.title),
        &json::to_string_pretty(&metadata).expect("failed to generate metadata json"),
    )
    .context("Failed to store metadata json")?;

    Ok(())
}

impl Settings {
    fn prepare(self: Arc<Self>, data: &Data, project: Project) -> Result<Option<Prepared>> {
        let settings = self;
        let Extract { name, x, y } = &settings.extract;
        let src = settings.file.source.as_str();

        info!("Preparing file '{src}': dataset '{name}'");

        // Check if the given names actually correspond to existing data columns
        if !data.contains(x) {
            warn!("specified x column '{x}' for dataset '{name}' does not exist in file '{src}'");
            return Ok(None);
        }
        if !data.contains(y) {
            warn!("specified y column '{y}' for dataset '{name}' does not exist in file '{src}'");
            return Ok(None);
        }

        // Extract the columns
        debug!("Extracting dataset '{name}' (x='{x}', y='{y}')");
        let xy = match data.clone_xy(x, y) {
            Some(xy) => xy,
            None => {
                bail!("Dataset '{name}' (x='{x}', y='{y}') has less than 2 values");
            }
        };

        // Check for NaN and infinities
        if !xy.is_finite() {
            bail!("File '{src}':'{name}' contains non-finite values");
        }

        // Prepare labels
        let x_label = x.to_owned();
        let y_label = y.to_owned();

        Ok(Some(Prepared {
            project,
            settings,
            x_label,
            y_label,
            xy,
        }))
    }
}

impl Prepared {
    fn preprocess(self) -> Result<(Preprocessed, Vec<SaveRecord>)> {
        let Self {
            project,
            settings,
            x_label,
            y_label,
            xy,
        } = self;

        let Extract { name, x, y: _ } = &settings.extract;
        let src = settings.file.source.as_str();

        info!("Preprocessing file '{src}': dataset '{name}'");
        let mut saves = vec![];

        let n_log2 = (xy.len() as f64).log2();
        trace!("Dataset '{src}':'{name}' length: ~2^{n_log2:.2}");

        // Monotonicity
        trace!("Making '{src}':'{name}' monotonic");
        let mut xy = xy.into_monotonic();

        // Output raw data
        trace!("Storing raw data for '{src}':'{name}'");
        saves.push(
            save(
                &project,
                name,
                "raw",
                &settings.file.dest,
                &x_label,
                &y_label,
                project.plot,
                xy.x(),
                xy.y(),
                json::json!({
                    "tags": settings.file.metadata,
                }),
            )
            .with_context(|| format!("Failed to store raw data for '{src}':'{name}'"))?,
        );

        // Impulse filtering
        // Do this before trimming, such that edge artifacts may be cut off afterwards
        if settings.preprocessing.impulse_filter > 0 {
            let width = settings.preprocessing.impulse_filter;
            let tuning = settings.preprocessing.impulse_tuning;
            trace!(
                "Applying impulse filter of width {width} and tuning {tuning} to '{src}':'{name}'"
            );
            xy.impulse_filter(width as usize, tuning);
        }

        // Masking
        for &Mask { left, right } in &settings.preprocessing.masks {
            ensure!(
                left >= xy.left_x(),
                "Mask {left} is outside domain {:?} for '{src}':'{name}'",
                xy.left_x()
            );
            ensure!(
                right <= xy.right_x(),
                "Mask {right} is outside domain {:?} for '{src}':'{name}'",
                xy.right_x()
            );

            trace!("Masking '{src}':'{name}' from {} to {}", left, right);
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
            trace!("Flipping trim domain around zero for '{src}':'{name}'");
            std::mem::swap(&mut trim_left, &mut trim_right);
            trim_left *= -1.0;
            trim_right *= -1.0;
        }

        if settings.preprocessing.invert_x {
            ensure!(
                trim_left != 0.0,
                "Can't trim to zero when inverting x for '{src}':'{name}'"
            );
            ensure!(
                trim_right != 0.0,
                "Can't trim to zero when inverting x for '{src}':'{name}'"
            );
        }

        ensure!(
            xy.left_x() <= trim_left,
            "Left trim {trim_left} is outside domain {:?} for '{src}':'{name}'",
            xy.left_x()
        );
        ensure!(
            xy.right_x() >= trim_right,
            "Right trim {trim_right} is outside domain {:?} for '{src}':'{name}'",
            xy.right_x()
        );
        trace!("Desired data domain: [{trim_left},{trim_right}]");
        xy.trim(trim_left, trim_right);
        trace!("Actual data domain: [{},{}]", xy.left_x(), xy.right_x());

        // Premultiplication
        // Use local override prefactor if set
        let mx = settings.preprocessing.prefactor_x;
        let my = settings.preprocessing.prefactor_y;
        if mx != 1.0 {
            trace!("Multiplying '{src}':'{name}' x by {mx}");
            xy.multiply_x(mx);
        }
        if my != 1.0 {
            trace!("Multiplying '{src}':'{name}' y by {my}");
            xy.multiply_y(my);
        }

        // Output preprocessed data
        trace!("Storing preprocessed data for '{src}':'{name}'");
        saves.push(
            save(
                &project,
                name,
                "preprocessed",
                &settings.file.dest,
                &x_label,
                &y_label,
                project.plot,
                xy.x(),
                xy.y(),
                json::json!({
                    "tags": settings.file.metadata,
                    "settings": settings.preprocessing,
                }),
            )
            .with_context(|| format!("Failed to store preprocessed data for '{src}':'{name}'"))?,
        );

        // 1/x
        if settings.preprocessing.invert_x {
            trace!("Inverting '{src}':'{name}' x");
            xy.invert_x();

            // Output inverted data
            trace!("Storing inverted data for '{src}':'{name}'");
            saves.push(
                save(
                    &project,
                    name,
                    "inverted",
                    &settings.file.dest,
                    &x_label,
                    &y_label,
                    false,
                    xy.x(),
                    xy.y(),
                    json::json!({
                        "tags": settings.file.metadata,
                    }),
                )
                .with_context(|| format!("Failed to store inverted data for '{src}':'{name}'"))?,
            );

            trace!("Inverted data domain: [{},{}]", xy.left_x(), xy.right_x());
        }

        let x_label = if settings.preprocessing.invert_x {
            format!("1/{x}")
        } else {
            x_label
        };

        Ok((
            Preprocessed {
                project,
                settings,
                x_label,
                y_label,
                xy,
            },
            saves,
        ))
    }
}

impl Preprocessed {
    fn process(
        self,
        n_interp_global: Option<u64>,
        n_interp_var: Option<u64>,
    ) -> Result<(Processed, Vec<SaveRecord>)> {
        let Self {
            project,
            settings,
            x_label,
            y_label,
            xy,
        } = self;
        let name = &settings.extract.name;
        let src = settings.file.source.as_str();

        info!("Processing file '{src}': dataset '{name}'");
        let mut saves = vec![];

        // Interpolation
        let (x, y) = match &settings.processing.interpolation {
            Some(algorithm) => {
                let n_interp = match algorithm.length() {
                    InterpolationLength::MinimumPerVariable => {
                        // Invariant: the interpolation length statistics routine will have returned Some
                        n_interp_var.expect("should have calculated n_interp_var")
                    }
                    InterpolationLength::Minimum => {
                        // Invariant: the interpolation length statistics routine will have returned Some
                        n_interp_global.expect("should have calculated n_interp_global")
                    }
                    InterpolationLength::Amount(n) => n,
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

                debug!("Interpolating '{src}':'{name}' using {deriv_str} at {n_interp} points using {algorithm} interpolation");

                let dx = xy.domain_len() / n_interp as f64;
                let x_eval = (0..n_interp)
                    .map(|i| (i as f64) * dx + xy.left_x())
                    .collect::<Vec<_>>();
                let y_eval =
                    interpolate_monotonic((*algorithm).into(), deriv, xy.x(), xy.y(), &x_eval)
                        .with_context(|| format!("Failed to make '{src}':'{name}' monotonic"))?;

                // Output post interpolation data
                trace!("Storing post-interpolation data for '{src}':'{name}'");
                saves.push(
                    save(
                        &project,
                        name,
                        "post_interpolation",
                        &settings.file.dest,
                        &x_label,
                        &y_label,
                        false,
                        &x_eval,
                        &y_eval,
                        json::json!({
                            "tags": settings.file.metadata,
                            "settings": settings.processing,
                            "interpolation_n": n_interp
                        }),
                    )
                    .with_context(|| {
                        format!("Failed to store post-interpolated data for '{src}':'{name}'")
                    })?,
                );

                (x_eval, y_eval)
            }
            None => xy.take_xy(),
        };

        let xy = MonotonicXY::new_unchecked(x, y);

        Ok((
            Processed {
                project,
                settings,
                x_label,
                y_label,
                xy,
            },
            saves,
        ))
    }
}

impl Processed {
    fn prepare_fft(
        self,
        fft_settings: Fft,
        domain_left: f64,
        domain_right: f64,
        sweep_index: Option<usize>,
    ) -> Result<PreparedFft> {
        let Self {
            project,
            settings,
            x_label,
            y_label,
            mut xy,
        } = self;
        let name = &settings.extract.name;
        let src = settings.file.source.as_str();

        let window_label = sweep_index
            .map(|i| format!("window #{i}"))
            .unwrap_or_else(|| "(full window)".to_owned());
        debug!("Preparing file '{src}': dataset '{name}' for FFT {window_label}");

        // Trim domain
        // The domain is calculated internally so we may assume its boundaries are correct
        trace!(
            "Trimming '{src}':'{name}' for FFT {window_label} to [{domain_left}, {domain_right}]"
        );
        xy.trim(domain_left, domain_right);

        // Extract requested total FFT length
        let desired_n = fft_settings.zero_pad as usize;
        let desired_n_log2 = (desired_n as f64).log2().floor() as usize;

        // Amount of nonzero points
        let n_data = xy.len();
        // Amount of padded zero points
        let n_pad = desired_n.saturating_sub(n_data);

        // FFT preprocessing: centering, windowing, padding
        if fft_settings.center {
            let mean = stats::mean(xy.y());

            trace!("Removing mean value {mean} from '{src}':'{name}'");
            xy.y_mut().iter_mut().for_each(|y| {
                *y -= mean;
            });
        }
        if fft_settings.hann {
            trace!("Applying Hann window to '{src}':'{name}'");
            xy.y_mut().iter_mut().enumerate().for_each(|(i, y)| {
                *y *= (i as f64 * std::f64::consts::PI / (n_data - 1) as f64)
                    .sin()
                    .powi(2);
            });

            // Avoid numerical error and leaking at the edges
            // Invariant: x/y can not be empty
            *xy.y_mut().first_mut().unwrap() = 0.0;
            *xy.y_mut().last_mut().unwrap() = 0.0;
        }

        if n_pad > 0 {
            trace!(
                    "Zero padding '{src}':'{name}' by {n_pad} to reach 2^{desired_n_log2} points total length ({} MB)",
                    desired_n * std::mem::size_of::<f64>() / 1024usize.pow(2)
                );
            xy.y_mut().resize(desired_n, 0.0);
        } else {
            ensure!(
                xy.y().len().is_power_of_two(),
                "Data length is larger than 2^{desired_n_log2}, so no padding can take place, but the length is not a power of two."
            );
        }

        // Split x and y
        let (x, y) = xy.take_xy();

        Ok(PreparedFft {
            inner: PreparedFftInner {
                project,
                settings,
                fft_settings,
                sweep_index,
                x_label,
                y_label,
                n_data,
                n_pad,
                domain_left,
                domain_right,
                x,
            },
            y,
        })
    }
}

impl PreparedFft {
    fn split_data(self) -> (PreparedFftInner, Vec<f64>) {
        let Self { inner, y } = self;
        (inner, y)
    }
}

impl PreparedFftInner {
    fn finish<F: Float + AsPrimitive<f64>>(
        self,
        mut fft: Vec<Complex<F>>,
    ) -> Result<Vec<SaveRecord>> {
        let Self {
            project,
            settings,
            fft_settings,
            sweep_index,
            x_label: _,
            y_label: _,
            n_data,
            n_pad,
            domain_left,
            domain_right,
            x,
        } = self;
        let name = &settings.extract.name;
        let src = settings.file.source.as_str();

        debug!("FFT postprocessing file '{src}': dataset '{name}'");
        let mut saves = vec![];

        // Prepare frequency space cutoffs
        let lower_cutoff = fft_settings.truncate_lower.unwrap_or(0.0);
        let upper_cutoff = fft_settings.truncate_upper.unwrap_or(f64::INFINITY);
        if lower_cutoff != 0.0 || upper_cutoff.is_finite() {
            trace!("Truncating FFT to {lower_cutoff}..{upper_cutoff} 'Hz' for '{src}':'{name}'");
        }

        let n = n_data + n_pad;
        let dt = x[1] - x[0];

        let start_idx = ((lower_cutoff * dt * n as f64).ceil() as usize).min(fft.len());
        let end_idx = ((upper_cutoff * dt * n as f64).ceil() as usize).min(fft.len());

        // Sampled frequencies : k/(N dt)
        // Note: this includes zero padding.
        let freq_normalisation = 1.0 / (dt * n as f64);
        let x = (start_idx..end_idx)
            .map(|i| i as f64 * freq_normalisation)
            .collect::<Vec<_>>();

        // Take absolute value and normalize
        // NB. Do this after truncation to save a huge amount of work
        trace!("Normalizing FFT by 1/{n_data}");
        let normalisation = 1.0 / n_data as f64;
        let fft = fft
            .drain(start_idx..end_idx)
            .map(|y| y.norm().as_() * normalisation)
            .collect::<Vec<_>>();

        // Output FFT
        debug!("Storing FFT for '{src}':'{name}'");
        saves.push(
            save(
                &project,
                name,
                &match fft_settings.sweep {
                    FftSweep::Full => "fft".to_owned(),
                    FftSweep::Lower { .. } => format!("fft_sweep_lower_{}", sweep_index.unwrap()),
                    FftSweep::Upper { .. } => format!("fft_sweep_upper_{}", sweep_index.unwrap()),
                    FftSweep::Windows { .. } => {
                        format!("fft_sweep_window_{}", sweep_index.unwrap())
                    }
                },
                &settings.file.dest,
                "Frequency",
                "FFT Amplitude",
                false,
                &x,
                &fft,
                json::json!({
                    "tags": settings.file.metadata,
                    "fft_domain": (domain_left, domain_right),
                    "settings": fft_settings,
                }),
            )
            .with_context(|| format!("Failed to store FFT for '{src}':'{name}'"))?,
        );

        Ok(saves)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Prepared {
    project: Project,
    settings: Arc<Settings>,
    x_label: String,
    y_label: String,
    xy: XY,
}

#[derive(Clone, Debug, PartialEq)]
struct Preprocessed {
    project: Project,
    settings: Arc<Settings>,
    x_label: String,
    y_label: String,
    xy: MonotonicXY,
}

#[derive(Clone, Debug, PartialEq)]
struct Processed {
    project: Project,
    settings: Arc<Settings>,
    x_label: String,
    y_label: String,
    xy: MonotonicXY,
}

#[derive(Clone, Debug, PartialEq)]
struct PreparedFft {
    inner: PreparedFftInner,
    y: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
struct PreparedFftInner {
    project: Project,
    settings: Arc<Settings>,
    fft_settings: Fft,
    sweep_index: Option<usize>,
    x_label: String,
    y_label: String,
    n_data: usize,
    n_pad: usize,
    domain_left: f64,
    domain_right: f64,
    x: Vec<f64>,
}

#[must_use]
#[derive(Clone, Debug, PartialEq, Serialize)]
struct SaveRecord {
    format: Format,
    stage: String,
    variable: String,
    path: String,
    metadata: json::Value,
}

fn save(
    project: &Project,
    name: &str,
    title: &str,
    dst: &str,
    x_label: &str,
    y_label: &str,
    plot: bool,
    x: &[f64],
    y: &[f64],
    metadata: json::Value,
) -> Result<SaveRecord> {
    let project_title = &project.title;

    let extra_dirs = match PathBuf::from(&dst).parent() {
        Some(extra_dirs) => format!("/{}", extra_dirs.to_string_lossy()),
        None => String::new(),
    };

    let _ = std::fs::create_dir_all(format!(
        "output/{project_title}/{name}/{title}{}",
        extra_dirs
    ));

    if plot {
        output::plot(
            x,
            y,
            title,
            x_label,
            y_label,
            format!("output/{project_title}/{name}/{title}/{dst}.png"),
        )
        .context("Failed to generate plot")?;
    }

    match project.format {
        Format::Csv => {
            let csv_path = format!("output/{project_title}/{name}/{title}/{dst}.csv");
            output::store_csv(x, y, &csv_path).context("Failed to store CSV")?;
            Ok(SaveRecord {
                format: project.format,
                stage: title.to_owned(),
                variable: name.to_owned(),
                path: csv_path,
                metadata,
            })
        }
        Format::MessagePack => {
            let msgpack_path = format!("output/{project_title}/{name}/{title}/{dst}.msg");
            output::store_messagepack(x, y, &msgpack_path)
                .context("Failed to store messagepack")?;
            Ok(SaveRecord {
                format: project.format,
                stage: title.to_owned(),
                variable: name.to_owned(),
                path: msgpack_path,
                metadata,
            })
        }
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
