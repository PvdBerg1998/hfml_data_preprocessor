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
mod processing;
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
use gsl_rust::fft;
use gsl_rust::interpolation::Derivative;
use gsl_rust::stats;
use itertools::Itertools;
use log::*;
use rayon::prelude::*;
use serde_json as json;
use settings::Template;
use simplelog::format_description;
use simplelog::*;
use std::{
    collections::HashMap,
    fs::File,
    path::PathBuf,
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

// The amount of dx samples we use to calculate the minimum dx of a dataset
const INTERP_MIN_DX_SAMPLES: usize = 5;

fn main() {
    gsl_rust::disable_error_handler(); // We handle errors ourselves and this aborts the program.

    if let Err(e) = _main() {
        error!("{e:?}");
    }
}

fn _main() -> Result<()> {
    let timestamp_start = Instant::now();

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
                .set_time_format_custom(format_description!("[hour]:[minute]:[second].[subsecond]"))
                .set_thread_mode(ThreadLogMode::Names)
                .set_thread_level(LevelFilter::Error)
                .set_location_level(LevelFilter::Off)
                .set_target_level(LevelFilter::Off)
                .build(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        ),
        WriteLogger::new(
            LevelFilter::Trace,
            ConfigBuilder::default()
                .set_time_format_custom(format_description!("[hour]:[minute]:[second].[subsecond]"))
                .set_thread_mode(ThreadLogMode::Names)
                .set_thread_level(LevelFilter::Error)
                .set_location_level(LevelFilter::Off)
                .set_target_level(LevelFilter::Off)
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
        .num_threads(if args.disable_threading {
            1
        } else if template.project.threading {
            0
        } else {
            1
        })
        .thread_name(|i| format!("worker {}", i + 1))
        .build_global()
        .context("Failed to start threadpool")?;

    if template.files.is_empty() {
        bail!("No files listed in template");
    }

    // Load files
    info!("Loading files");
    let timestamp_loading = Instant::now();
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
        files: _,
        extract,
        rename: _,
    } = template;

    // Dumping area for all save records
    // We just guesstimate the storage required: 5 types of output for a normal FFT
    let mut save_records = Vec::with_capacity(settings.len() * 5);

    // Prepare and preprocess data in parallel
    info!("Preprocessing files");
    let timestamp_preprocessing = Instant::now();
    let preprocessed = settings
        //.clone()
        .into_par_iter()
        .map(Arc::new)
        .filter_map(|settings| {
            // Warn about possibly invalid configurations
            // Todo: add more

            // Very small interpolation amount
            if let InterpolationLength::Amount(n) = settings.processing.interpolation_n {
                if n < 64 {
                    warn!("Interpolation amount is probably too small");
                }
            }

            // X inversion without interpolation
            if settings.processing.interpolation.is_none() && settings.preprocessing.invert_x {
                warn!("Inverting x without interpolation will most likely result in wrong results");
            }

            // Derivative without interpolation
            if settings.processing.interpolation.is_none()
                && matches!(
                    settings.processing.derivative,
                    Derivative::First | Derivative::Second
                )
            {
                warn!("Derivative will not be applied without interpolation");
            }

            // FFT without interpolation
            if settings.processing.interpolation.is_none() && fft.is_some() {
                warn!("FFT without interpolation will result in wrong frequencies if your data is not uniformly sampled");
            }

            // Linear interpolation with second derivative
            if matches!(
                settings.processing.interpolation,
                Some(InterpolationAlgorithm::Linear { .. })
            ) && matches!(settings.processing.derivative, Derivative::Second)
            {
                warn!("Second derivative using linear interpolation will result in zero");
            }

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
    let timestamp_dx = Instant::now();
    let n_per_var = preprocessed
        .iter()
        .filter(|preprocessed| {
            // If we are not interpolating this data pair,
            // it shouldn't contribute to the stats.
            // It also saves time to skip this step if it's not required
            matches!(
                preprocessed.settings.processing.interpolation_n,
                InterpolationLength::Minimum | InterpolationLength::MinimumPerVariable
            )
        })
        .map(|preprocessed| {
            let name = &preprocessed.settings.extract.name;
            let src = preprocessed.settings.file.source.as_str();
            trace!("Calculating minimum delta x for all '{src}':'{name}'");

            // Loop over x, compare neighbours, sort
            let mut dx = preprocessed
                .xy()
                .x()
                .windows(2)
                .map(|window| {
                    let (a, b) = (window[0], window[1]);
                    (b - a).abs()
                })
                .map(float_ord::FloatOrd)
                .collect::<Vec<_>>();
            dx.sort_unstable();

            // Take the median of the smallest N intervals
            // This is done to avoid using a spurious extremely small interval
            let median_dx = dx
                .get(INTERP_MIN_DX_SAMPLES / 2)
                .unwrap_or_else(|| {
                    // Invariant: x/y have a length of >= 2 so at least 1 dx should exist
                    dx.last().unwrap()
                })
                .0;

            let n = (preprocessed.xy().domain_len() / median_dx).ceil() as u64;
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
    info!("Processing files");
    let timestamp_processing = Instant::now();
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
    let timestamp_fft = Instant::now();
    if let Some(fft) = fft {
        #[cfg(feature = "cuda")]
        let fft_amount = processed.len()
            * match fft.sweep {
                FftSweep::Full => 1,
                _ => fft.sweep_steps,
            };

        info!("Preparing files for FFT");
        let prepared = processed
            .into_par_iter()
            .map(|processed| {
                // Extract boundaries
                let invert_x = processed.settings.preprocessing.invert_x;
                let left = processed.xy().left_x();
                let right = processed.xy().right_x();

                // Left and right boundaries in units of x
                let x_left = if invert_x { 1.0 / right } else { left };
                let x_right = if invert_x { 1.0 / left } else { right };

                match fft.sweep {
                    FftSweep::Full => Ok(vec![processed.prepare_fft(
                        fft.clone(),
                        left,
                        right,
                        None,
                    )?]),
                    FftSweep::Lower => {
                        // Sweep lower boundary while the upper boundary stays fixed
                        // Sweep uniformly in x
                        let dx = (x_right - x_left) / fft.sweep_steps as f64;

                        // Iterate left boundary down, starting 1 tick left from the right side
                        let prepared = (1..=fft.sweep_steps)
                            .into_par_iter()
                            .map(|i| {
                                let x_left = x_right - i as f64 * dx;
                                if invert_x {
                                    (i, 1.0 / x_right, 1.0 / x_left)
                                } else {
                                    (i, x_left, x_right)
                                }
                            })
                            .map(|(i, left, right)| {
                                processed
                                    .clone()
                                    .prepare_fft(fft.clone(), left, right, Some(i))
                            })
                            .collect::<Result<Vec<_>>>()?;
                        Ok(prepared)
                    }
                    FftSweep::Upper => {
                        // Sweep upper boundary while the lower boundary stays fixed
                        // Sweep uniformly in x
                        let dx = (x_right - x_left) / fft.sweep_steps as f64;

                        // Iterate right boundary up, starting 1 tick right from the left side
                        let prepared = (1..=fft.sweep_steps)
                            .into_par_iter()
                            .map(|i| {
                                let x_right = x_left + i as f64 * dx;
                                if invert_x {
                                    (i, 1.0 / x_right, 1.0 / x_left)
                                } else {
                                    (i, x_left, x_right)
                                }
                            })
                            .map(|(i, left, right)| {
                                processed
                                    .clone()
                                    .prepare_fft(fft.clone(), left, right, Some(i))
                            })
                            .collect::<Result<Vec<_>>>()?;
                        Ok(prepared)
                    }
                    FftSweep::Windows => {
                        // Sweep center of window
                        // Use 50% overlap between the windows
                        let dx = (right - left) / ((fft.sweep_steps + 1) as f64);

                        // Move the window along the domain
                        let prepared = (0..fft.sweep_steps)
                            .into_par_iter()
                            .map(|i| {
                                // NB. Careful with shadowing!
                                let right = left + (i + 2) as f64 * dx;
                                let left = left + i as f64 * dx;
                                (i, left, right)
                            })
                            .map(|(i, left, right)| {
                                processed
                                    .clone()
                                    .prepare_fft(fft.clone(), left, right, Some(i))
                            })
                            .collect::<Result<Vec<_>>>()?;
                        Ok(prepared)
                    }
                }
            })
            .collect::<Result<Vec<_>>>()
            .context("Failed to prepare FFTs")?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        let use_cuda = {
            if args.disable_gpu {
                false
            } else if fft.cuda {
                #[cfg(feature = "cuda")]
                if cufft::gpu_count() == 0 {
                    warn!("No CUDA capable GPUs detected, using CPU instead");
                    false
                } else {
                    true
                }

                #[cfg(not(feature = "cuda"))]
                {
                    warn!("Requested GPU FFT but CUDA support was disabled during build");
                    false
                }
            } else {
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
                let fft_len = fft.zero_pad;

                // Check if we need to split the batches
                // We conservatively do not allocate more than 1/4th of the available memory,
                // as we expect to need around 3N floats + some workspace.
                let gpu_memory_bytes = gpu_memory_bytes as usize / 4;
                let single_fft_bytes = fft_len * std::mem::size_of::<f64>();
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

                for (i, prepared) in prepared
                    .into_iter()
                    .chunks(ffts_per_batch)
                    .into_iter()
                    .enumerate()
                {
                    info!("Preparing FFT batch #{i}");

                    // Evaluate a chunk of prepared FFTs
                    let prepared = prepared.collect::<Vec<_>>();

                    // Only transfer the least amount necessary
                    // Invariant: chunk cannot be empty
                    let gpu_out_len = prepared
                        .iter()
                        .map(|prepared| prepared.minimum_fft_len())
                        .max()
                        .unwrap();
                    trace!("FFT batch #{i} will transfer {gpu_out_len} complex values");

                    // Split into settings and data
                    let (prepared, data): (Vec<_>, Vec<_>) = prepared
                        .into_iter()
                        .map(|prepared| prepared.split_data())
                        .unzip();

                    // Compute FFT on GPU
                    info!("Running GPU FFT batch #{i}");
                    let data_refs = data.iter().map(|y| y.as_slice()).collect::<Vec<_>>();
                    let fft = cufft::fft64_batch(&data_refs, fft_len, Some(gpu_out_len))
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
            prepared
                .into_par_iter()
                .map(|mut prepared| {
                    // Zero pad on CPU
                    prepared
                        .zero_pad()
                        .context("Failed to apply zero padding")?;

                    // Split into settings and data
                    let (prepared, mut data) = prepared.split_data();

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

    // Store metadata json for postprocessing convenience
    info!("Storing metadata json");
    let timestamp_metadata = Instant::now();

    // Collect all unique tags
    let unique_tags = save_records
        .iter()
        .filter_map(|record| record.metadata().as_object())
        .filter_map(|map| map.get("tags"))
        .filter_map(|tags| tags.as_object())
        .flat_map(|tags| tags.iter())
        .into_group_map()
        .into_iter()
        .map(|(k, values)| {
            // We can't use Hash nor Ord so we have to resort to a slow elementwise PartialEq search
            let mut unique = vec![];
            for value in values {
                if !unique.contains(&value) {
                    unique.push(value);
                }
            }

            // Try to sort the data by downcasting
            if unique.iter().all(|x| x.is_i64()) {
                trace!("Sorting {k} as i64");
                unique.sort_unstable_by_key(|x| x.as_i64().unwrap());
            } else if unique.iter().all(|x| x.is_u64()) {
                trace!("Sorting {k} as u64");
                unique.sort_unstable_by_key(|x| x.as_u64().unwrap());
            } else if unique.iter().all(|x| x.is_f64()) {
                trace!("Sorting {k} as f64");
                unique.sort_unstable_by_key(|x| float_ord::FloatOrd(x.as_f64().unwrap()));
            } else if unique.iter().all(|x| x.is_string()) {
                trace!("Sorting {k} as string");
                unique.sort_unstable_by(|a, b| {
                    let a = a.as_str().unwrap();
                    let b = b.as_str().unwrap();
                    natord::compare(a, b)
                });
            } else if unique.iter().all(|x| x.is_boolean()) {
                trace!("Sorting {k} as boolean");
                unique.sort_unstable_by_key(|x| x.as_bool().unwrap());
            }

            (k, unique)
        })
        .collect::<HashMap<_, _>>();

    let metadata = json::json!({
        "version": version,
        "log": log_path,
        "interpolation_stats": {
            "max_per_variable": max_n_per_var,
            "max": max_n
        },
        "variables": extract.into_iter().map(|extract| extract.name).collect::<Vec<_>>(),
        "tags": unique_tags,
        "output": save_records,
        //"settings": settings,
        //"processed_files": files.into_iter().map(|file| file.source).collect::<Vec<_>>(),
    });
    std::fs::write(
        format!("output/{}/metadata.json", project.title),
        &json::to_string_pretty(&metadata).expect("failed to generate metadata json"),
    )
    .context("Failed to store metadata json")?;

    let timestamp_end = Instant::now();
    let runtime_ms = (timestamp_end - timestamp_start).as_millis();

    // Calculate time spent in each part
    let dt_loading_ms = (timestamp_preprocessing - timestamp_loading).as_millis();
    let dt_preprocessing_ms = (timestamp_dx - timestamp_preprocessing).as_millis();
    let dt_dx_ms = (timestamp_processing - timestamp_dx).as_millis();
    let dt_processing_ms = (timestamp_fft - timestamp_processing).as_millis();
    let dt_fft_ms = (timestamp_metadata - timestamp_fft).as_millis();
    let dt_metadata_ms = (timestamp_end - timestamp_metadata).as_millis();
    info!(
        "Finished in {runtime_ms} ms:
    Loading: {dt_loading_ms} ms
    Preprocessing: {dt_preprocessing_ms} ms
    Interpolation statistics: {dt_dx_ms} ms
    Processing: {dt_processing_ms} ms
    FFT: {dt_fft_ms} ms
    Metadata: {dt_metadata_ms} ms"
    );

    Ok(())
}

pub fn has_dup<T: PartialEq>(slice: &[T]) -> bool {
    for i in 1..slice.len() {
        if slice[i..].contains(&slice[i - 1]) {
            return true;
        }
    }
    false
}

#[derive(Clone, Debug, PartialEq, Eq, Parser)]
struct Args {
    template: PathBuf,
    #[clap(short, long, parse(from_occurrences))]
    verbose: usize,
    #[clap(short, long)]
    quiet: bool,
    #[clap(long)]
    disable_gpu: bool,
    #[clap(long)]
    disable_threading: bool,
}
