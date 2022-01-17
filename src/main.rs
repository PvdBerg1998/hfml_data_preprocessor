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
mod output;
mod parsing;
mod settings;

use anyhow::anyhow;
use anyhow::bail;
use anyhow::Result;
use gsl_rust::interpolation::interpolate_monotonic;
use log::*;
use once_cell::sync::OnceCell;
use parsing::Data;
use rayon::prelude::*;
use settings::Interpolation;
use simplelog::*;
use std::collections::HashMap;
use std::{
    fs::File,
    time::{SystemTime, UNIX_EPOCH},
};
use sysinfo::RefreshKind;
use sysinfo::SystemExt;

// todo: derivative, processing
// https://www.gnu.org/software/gsl/doc/html/interp.html#c.gsl_interp_eval_deriv_e

static SETTINGS: OnceCell<settings::Settings> = OnceCell::new();

fn main() {
    if let Err(e) = _main() {
        error!("{e}");
    }
}

fn _main() -> Result<()> {
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let _ = std::fs::create_dir("log");
    CombinedLogger::init(vec![
        TermLogger::new(
            LevelFilter::Info,
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

    let settings = settings::load("Settings.toml")?;
    debug!("Settings: {settings:#?}");
    SETTINGS.set(settings).unwrap();

    let settings = SETTINGS.get().unwrap();
    for path in settings.paths()? {
        info!("Loading file {}", path.display());
        let s = std::fs::read_to_string(path)?;
        process_file(&s)?;
    }

    Ok(())
}

fn process_file(s: &str) -> Result<()> {
    let settings = SETTINGS.get().unwrap();

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
    info!("Columns (after replacing): {:#?}", data.columns());

    // Extract data pairs
    if settings.project.threading {
        rayon::ThreadPoolBuilder::new()
            .thread_name(|i| format!("worker {}", i + 1))
            .build()?
            .install(|| {
                settings
                    .extract
                    .pairs
                    .par_iter()
                    .try_for_each(|(name, xy)| process_pair(name, xy, &data))
            })?;
    } else {
        settings
            .extract
            .pairs
            .iter()
            .try_for_each(|(name, xy)| process_pair(name, xy, &data))?;
    }

    Ok(())
}

fn process_pair(name: &str, xy: &HashMap<String, String>, data: &Data) -> Result<()> {
    let settings = SETTINGS.get().unwrap();

    let x = xy
        .get("x")
        .ok_or_else(|| anyhow!("Missing x column specification for dataset {name}"))?;
    let y = xy
        .get("y")
        .ok_or_else(|| anyhow!("Missing y column specification for dataset {name}"))?;

    if !data.contains(&*x) {
        bail!("specified x column {x} for dataset {name} does not exist");
    }
    if !data.contains(&*y) {
        bail!("specified y column {y} for dataset {name} does not exist");
    }

    info!("Extracting dataset {name} (x={x}, y={y})");
    let mut xy = data.xy(&*x, &*y);

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
    let xy = xy.into_monotonic();

    // Trimming
    if settings.preprocessing.trim_left >= settings.preprocessing.trim_right {
        bail!("Trimmed domain is empty or negative");
    }
    let trim = if settings.preprocessing.invert_x {
        if settings.preprocessing.trim_left == 0.0 || settings.preprocessing.trim_right == 0.0 {
            bail!("Can't use zero as trim boundary when inverting x");
        }
        [
            1.0 / settings.preprocessing.trim_right,
            1.0 / settings.preprocessing.trim_left,
        ]
    } else {
        [
            settings.preprocessing.trim_left,
            settings.preprocessing.trim_right,
        ]
    };

    let domain = [xy.min_x(), xy.max_x()];
    if trim[0] < domain[0] {
        bail!(
            "Left trim {} is outside domain {:?} for dataset {name}",
            trim[0],
            domain
        );
    }
    if trim[1] > domain[1] {
        bail!(
            "Right trim {} is outside domain {:?} for dataset {name}",
            trim[1],
            domain
        );
    }
    info!("Domain: {domain:?}");
    info!("Trimmed domain: {trim:?}");

    // Interpolation
    let x_eval;
    let y_eval;
    let out_x;
    let out_y;

    if let Some(Interpolation { n, algorithm }) = &settings.preprocessing.interpolation {
        // Parse and process one of:
        // - log2("n")
        // - log2("2^n")
        // - "min"
        // This rounds to the nearest higher power of 2.
        let interp_log2: u32 = match n.parse::<u64>() {
            Ok(n) => (n as f64).log2().ceil() as _,
            Err(_) => match n.as_str() {
                "min" => {
                    debug!("Calculating minimum delta x");
                    match xy
                        .trimmed_xy(trim[0], trim[1])
                        .0
                        .windows(2)
                        .map(|window| {
                            let (a, b) = (window[0], window[1]);
                            (b - a).abs()
                        })
                        .min_by_key(|&dx| float_ord::FloatOrd(dx))
                    {
                        Some(dx) => {
                            let n = (trim[1] - trim[0]) / dx;
                            n.log2().ceil() as _
                        }
                        None => bail!("Unable to determine minimum delta x"),
                    }
                }
                _ => {
                    if n.starts_with("2^") {
                        match n[2..].parse::<u32>() {
                            Ok(n_log2) => n_log2,
                            Err(_) => {
                                bail!("Invalid interpolation power");
                            }
                        }
                    } else {
                        bail!("Invalid interpolation length");
                    }
                }
            },
        };

        info!(
            "Interpolating to 2^{} points using {:?} interpolation",
            interp_log2, algorithm
        );

        if interp_log2 >= 60 {
            bail!("Interpolation length is too large for a 64 bit system");
        }

        let n_interp = 2u64.pow(interp_log2);
        let required_bytes = n_interp * std::mem::size_of::<f64>() as u64 * 2;

        // Check free RAM
        let mut sysinfo =
            sysinfo::System::new_with_specifics(RefreshKind::with_memory(RefreshKind::default()));
        sysinfo.refresh_memory();
        debug!(
            "Interpolation will require approximately {} kB",
            required_bytes / 1024
        );
        if required_bytes > sysinfo.free_memory() {
            bail!(
                "Aborting due to predicted OOM: {} GB required",
                required_bytes / 1024u64.pow(3)
            );
        }

        // Execute interpolation
        let delta = (trim[1] - trim[0]) / n_interp as f64;
        x_eval = (0..n_interp)
            .map(|i| (i as f64) * delta + trim[0] as f64)
            .collect::<Box<[_]>>();
        y_eval = interpolate_monotonic((*algorithm).into(), xy.x(), xy.y(), &x_eval)?;

        out_x = x_eval.as_ref();
        out_y = y_eval.as_ref();
    } else {
        info!("Skipping interpolation, directly truncating instead");
        let (trimmed_x, trimmed_y) = xy.trimmed_xy(trim[0], trim[1]);
        out_x = trimmed_x;
        out_y = trimmed_y;
    }

    // Output
    let csv_path = format!("output/raw_{name}.csv");
    let png_path = format!("output/raw_{name}.png");
    let _ = std::fs::create_dir("output");

    debug!("Storing data to {csv_path}");
    output::store_csv(&out_x, &out_y, &csv_path)?;

    debug!("Plotting data to {png_path}");
    output::plot_csv(&csv_path, name, &*x, &*y, &png_path)?;

    Ok(())
}
