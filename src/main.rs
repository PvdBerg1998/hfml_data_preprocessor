//mod maxpol;
mod output;
mod parsing;
mod settings;

use anyhow::anyhow;
use anyhow::bail;
use anyhow::Result;
use log::*;
use once_cell::sync::OnceCell;
use simplelog::*;
use std::{
    fs::File,
    time::{SystemTime, UNIX_EPOCH},
};

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
            LevelFilter::Debug,
            Config::default(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        ),
        WriteLogger::new(
            LevelFilter::Info,
            Config::default(),
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

    let mut data: parsing::Data = s.parse()?;
    info!("Columns: {:#?}", data.columns());

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
    debug!("Columns after replace: {:#?}", data.columns());

    // Extract data pairs
    for (name, xy) in settings.extract.pairs.iter() {
        let x = xy
            .get("x")
            .ok_or_else(|| anyhow!("Missing x column specification for dataset {name}"))?;
        let y = xy
            .get("y")
            .ok_or_else(|| anyhow!("Missing y column specification for dataset {name}"))?;

        if !data.contains(&*x) {
            bail!("Specified x column {x} for dataset {name} does not exist");
        }
        if !data.contains(&*y) {
            bail!("Specified y column {y} for dataset {name} does not exist");
        }

        info!("Extracting dataset {name} (x={x}, y={y})");
        let mut xy = data.xy(&*x, &*y);

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

        if settings.preprocessing.invert_x {
            info!("Inverting x");
            xy.invert_x();
        }

        // todo : interpolation

        debug!("Sorting data");
        let xy = xy.to_monotonic();

        let csv_path = format!("output/raw_{name}.csv");
        let png_path = format!("output/raw_{name}.png");
        let _ = std::fs::create_dir("output");

        debug!("Storing data to {csv_path}");
        output::store_csv(xy.x(), xy.y(), &csv_path)?;

        debug!("Plotting data to {png_path}");
        output::plot_csv(&csv_path, name, &*x, &*y, &png_path)?;

        debug!("Ready!");
    }

    Ok(())
}
