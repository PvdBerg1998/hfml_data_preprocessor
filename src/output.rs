/*
    output.rs
    Copyright (C) 2022 Pim van den Berg

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

use crate::settings::Format;
use crate::settings::Project;
use anyhow::Context;
use anyhow::Result;
use plotters::prelude::*;
use serde::Serialize;
use serde_json as json;
use std::fmt::Write as FmtWrite;
use std::io::Write as IoWrite;
use std::path::PathBuf;
use std::{fs::File, path::Path};

pub fn save(
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
        store_plot(
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
            store_csv(x, y, &csv_path).context("Failed to store CSV")?;
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
            store_messagepack(x, y, &msgpack_path).context("Failed to store messagepack")?;
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

pub fn store_csv<P: AsRef<Path>>(x: &[f64], y: &[f64], path: P) -> Result<()> {
    assert_eq!(x.len(), y.len());

    // Preallocate string of size 2^23 ~ 8 MB
    let mut w = String::with_capacity(2usize.pow(23));

    let mut x_buf = ryu::Buffer::new();
    let mut y_buf = ryu::Buffer::new();

    for (x, y) in x.iter().zip(y.iter()) {
        // Assume we only deal with finite values
        let x = x_buf.format_finite(*x);
        let y = y_buf.format_finite(*y);

        // Writing to a string never fails so we remove the error handling branches
        let _ = w.write_str(x);
        let _ = w.write_char(',');
        let _ = w.write_str(y);
        let _ = w.write_char('\n');
    }

    // Write to file in one go
    let mut f = File::create(path)?;
    f.set_len(w.len() as u64)?;
    f.write_all(w.as_bytes())?;

    Ok(())
}

pub fn store_messagepack<P: AsRef<Path>>(x: &[f64], y: &[f64], path: P) -> Result<()> {
    assert_eq!(x.len(), y.len());

    // Preallocate estimated amount of bytes
    // MessagePack spec: 9 bytes per f64
    // Add 64 bytes extra to have plenty of space for padding/markers/...
    let mut w = Vec::with_capacity(x.len() * 9 * 2 + 64);

    rmp::encode::write_array_len(&mut w, 2)?;

    rmp::encode::write_array_len(&mut w, x.len() as u32)?;
    for x in x {
        rmp::encode::write_f64(&mut w, *x)?;
    }

    rmp::encode::write_array_len(&mut w, y.len() as u32)?;
    for y in y {
        rmp::encode::write_f64(&mut w, *y)?;
    }

    // Write to file in one go
    let mut f = File::create(path)?;
    f.set_len(w.len() as u64)?;
    f.write_all(&w)?;

    Ok(())
}

pub fn store_plot<P: AsRef<Path>>(
    x: &[f64],
    y: &[f64],
    title: &str,
    x_label: &str,
    y_label: &str,
    out: P,
) -> Result<()> {
    let _ = std::fs::remove_file(&out);

    let xmin = *x.first().expect("empty dataset");
    let xmax = *x.last().expect("empty dataset");
    let ymin = *y
        .iter()
        .min_by_key(|&&f| float_ord::FloatOrd(f))
        .expect("empty dataset");
    let ymax = *y
        .iter()
        .max_by_key(|&&f| float_ord::FloatOrd(f))
        .expect("empty dataset");

    if xmin >= xmax {
        panic!("plotter only works with monotonically increasing data");
    }

    let root = BitMapBackend::new(&out, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    const FONT_STYLE: (&str, u32) = ("sans-serif", 24);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, FONT_STYLE.into_font())
        .margin(10u32)
        .x_label_area_size(60u32)
        .y_label_area_size(120u32)
        .build_cartesian_2d(xmin..xmax, ymin..ymax)?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .x_label_style(FONT_STYLE)
        .y_label_style(FONT_STYLE)
        .y_label_formatter(&|y| format!("{y:+.2e}"))
        .axis_desc_style(FONT_STYLE)
        .draw()?;

    chart.draw_series(LineSeries::new(
        x.iter().copied().zip(y.iter().copied()),
        RED,
    ))?;

    root.present()?;

    Ok(())
}

#[must_use]
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SaveRecord {
    format: Format,
    stage: String,
    variable: String,
    path: String,
    metadata: json::Value,
}

impl SaveRecord {
    pub fn metadata(&self) -> &json::Value {
        &self.metadata
    }
}
