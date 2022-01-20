/*
    output.rs
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

use anyhow::bail;
use anyhow::Result;
use std::fmt::Display;
use std::io::Write as IoWrite;
use std::process::Command;
use std::process::Stdio;
use std::{fs::File, io::BufWriter, path::Path};

pub fn store_csv<P: AsRef<Path>>(x: &[f64], y: &[f64], path: P) -> Result<()> {
    assert_eq!(x.len(), y.len());
    let mut w = BufWriter::new(File::create(path)?);
    for (x, y) in x.iter().zip(y.iter()) {
        writeln!(&mut w, "{x},{y}")?;
    }
    Ok(())
}

pub fn plot_csv<P: AsRef<Path> + Display>(
    csv: P,
    title: &str,
    xlabel: &str,
    ylabel: &str,
    out: P,
) -> Result<()> {
    // Build gnuplot source
    let source = format!(
        "\
set terminal pngcairo size 1920,1080
set datafile separator ','
set output '{out}'
set title '{title}' font ',24'
set xlabel '{xlabel}' font ',16'
set ylabel '{ylabel}' font ',16'
set autoscale xy
set key off
plot '{csv}' using 1:2 with lines lw 2
exit"
    );

    // Spawn gnuplot process
    let mut child = Command::new("gnuplot")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()?;
    let mut stdin = child.stdin.take().expect("piped stdin");
    stdin.write_all(source.as_bytes())?;
    drop(stdin);
    if child.wait()?.success() {
        Ok(())
    } else {
        bail!("gnuplot returned error code");
    }
}
