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

use anyhow::ensure;
use anyhow::Result;
use std::fmt::Display;
use std::fmt::Write as FmtWrite;
use std::io::Write as IoWrite;
use std::process::Command;
use std::process::Stdio;
use std::{fs::File, path::Path};

pub fn store_csv<P: AsRef<Path>>(x: &[f64], y: &[f64], path: P) -> Result<()> {
    assert_eq!(x.len(), y.len());
    // Preallocate string of size 2^22 ~ 4 MB
    let mut w = String::with_capacity(2usize.pow(22));
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
    }
    // Write to file in one go
    let mut f = File::create(path)?;
    f.set_len(w.len() as u64)?;
    f.write_all(w.as_bytes())?;
    Ok(())
}

pub fn plot_csv<P: AsRef<Path> + Display>(
    csv: P,
    title: &str,
    xlabel: &str,
    ylabel: &str,
    out: P,
) -> Result<()> {
    let title = title.replace('_', r"\_");
    let xlabel = xlabel.replace('_', r"\_");
    let ylabel = ylabel.replace('_', r"\_");

    let _ = std::fs::remove_file(&out);

    // Build gnuplot source
    let source = format!(
        "\
set terminal png size 640,480
set output '{out}'
set datafile separator ','
set title '{title}'
set xlabel '{xlabel}'
set ylabel '{ylabel}'
set autoscale xy
set key off
plot '{csv}' using 1:2 with lines lw 2
set output
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

    ensure!(child.wait()?.success(), "gnuplot returned error code");

    Ok(())
}
