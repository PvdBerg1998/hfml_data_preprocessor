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
