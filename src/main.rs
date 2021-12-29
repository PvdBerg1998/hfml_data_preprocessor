use std::collections::HashMap;

use anyhow::Result;
use std::io::{BufWriter, Write};

fn main() -> Result<()> {
    //disable_error_handler();

    let file = std::fs::read_to_string("file.002.dat")?;

    // Values are tab separated
    let headers = file
        .lines()
        .map(|line| line.trim().split('\t'))
        .flatten()
        .take_while(|entry| entry.parse::<f64>().is_err())
        .collect::<Vec<_>>();

    //dbg!(&headers);

    // Split data into columns
    let mut map = HashMap::new();
    for (i, header) in headers.iter().enumerate() {
        let column = file
            .lines()
            .flat_map(|line| line.trim().split('\t'))
            .filter_map(|entry| entry.parse::<f64>().ok())
            .skip(i)
            .step_by(headers.len())
            .collect::<Box<[f64]>>();
        map.insert(header.to_owned(), column);
    }

    let x = &map["Field"];
    let y = &map["S1_Vxx_8_13_x"];

    // Sort
    let perm = permutation::sort_by_key(x.as_ref(), |x| float_ord::FloatOrd(*x));
    let x = perm.apply_slice(x.as_ref());
    let y = perm.apply_slice(y.as_ref());

    let lower = x.iter().position(|&b| b >= 4.0).unwrap();
    let upper = x.iter().position(|&b| b >= 32.95).unwrap();

    let x = &x[lower..upper];
    let y = &y[lower..upper];

    dbg!(x.len());

    let a = x[0];
    let b = x[x.len() - 1];

    let mut mr_nbreak = std::env::args().nth(1).unwrap().parse::<f64>().unwrap();
    if mr_nbreak <= 1.0 {
        // interpret as fraction
        mr_nbreak *= x.len() as f64;
    }
    let mr_nbreak = mr_nbreak.floor() as usize;
    dbg!(mr_nbreak);
    println!("MR Fitting DoF: {}", mr_nbreak * mr_nbreak);

    let mut full_nbreak = std::env::args().nth(2).unwrap().parse::<f64>().unwrap();
    if full_nbreak <= 1.0 {
        // interpret as fraction
        full_nbreak *= x.len() as f64;
    }
    let full_nbreak = full_nbreak.floor() as usize;
    dbg!(full_nbreak);
    println!("Full Fitting DoF: {}", full_nbreak * full_nbreak);

    let mr_fit = gsl_rust::bspline::BSpline::fit(4, a, b, mr_nbreak, x, y)?;
    let full_fit = gsl_rust::bspline::BSpline::fit(4, a, b, full_nbreak, x, y)?;

    let n = 10_000;
    let delta = (b - a) / n as f64;
    let x_interp = (0..n).map(|i| i as f64 * delta + a).collect::<Box<[f64]>>();

    let mr_eval = mr_fit.eval::<1>(&x_interp)?;
    let full_eval = full_fit.eval::<1>(&x_interp)?;

    store("raw.csv", x, y)?;

    store("mr_fit.csv", x_interp.iter(), mr_eval.y.iter())?;
    store("mr_residuals.csv", x, mr_fit.fit.residuals.iter())?;
    store(
        "mr_dv.csv",
        x_interp.iter().skip(100),
        mr_eval.dv_flat().iter().skip(100),
    )?;
    store("full_fit.csv", x_interp.iter(), full_eval.y.iter())?;
    store("full_residuals.csv", x, full_fit.fit.residuals.iter())?;
    store(
        "full_dv.csv",
        x_interp.iter().skip(100),
        full_eval.dv_flat().iter().skip(100),
    )?;

    std::process::Command::new("gnuplot")
        .arg("plot.gp")
        .spawn()?;

    Ok(())
}

fn store<'a>(
    to: &str,
    x: impl IntoIterator<Item = &'a f64>,
    y: impl IntoIterator<Item = &'a f64>,
) -> Result<()> {
    let mut out = BufWriter::with_capacity(2usize.pow(16), std::fs::File::create(to)?);
    for (x, y) in x.into_iter().zip(y.into_iter()) {
        writeln!(&mut out, "{},{}", x, y)?;
    }
    Ok(())
}
