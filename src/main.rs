use std::collections::HashMap;

use anyhow::Result;
use num_complex::Complex64;
use std::io::{BufWriter, Write};

fn main() -> Result<()> {
    let db = sled::Config::default()
        .path("maxpol_coefficients")
        .use_compression(true)
        .mode(sled::Mode::LowSpace)
        .open()?;

    let n = std::env::args().nth(1).unwrap().parse::<usize>().unwrap();
    let l = std::env::args().nth(2).unwrap().parse::<usize>().unwrap();
    let filter_factor = std::env::args().nth(3).unwrap().parse::<f64>().unwrap();

    let p = (((1.0 - filter_factor) * 2.0 * l as f64).round() as usize)
        .max(n)
        .min(2 * l);

    dbg!(p);

    let key = bincode::serialize::<(usize, usize, usize)>(&(n, l, p)).unwrap();
    let coeff = bincode::deserialize::<Vec<f64>>(
        &db.get(&key)?
            .ok_or_else(|| anyhow::anyhow!("value does not exist in database"))?,
    )?;

    let x = (-(l as isize)..l as isize)
        .map(|x| x as f64)
        .collect::<Vec<f64>>();
    let y = &coeff;
    store("maxpol_coeff.csv", x.iter(), y.iter())?;

    let x = (0..1000)
        .map(|x| x as f64 / 1000.0 * std::f64::consts::PI)
        .collect::<Vec<f64>>();
    let y = x
        .iter()
        .map(|&w| {
            coeff
                .iter()
                .enumerate()
                .map(|(i, c)| c * (Complex64::i() * (i + 1) as f64 * w).exp())
                .sum::<Complex64>()
                .norm()
        })
        .collect::<Vec<f64>>();
    store("maxpol_spectrum.csv", x.iter(), y.iter())?;

    let y = x.iter().map(|&w| w.powi(n as i32)).collect::<Vec<f64>>();
    store("dv_spectrum.csv", x.iter(), y.iter())?;

    std::process::Command::new("gnuplot")
        .arg("plot_maxpol_coeff.gp")
        .spawn()?;

    std::process::Command::new("gnuplot")
        .arg("plot_maxpol_spectrum.gp")
        .spawn()?;

    Ok(())
}

fn _main() -> Result<()> {
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

    let x_inv = x.iter().map(|&x| 1.0 / x).collect::<Vec<_>>();
    let x_interp_inv = x_interp.iter().map(|&x| 1.0 / x).collect::<Vec<_>>();
    store("raw.csv", x_inv.iter(), y)?;

    store("mr_fit.csv", x_interp_inv.iter(), mr_eval.y.iter())?;
    store(
        "mr_residuals.csv",
        x_inv.iter(),
        mr_fit.fit.residuals.iter(),
    )?;
    store(
        "mr_dv.csv",
        x_interp_inv.iter().skip(100),
        mr_eval.dv_flat().iter().skip(100),
    )?;
    store("full_fit.csv", x_interp_inv.iter(), full_eval.y.iter())?;
    store(
        "full_residuals.csv",
        x_inv.iter(),
        full_fit.fit.residuals.iter(),
    )?;
    store(
        "full_dv.csv",
        x_interp_inv.iter().skip(100),
        full_eval.dv_flat().iter().skip(100),
    )?;

    std::process::Command::new("gnuplot")
        .arg("plot.gp")
        .spawn()?;

    Ok(())
}

fn store<'a, 'b>(
    to: &str,
    x: impl IntoIterator<Item = &'a f64>,
    y: impl IntoIterator<Item = &'b f64>,
) -> Result<()> {
    let mut out = BufWriter::with_capacity(2usize.pow(16), std::fs::File::create(to)?);
    for (x, y) in x.into_iter().zip(y.into_iter()) {
        writeln!(&mut out, "{},{}", x, y)?;
    }
    Ok(())
}
