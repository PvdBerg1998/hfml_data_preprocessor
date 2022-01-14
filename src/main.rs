mod maxpol;
mod parsing;

use anyhow::Result;
use convolve::convolve_scalar;
use parsing::*;
use std::io::{BufWriter, Write};

fn main() -> Result<()> {
    println!("Parsing data");
    let mut data: Data = std::fs::read_to_string("file.013.dat")?.parse()?;
    println!("{}", &data);

    data.rename_column("Field", "B");
    data.rename_column("B(T)", "B");
    data.rename_column("V_68_x", "Vxx");

    let xy = data.xy("B", "S2_Vxy_5_19_x").to_monotonic();
    let len = xy.x().len();

    println!("Loading MaxPol kernel");
    // NB. Apparently the database is not cross platform. Todo: bug report
    let db = maxpol::db()?;

    let filter_factor = std::env::args().nth(1).unwrap().parse::<f64>().unwrap();
    let kernel2 = maxpol::load_kernel(&db, 1, 100, filter_factor)?;

    // todo: in place mutation would be faster
    // todo: need larger kernels? => not really, we just need more smoothing power.
    // todo: can cut off edges?

    println!("Convolving MaxPol kernel with data");
    let out = convolve_scalar(&kernel2, xy.y());

    println!("Storing MaxPol result");
    store("raw.csv", xy.x(), xy.y())?;
    store("maxpol_dv.csv", &xy.x()[100..len - 100], &out)?;

    // Data boundaries
    let a = xy.min_x();
    let b = xy.max_x();

    // Extract spline nbreak from CLI
    let mut nbreak = std::env::args().nth(2).unwrap().parse::<f64>().unwrap();
    if nbreak <= 1.0 {
        // interpret as fraction
        nbreak *= xy.x().len() as f64;
    }
    let nbreak = nbreak.floor() as usize;
    println!("Spline nbreak: {}", nbreak);
    println!("Fitting DoF: {}", nbreak * nbreak);

    println!("Fitting BSpline");
    let fit = gsl_rust::bspline::BSpline::fit(4, a, b, nbreak, xy.x(), xy.y())?;

    println!("Evaluating BSpline");
    //let n = 10_000;
    //let delta = (b - a) / n as f64;
    //let x_interp = (0..n).map(|i| i as f64 * delta + a).collect::<Box<[f64]>>();
    let x_interp = xy.x();
    let eval = fit.eval::<1>(&x_interp)?;

    // todo: gsl_rust provide a simpler way to extract n'th derivative because this is annoying

    println!("Storing BSpline results");
    store("spline_fit.csv", &x_interp, &eval.y)?;
    store("spline_residuals.csv", xy.x(), &fit.fit.residuals)?;
    store(
        "spline_dv.csv",
        &x_interp[100..len - 100],
        &eval.dv_flat()[100..len - 100],
    )?;

    println!("MaxPol FFT");
    let fft_len = 2.0f64.powi((out.len() as f64).log2().floor() as i32) as usize;
    let mut fft = (&out[dbg!(out.len() - fft_len)..]).to_vec();
    gsl_rust::fft::fft64_packed(&mut fft)?;
    let fft = gsl_rust::fft::fft64_unpack_norm(&fft);

    println!("Storing MaxPol FFT");
    store1("maxpol_fft.csv", &fft)?;

    println!("BSpline FFT");
    let fft_len = 2.0f64.powi((eval.dv_flat().len() as f64).log2().floor() as i32) as usize;
    let mut fft = (&eval.dv_flat()[dbg!(eval.dv_flat().len() - fft_len)..]).to_vec();
    gsl_rust::fft::fft64_packed(&mut fft)?;
    let fft = gsl_rust::fft::fft64_unpack_norm(&fft);

    println!("Storing BSpline FFT");
    store1("spline_fft.csv", &fft)?;

    // println!("Residual FFT");
    // let fft_len = 2.0f64.powi((fit.fit.residuals.len() as f64).log2().floor() as i32) as usize;
    // let mut fft = (&fit.fit.residuals[0..fft_len]).to_vec();
    // gsl_rust::fft::fft64_packed(&mut fft)?;
    // let fft = gsl_rust::fft::fft64_unpack_norm(&fft);

    // println!("Storing residual FFT");
    // store1("spline_residual_fft.csv", &fft[10..])?;

    println!("Plotting results");
    std::process::Command::new("gnuplot")
        .arg("plot.gp")
        .spawn()?;

    Ok(())
}
/*
// fn _main() -> Result<()> {

//     let x = (-(l as isize)..l as isize)
//         .map(|x| x as f64)
//         .collect::<Vec<f64>>();
//     let y = &coeff;
//     store("maxpol_coeff.csv", x.iter(), y.iter())?;

//     let x = (0..1000)
//         .map(|x| x as f64 / 1000.0 * std::f64::consts::PI)
//         .collect::<Vec<f64>>();
//     let y = x
//         .iter()
//         .map(|&w| {
//             coeff
//                 .iter()
//                 .enumerate()
//                 .map(|(i, c)| c * (Complex64::i() * (i + 1) as f64 * w).exp())
//                 .sum::<Complex64>()
//                 .norm()
//         })
//         .collect::<Vec<f64>>();
//     store("maxpol_spectrum.csv", x.iter(), y.iter())?;

//     let y = x.iter().map(|&w| w.powi(n as i32)).collect::<Vec<f64>>();
//     store("dv_spectrum.csv", x.iter(), y.iter())?;

//     std::process::Command::new("gnuplot")
//         .arg("plot_maxpol_coeff.gp")
//         .spawn()?;

//     std::process::Command::new("gnuplot")
//         .arg("plot_maxpol_spectrum.gp")
//         .spawn()?;

//     Ok(())
// }

fn __main() -> Result<()> {
    //disable_error_handler();

    let file = std::fs::read_to_string("file.013.dat")?;

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



    Ok(())
}
*/

fn store1<'a, 'b>(to: &str, y: &[f64]) -> Result<()> {
    let mut out = BufWriter::with_capacity(2usize.pow(16), std::fs::File::create(to)?);
    for y in y.iter() {
        writeln!(&mut out, "{}", y)?;
    }
    Ok(())
}

fn store<'a, 'b>(to: &str, x: &[f64], y: &[f64]) -> Result<()> {
    assert_eq!(x.len(), y.len());
    let mut out = BufWriter::with_capacity(2usize.pow(16), std::fs::File::create(to)?);
    for (x, y) in x.iter().zip(y.iter()) {
        writeln!(&mut out, "{},{}", x, y)?;
    }
    Ok(())
}
