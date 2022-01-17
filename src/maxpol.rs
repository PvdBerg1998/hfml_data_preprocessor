use anyhow::Result;
use sled::Db;

pub fn db() -> Result<Db> {
    Ok(sled::Config::default()
        .path("maxpol_coefficients")
        .use_compression(true)
        .mode(sled::Mode::LowSpace)
        .open()?)
}

pub fn load_kernel(db: &sled::Db, n: usize, l: usize, filter_factor: f64) -> Result<Box<[f64]>> {
    let p = ((n as f64 + (1.0 - filter_factor) * 2.0 * (l - n) as f64).round() as usize)
        .max(n)
        .min(2 * l);

    let key = bincode::serialize::<(usize, usize, usize)>(&(n, l, p)).unwrap();
    let coeff = bincode::deserialize::<Vec<f64>>(
        &db.get(&key)?
            .ok_or_else(|| anyhow::anyhow!("value does not exist in database"))?,
    )?;
    Ok(coeff.into_boxed_slice())
}

pub fn load_raw(s: &str) -> Box<[f64]> {
    s.lines()
        .map(|line| fast_float::parse(line).expect("parse error"))
        .collect()
}

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