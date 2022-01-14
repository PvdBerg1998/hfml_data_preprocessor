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
