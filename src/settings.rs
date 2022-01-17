use anyhow::anyhow;
use anyhow::Result;
use num_traits::one;
use serde::Deserialize;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

pub fn load<P: AsRef<Path>>(path: P) -> Result<Settings> {
    let s = std::fs::read_to_string(path)?;
    Ok(toml::from_str(&s)?)
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Settings {
    pub project: Project,
    pub rename: Rename,
    pub extract: Extract,
    #[serde(default)]
    pub preprocessing: Preprocessing,
    pub processing: Option<Processing>,
}

impl Settings {
    pub fn paths(&self) -> Result<Vec<PathBuf>> {
        self.project.paths()
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Project {
    pub title: String,
    pub files: Vec<String>,
}

impl Project {
    pub fn paths(&self) -> Result<Vec<PathBuf>> {
        let paths = self
            .files
            .iter()
            .map(|f| {
                if f.contains("..") {
                    // assume its a range
                    let mut split = f.split("..");
                    let a = split
                        .next()
                        .ok_or_else(|| anyhow!("Invalid file range"))?
                        .parse()?;
                    let b = split
                        .next()
                        .ok_or_else(|| anyhow!("Invalid file range"))?
                        .parse()?;
                    Ok((a..b)
                        .map(|i: usize| PathBuf::from(format!("file.{i:0>3}.dat")))
                        .collect::<Vec<_>>())
                } else {
                    // try to parse as an index
                    let a = f.parse::<usize>();
                    match a {
                        Ok(a) => Ok(vec![PathBuf::from(format!("file.{a:0>3}.dat"))]),
                        Err(_) => {
                            // assume its a full path
                            Ok(vec![PathBuf::from(f)])
                        }
                    }
                }
            })
            .collect::<Result<Vec<Vec<_>>>>();
        Ok(paths?.into_iter().flatten().collect())
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Rename {
    #[serde(flatten)]
    pub columns: HashMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Extract {
    #[serde(flatten)]
    pub pairs: HashMap<String, HashMap<String, String>>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct Preprocessing {
    #[serde(default)]
    pub prefactor: Prefactor,
    #[serde(default)]
    pub invert_x: bool,
    #[serde(default)]
    pub interpolation_log2: u16,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Prefactor {
    #[serde(default = "one")]
    pub x: f64,
    #[serde(default = "one")]
    pub y: f64,
}

impl Default for Prefactor {
    fn default() -> Self {
        Prefactor { x: 1.0, y: 1.0 }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProcessingKind {
    FFT,
    Symmetrize,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Processing {
    pub kind: Option<ProcessingKind>,
    pub fft: Option<FFT>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct FFT {
    pub zero_pad_log2: u16,
    pub cuda: bool,
    pub center: bool,
}
