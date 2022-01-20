/*
    setting.rs
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

use anyhow::Result;
use num_traits::one;
use serde::Deserialize;
use std::fmt;
use std::{collections::HashMap, path::Path};

pub fn load<P: AsRef<Path>>(path: P) -> Result<Settings> {
    let s = std::fs::read_to_string(path)?;
    Ok(toml::from_str(&s)?)
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Settings {
    pub project: Project,
    pub rename: Rename,
    pub extract: Extract,
    pub preprocessing: Preprocessing,
    pub processing: Option<Processing>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Project {
    pub files: Vec<File>,
    pub output: Vec<Output>,
    pub threading: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Output {
    Raw,
    Intermediate,
    Final,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct File {
    pub source: String,
    pub dest: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Rename {
    #[serde(flatten)]
    pub columns: HashMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Extract {
    #[serde(flatten)]
    pub pairs: HashMap<String, HashMap<String, String>>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Preprocessing {
    #[serde(default)]
    pub prefactor: Prefactor,
    #[serde(default)]
    pub invert_x: bool,
    pub interpolation: Option<Interpolation>,
    pub trim_left: f64,
    pub trim_right: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Interpolation {
    pub n: String,
    #[serde(default)]
    pub algorithm: Algorithm,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Algorithm {
    Linear,
    Steffen,
}

impl fmt::Display for Algorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Algorithm::Linear => write!(f, "linear"),
            Algorithm::Steffen => write!(f, "Steffen spline"),
        }
    }
}

impl Default for Algorithm {
    fn default() -> Self {
        Algorithm::Linear
    }
}

#[allow(clippy::from_over_into)]
impl Into<gsl_rust::interpolation::Algorithm> for Algorithm {
    fn into(self) -> gsl_rust::interpolation::Algorithm {
        match self {
            Algorithm::Linear => gsl_rust::interpolation::Algorithm::Linear,
            Algorithm::Steffen => gsl_rust::interpolation::Algorithm::Steffen,
        }
    }
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

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Processing {
    pub kind: ProcessingKind,
    pub fft: Option<Fft>,
}

#[derive(Copy, Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProcessingKind {
    Fft,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Fft {
    pub zero_pad: String,
    pub cuda: bool,
    pub center: bool,
    pub hann: bool,
    pub truncate_lower: Option<f64>,
    pub truncate_upper: Option<f64>,
}
