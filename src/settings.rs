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
    pub fft: Option<Fft>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Project {
    pub title: String,
    pub files: Vec<File>,
    #[serde(default = "all_output")]
    pub output: Vec<Output>,
    #[serde(default = "all_output")]
    pub gnuplot: Vec<Output>,
    #[serde(default = "_true")]
    pub threading: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Output {
    Raw,
    PreInterpolation,
    PostInterpolation,
    Fft,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct File {
    pub source: String,
    pub dest: String,
    #[serde(default)]
    pub masks: Vec<Mask>,
    // File specific prefactor override
    pub prefactor_x: Option<f64>,
    pub prefactor_y: Option<f64>,
    #[serde(default)]
    pub impulse_filter: u32,
    #[serde(default = "one")]
    pub impulse_tuning: f64,
}

#[derive(Copy, Clone, Debug, PartialEq, Deserialize)]
pub struct Mask {
    pub left: f64,
    pub right: f64,
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
    #[serde(default = "one")]
    pub prefactor_x: f64,
    #[serde(default = "one")]
    pub prefactor_y: f64,
    #[serde(default)]
    pub invert_x: bool,
    pub interpolation: Option<InterpolationAlgorithm>,
    pub interpolation_n: Option<String>,
    pub trim_left: Option<f64>,
    pub trim_right: Option<f64>,
    #[serde(default)]
    pub derivative: u32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InterpolationAlgorithm {
    Linear,
    Steffen,
}

impl fmt::Display for InterpolationAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InterpolationAlgorithm::Linear => write!(f, "linear"),
            InterpolationAlgorithm::Steffen => write!(f, "Steffen spline"),
        }
    }
}

#[allow(clippy::from_over_into)]
impl Into<gsl_rust::interpolation::Algorithm> for InterpolationAlgorithm {
    fn into(self) -> gsl_rust::interpolation::Algorithm {
        match self {
            InterpolationAlgorithm::Linear => gsl_rust::interpolation::Algorithm::Linear,
            InterpolationAlgorithm::Steffen => gsl_rust::interpolation::Algorithm::Steffen,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Fft {
    pub zero_pad: String,
    #[serde(default = "_true")]
    pub cuda: bool,
    #[serde(default)]
    pub center: bool,
    #[serde(default = "_true")]
    pub hann: bool,
    pub truncate_lower: Option<f64>,
    pub truncate_upper: Option<f64>,
}

fn _true() -> bool {
    true
}

fn one() -> f64 {
    1.0
}

fn all_output() -> Vec<Output> {
    vec![
        Output::Raw,
        Output::PreInterpolation,
        Output::PostInterpolation,
        Output::Fft,
    ]
}
