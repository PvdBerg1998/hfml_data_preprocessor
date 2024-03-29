/*
    setting.rs
    Copyright (C) 2022 Pim van den Berg

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

use crate::has_dup;
use anyhow::anyhow;
use anyhow::ensure;
use anyhow::Context;
use anyhow::Result;
use gsl_rust::interpolation::Derivative;
use itertools::Itertools;
use serde::de::IntoDeserializer;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::Serializer;
use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use toml::Value;

const INTERP_OPTION_MIN_VAR: &str = "minvar";
const INTERP_OPTION_MIN_GLOBAL: &str = "min";

pub fn load<P: AsRef<Path>>(path: P) -> Result<Template> {
    let s = std::fs::read_to_string(path).context("Failed to read template file")?;
    let root = toml::from_str::<Value>(&s).context("Failed to deserialize TOML root")?;

    let project = root
        .get("project")
        .ok_or_else(|| anyhow!("Project settings missing"))?;
    let project = <Project as Deserialize>::deserialize(project.to_owned().into_deserializer())
        .context("Failed to deserialize project section")?;

    let preprocessing_global = root
        .get("preprocessing")
        .and_then(|preprocessing| preprocessing.get("global"));
    let preprocessing_global = match preprocessing_global {
        Some(preprocessing_global) => <Preprocessing as Deserialize>::deserialize(
            preprocessing_global.to_owned().into_deserializer(),
        )
        .context("Failed to deserialize global preprocessing section")?,
        None => toml::from_str::<'_, Preprocessing>("")
            .expect("preprocessing fields dont all have defaults"),
    };

    let processing_global = root
        .get("processing")
        .and_then(|processing| processing.get("global"));
    let processing_global = match processing_global {
        Some(processing_global) => <Processing as Deserialize>::deserialize(
            processing_global.to_owned().into_deserializer(),
        )
        .context("Failed to deserialize global processing section")?,
        None => {
            toml::from_str::<'_, Processing>("").expect("processing fields dont all have defaults")
        }
    };

    let fft = root.get("fft");
    let fft = match fft {
        Some(fft) => Some(
            <Fft as Deserialize>::deserialize(fft.to_owned().into_deserializer())
                .context("Failed to deserialize FFT section")?,
        ),
        None => None,
    };

    let files = root.get("file");
    let files = match files {
        Some(files) => {
            <Vec<File> as Deserialize>::deserialize(files.to_owned().into_deserializer())
                .context("Failed to deserialize files")?
        }
        None => vec![],
    };

    // Check uniqueness of destinations
    let dest_names = files
        .iter()
        .map(|file| file.dest.as_str())
        .collect::<Vec<_>>();
    ensure!(
        !has_dup(&dest_names),
        "file destinations may not contain duplicates"
    );

    let rename = root.get("rename");
    let rename = match rename.and_then(|rename| rename.as_table()) {
        Some(rename) => rename
            .iter()
            .map(|(from, to)| {
                let from = from.to_owned();
                let to = to
                    .as_str()
                    .ok_or_else(|| anyhow!("Rename entry invalid: expected string"))?
                    .to_owned();
                Ok(Rename { from, to })
            })
            .filter_ok(|rename| rename.from != rename.to)
            .collect::<Result<Vec<_>>>()
            .context("Failed to deserialize rename section")?,
        None => vec![],
    };

    let extract = root.get("extract");
    let extract = extract
        .ok_or_else(|| anyhow!("Extraction settings missing"))?
        .as_table()
        .ok_or_else(|| anyhow!("Extraction settings invalid: expected table"))?;
    let extract = extract
        .iter()
        .map(|(name, value)| {
            let name = name.to_owned();
            let table = value
                .as_table()
                .ok_or_else(|| anyhow!("Extraction entry invalid: expected table"))?;
            let x = table
                .get("x")
                .ok_or_else(|| anyhow!("Extraction entry missing x"))?
                .as_str()
                .ok_or_else(|| anyhow!("Extraction extry invalid: expected x as string"))?
                .to_owned();
            let y = table
                .get("y")
                .ok_or_else(|| anyhow!("Extraction entry missing y"))?
                .as_str()
                .ok_or_else(|| anyhow!("Extraction extry invalid: expected y as string"))?
                .to_owned();
            Ok(Extract { name, x, y })
        })
        .collect::<Result<Vec<_>>>()
        .context("Failed to deserialize extract section")?;

    // Build individual settings
    let mut settings = files
        .iter()
        .cartesian_product(extract.iter())
        .map(|(file, extract)| Settings {
            file: file.to_owned(),
            extract: extract.to_owned(),
            preprocessing: preprocessing_global.clone(),
            processing: processing_global,
        })
        .collect::<Vec<_>>();

    /*

    Individual settings ordering:
    [processing.global]
    [processing.Vxx]
    [processing."file.013.dat"]
    [processing."file.013.dat".Vxx]

    */

    // [processing.Vxx]
    for var in extract.iter().map(|extract| extract.name.as_str()) {
        let preprocessing = root
            .get("preprocessing")
            .and_then(|preprocessing| preprocessing.get(var));

        let processing = root
            .get("processing")
            .and_then(|processing| processing.get(var));

        if let Some(specific) = preprocessing {
            update_preprocessing(
                settings
                    .iter_mut()
                    .filter(|settings| settings.extract.name == var),
                specific,
            )
            .with_context(|| {
                format!("Failed to deserialize preprocessing specialisation for variable {var}")
            })?;
        }
        if let Some(specific) = processing {
            update_processing(
                settings
                    .iter_mut()
                    .filter(|settings| settings.extract.name == var),
                specific,
            )
            .with_context(|| {
                format!("Failed to deserialize processing specialisation for variable {var}")
            })?;
        }
    }

    // [processing."dir1/measurement_013"]
    for dest in files.iter().map(|file| file.dest.as_str()) {
        let preprocessing = root
            .get("preprocessing")
            .and_then(|preprocessing| preprocessing.get(dest));

        let processing = root
            .get("processing")
            .and_then(|processing| processing.get(dest));

        if let Some(specific) = preprocessing {
            update_preprocessing(
                settings
                    .iter_mut()
                    .filter(|settings| settings.file.dest == dest),
                specific,
            )
            .with_context(|| {
                format!("Failed to deserialize preprocessing specialisation for dest {dest}")
            })?;
        }
        if let Some(specific) = processing {
            update_processing(
                settings
                    .iter_mut()
                    .filter(|settings| settings.file.dest == dest),
                specific,
            )
            .with_context(|| {
                format!("Failed to deserialize processing specialisation for dest {dest}")
            })?;
        }
    }

    // [processing."dir1/measurement_013".Vxx]
    for dest in files.iter().map(|file| file.dest.as_str()) {
        for var in extract.iter().map(|extract| extract.name.as_str()) {
            let preprocessing = root
                .get("preprocessing")
                .and_then(|preprocessing| preprocessing.get(dest))
                .and_then(|specific| specific.get(var));

            let processing = root
                .get("processing")
                .and_then(|processing| processing.get(dest))
                .and_then(|specific| specific.get(var));

            if let Some(specific) = preprocessing {
                update_preprocessing(
                    settings
                        .iter_mut()
                        .filter(|settings| settings.file.dest == dest)
                        .filter(|settings| settings.extract.name == var),
                    specific,
                ).with_context(|| format!("Failed to deserialize preprocessing specialisation for dest {dest} variable {var}"))?;
            }
            if let Some(specific) = processing {
                update_processing(
                    settings
                        .iter_mut()
                        .filter(|settings| settings.file.dest == dest)
                        .filter(|settings| settings.extract.name == var),
                    specific,
                ).with_context(|| format!("Failed to deserialize processing specialisation for dest {dest} variable {var}"))?;
            }
        }
    }

    Ok(Template {
        project,
        rename,
        files,
        extract,
        settings,
        fft,
    })
}

fn update_preprocessing<'a>(
    iter: impl Iterator<Item = &'a mut Settings>,
    specific: &toml::Value,
) -> Result<()> {
    for settings in iter {
        if let Some(impulse_filter) = specific.get("impulse_filter") {
            settings.preprocessing.impulse_filter =
                <_ as Deserialize>::deserialize(impulse_filter.to_owned().into_deserializer())
                    .context("Failed to deserialize impulse_filter")?;
        }
        if let Some(impulse_tuning) = specific.get("impulse_tuning") {
            settings.preprocessing.impulse_tuning =
                deserialize_impulse_tuning(impulse_tuning.to_owned().into_deserializer())
                    .context("Failed to deserialize impulse_tuning")?;
        }
        if let Some(masks) = specific.get("masks") {
            settings.preprocessing.masks =
                <_ as Deserialize>::deserialize(masks.to_owned().into_deserializer())
                    .context("Failed to deserialize masks")?;
        }
        if let Some(trim_left) = specific.get("trim_left") {
            settings.preprocessing.trim_left = Some(
                <_ as Deserialize>::deserialize(trim_left.to_owned().into_deserializer())
                    .context("Failed to deserialize trim_left")?,
            );
        }
        if let Some(trim_right) = specific.get("trim_right") {
            settings.preprocessing.trim_right = Some(
                <_ as Deserialize>::deserialize(trim_right.to_owned().into_deserializer())
                    .context("Failed to deserialize trim_right")?,
            );
        }
        if let Some(prefactor_x) = specific.get("prefactor_x") {
            settings.preprocessing.prefactor_x =
                <_ as Deserialize>::deserialize(prefactor_x.to_owned().into_deserializer())
                    .context("Failed to deserialize prefactor_x")?;
        }
        if let Some(prefactor_y) = specific.get("prefactor_y") {
            settings.preprocessing.prefactor_y =
                <_ as Deserialize>::deserialize(prefactor_y.to_owned().into_deserializer())
                    .context("Failed to deserialize prefactor_y")?;
        }
        if let Some(invert_x) = specific.get("invert_x") {
            settings.preprocessing.invert_x =
                <_ as Deserialize>::deserialize(invert_x.to_owned().into_deserializer())
                    .context("Failed to deserialize invert_x")?;
        }
    }
    Ok(())
}

fn update_processing<'a>(
    iter: impl Iterator<Item = &'a mut Settings>,
    specific: &toml::Value,
) -> Result<()> {
    for settings in iter {
        if let Some(interpolation) = specific.get("interpolation") {
            settings.processing.interpolation = Some(
                <_ as Deserialize>::deserialize(interpolation.to_owned().into_deserializer())
                    .context("Failed to deserialize interpolation")?,
            );
        }
        if let Some(interpolation_n) = specific.get("interpolation_n") {
            settings.processing.interpolation_n =
                <_ as Deserialize>::deserialize(interpolation_n.to_owned().into_deserializer())
                    .context("Failed to deserialize interpolation_n")?;
        }
        if let Some(derivative) = specific.get("derivative") {
            settings.processing.derivative =
                deserialize_derivative(derivative.to_owned().into_deserializer())
                    .context("Failed to deserialize derivative")?;
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct Template {
    pub project: Project,
    pub rename: Vec<Rename>,
    pub files: Vec<File>,
    pub extract: Vec<Extract>,
    pub settings: Vec<Settings>,
    pub fft: Option<Fft>,
}

impl Template {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        load(path)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Project {
    #[serde(deserialize_with = "deserialize_string_sanitized")]
    pub title: String,
    #[serde(default = "_true")]
    pub plot: bool,
    #[serde(default)]
    pub format: Format,
    #[serde(default = "_true")]
    pub threading: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Csv,
    MessagePack,
}

impl Default for Format {
    fn default() -> Self {
        Format::MessagePack
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Settings {
    pub file: File,
    pub extract: Extract,
    pub preprocessing: Preprocessing,
    pub processing: Processing,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct File {
    pub source: String,
    #[serde(deserialize_with = "deserialize_string_sanitized")]
    pub dest: String,
    #[serde(flatten)]
    pub metadata: HashMap<String, Value>,
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mask {
    pub left: f64,
    pub right: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rename {
    pub from: String,
    pub to: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Extract {
    #[serde(deserialize_with = "deserialize_string_sanitized")]
    pub name: String,
    pub x: String,
    pub y: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Preprocessing {
    #[serde(default)]
    pub impulse_filter: u32,
    #[serde(default = "one")]
    #[serde(deserialize_with = "deserialize_impulse_tuning")]
    // Invariant: > 0.0
    pub impulse_tuning: f64,
    #[serde(default)]
    pub masks: Vec<Mask>,
    pub trim_left: Option<f64>,
    pub trim_right: Option<f64>,
    #[serde(default = "one")]
    pub prefactor_x: f64,
    #[serde(default = "one")]
    pub prefactor_y: f64,
    #[serde(default)]
    pub invert_x: bool,
}

fn deserialize_impulse_tuning<'de, D: Deserializer<'de>>(de: D) -> Result<f64, D::Error> {
    let tuning = f64::deserialize(de)?;
    if tuning < 0.0 {
        return Err(serde::de::Error::custom("Impulse tuning must be positive"));
    }
    Ok(tuning)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Processing {
    #[serde(default)]
    pub interpolation: Option<InterpolationAlgorithm>,
    #[serde(default)]
    pub interpolation_n: InterpolationLength,
    #[serde(serialize_with = "serialize_derivative")]
    #[serde(deserialize_with = "deserialize_derivative")]
    #[serde(default = "no_derivative")]
    pub derivative: Derivative,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InterpolationAlgorithm {
    Linear,
    Steffen,
}

impl From<InterpolationAlgorithm> for gsl_rust::interpolation::Algorithm {
    fn from(val: InterpolationAlgorithm) -> Self {
        match val {
            InterpolationAlgorithm::Linear { .. } => gsl_rust::interpolation::Algorithm::Linear,
            InterpolationAlgorithm::Steffen { .. } => gsl_rust::interpolation::Algorithm::Steffen,
        }
    }
}

impl fmt::Display for InterpolationAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InterpolationAlgorithm::Linear { .. } => write!(f, "linear"),
            InterpolationAlgorithm::Steffen { .. } => write!(f, "Steffen spline"),
        }
    }
}

fn serialize_derivative<S: Serializer>(val: &Derivative, ser: S) -> Result<S::Ok, S::Error> {
    match val {
        Derivative::None => ser.serialize_u32(0),
        Derivative::First => ser.serialize_u32(1),
        Derivative::Second => ser.serialize_u32(2),
    }
}

fn deserialize_derivative<'de, D: Deserializer<'de>>(de: D) -> Result<Derivative, D::Error> {
    let n = u32::deserialize(de)?;
    Ok(match n {
        0 => Derivative::None,
        1 => Derivative::First,
        2 => Derivative::Second,
        _ => {
            return Err(serde::de::Error::custom(
                "Only the 0th, 1st and 2nd derivative are supported",
            ))
        }
    })
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InterpolationLength {
    Amount(u64),
    MinimumPerVariable,
    Minimum,
}

impl Default for InterpolationLength {
    fn default() -> Self {
        InterpolationLength::Minimum
    }
}

impl Serialize for InterpolationLength {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        match *self {
            InterpolationLength::Amount(n) => {
                if n.is_power_of_two() {
                    let n_log2 = (n as f64).log2().floor() as u64;
                    ser.serialize_str(&format!("2^{n_log2}"))
                } else {
                    ser.serialize_u64(n)
                }
            }
            InterpolationLength::MinimumPerVariable => ser.serialize_str(INTERP_OPTION_MIN_VAR),
            InterpolationLength::Minimum => ser.serialize_str(INTERP_OPTION_MIN_GLOBAL),
        }
    }
}

impl<'de> Deserialize<'de> for InterpolationLength {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<InterpolationLength, D::Error> {
        let n = String::deserialize(de)?;
        if n == INTERP_OPTION_MIN_VAR {
            Ok(InterpolationLength::MinimumPerVariable)
        } else if n == INTERP_OPTION_MIN_GLOBAL {
            Ok(InterpolationLength::Minimum)
        } else {
            match n.parse::<u64>() {
                Ok(n) => Ok(InterpolationLength::Amount(n)),
                Err(_) => Ok(InterpolationLength::Amount(
                    2u64.pow(parse_log2(&n).map_err(serde::de::Error::custom)?),
                )),
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Fft {
    #[serde(serialize_with = "serialize_zero_pad")]
    #[serde(deserialize_with = "deserialize_zero_pad")]
    pub zero_pad: usize,
    #[serde(default = "_true")]
    pub cuda: bool,
    #[serde(default = "_true")]
    pub center: bool,
    #[serde(default = "_true")]
    pub hann: bool,
    pub truncate_lower: Option<f64>,
    pub truncate_upper: Option<f64>,
    #[serde(default)]
    pub sweep: FftSweep,
    #[serde(default)]
    #[serde(deserialize_with = "deserialize_sweep_steps")]
    // Invariant: > 1
    pub sweep_steps: usize,
}

fn serialize_zero_pad<S: Serializer>(val: &usize, ser: S) -> Result<S::Ok, S::Error> {
    let n = *val;
    if n.is_power_of_two() {
        let n_log2 = (n as f64).log2().floor() as usize;
        ser.serialize_str(&format!("2^{n_log2}"))
    } else {
        ser.serialize_u64(n as u64)
    }
}

fn deserialize_zero_pad<'de, D: Deserializer<'de>>(de: D) -> Result<usize, D::Error> {
    let n = String::deserialize(de)?;
    match n.parse::<usize>() {
        Ok(n) => Ok(n),
        Err(_) => Ok(2usize.pow(parse_log2(&n).map_err(serde::de::Error::custom)?)),
    }
}

fn deserialize_sweep_steps<'de, D: Deserializer<'de>>(de: D) -> Result<usize, D::Error> {
    let n = usize::deserialize(de)?;
    if n <= 1 {
        return Err(serde::de::Error::custom(
            "Sweep steps must be larger than one",
        ));
    }
    Ok(n)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FftSweep {
    Full,
    Lower,
    Upper,
    Windows,
}

impl Default for FftSweep {
    fn default() -> Self {
        FftSweep::Full
    }
}

fn _true() -> bool {
    true
}

fn one() -> f64 {
    1.0
}

fn no_derivative() -> Derivative {
    Derivative::None
}

fn deserialize_string_sanitized<'de, D: Deserializer<'de>>(de: D) -> Result<String, D::Error> {
    let s = String::deserialize(de)?;

    // Should be non-empty
    if s.is_empty() {
        return Err(serde::de::Error::custom("String must be non-empty"));
    }

    // No weird characters allowed
    if !s.is_ascii() {
        return Err(serde::de::Error::custom(format!(
            "String '{s}' may only contain ASCII"
        )));
    }

    // Check if there are no escape characters at the ends
    if s.starts_with('.')
        || s.ends_with('.')
        || s.starts_with('/')
        || s.starts_with('\\')
        || s.ends_with('/')
        || s.ends_with('\\')
    {
        return Err(serde::de::Error::custom(
            "String may not contain escape sequences",
        ));
    }

    // Replace whitespace with _
    let s = s.replace(' ', "_");
    let s = s.replace('\t', "_");
    let s = s.replace('\r', "");
    let s = s.replace('\n', "_");

    Ok(s)
}

fn parse_log2(n: &str) -> Result<u32, String> {
    if let Some(stripped) = n.strip_prefix("2^") {
        match stripped.parse::<u32>() {
            Ok(n_log2) => Ok(n_log2),
            Err(_) => Err(format!("Invalid power of 2: {n}")),
        }
    } else {
        Err(format!("Invalid value: {n}"))
    }
}
