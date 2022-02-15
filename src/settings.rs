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

use crate::has_dup;
use anyhow::anyhow;
use anyhow::ensure;
use anyhow::Result;
use itertools::Itertools;
use serde::de::IntoDeserializer;
use serde::Deserialize;
use std::fmt;
use std::path::Path;
use toml::Value;

pub const INTERP_OPTION_MIN_VAR: &str = "minvar";
pub const INTERP_OPTION_MIN: &str = "min";

pub fn load<P: AsRef<Path>>(path: P) -> Result<Template> {
    let s = std::fs::read_to_string(path)?;
    let root = toml::from_str::<Value>(&s)?;

    let project = root
        .get("project")
        .ok_or(anyhow!("project settings missing"))?;
    let project = <Project as Deserialize>::deserialize(project.to_owned().into_deserializer())?;

    let preprocessing_global = root
        .get("preprocessing")
        .and_then(|preprocessing| preprocessing.get("global"));
    let preprocessing_global = match preprocessing_global {
        Some(preprocessing_global) => <Preprocessing as Deserialize>::deserialize(
            preprocessing_global.to_owned().into_deserializer(),
        )?,
        None => toml::from_str::<'_, Preprocessing>("")
            .expect("preprocessing fields dont all have defaults"),
    };

    let processing_global = root
        .get("processing")
        .and_then(|processing| processing.get("global"));
    let processing_global = match processing_global {
        Some(processing_global) => <Processing as Deserialize>::deserialize(
            processing_global.to_owned().into_deserializer(),
        )?,
        None => {
            toml::from_str::<'_, Processing>("").expect("processing fields dont all have defaults")
        }
    };

    let fft = root.get("fft");
    let fft = match fft {
        Some(fft) => Some(<Fft as Deserialize>::deserialize(
            fft.to_owned().into_deserializer(),
        )?),
        None => None,
    };

    let files = root.get("file");
    let files = match files {
        Some(files) => {
            <Vec<File> as Deserialize>::deserialize(files.to_owned().into_deserializer())?
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
                    .ok_or(anyhow!("rename entry invalid: expected string"))?
                    .to_owned();
                Ok(Rename { from, to })
            })
            .collect::<Result<Vec<_>>>()?,
        None => vec![],
    };

    let extract = root.get("extract");
    let extract = extract
        .ok_or(anyhow!("extraction settings missing"))?
        .as_table()
        .ok_or(anyhow!("extraction settings invalid: expected table"))?;
    let extract = extract
        .iter()
        .map(|(name, value)| {
            let name = name.to_owned();
            let table = value
                .as_table()
                .ok_or(anyhow!("extraction entry invalid: expected table"))?;
            let x = table
                .get("x")
                .ok_or(anyhow!("extraction entry missing x"))?
                .as_str()
                .ok_or(anyhow!("extraction extry invalid: expected x as string"))?
                .to_owned();
            let y = table
                .get("y")
                .ok_or(anyhow!("extraction entry missing y"))?
                .as_str()
                .ok_or(anyhow!("extraction extry invalid: expected y as string"))?
                .to_owned();
            Ok(Extract { name, x, y })
        })
        .collect::<Result<Vec<_>>>()?;

    // Build individual settings
    let mut settings = files
        .iter()
        .cartesian_product(extract.iter())
        .map(|(file, extract)| Settings {
            file: file.to_owned(),
            extract: extract.to_owned(),
            preprocessing: preprocessing_global.clone(),
            processing: processing_global.clone(),
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
            )?;
        }
        if let Some(specific) = processing {
            update_processing(
                settings
                    .iter_mut()
                    .filter(|settings| settings.extract.name == var),
                specific,
            )?;
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
            )?;
        }
        if let Some(specific) = processing {
            update_processing(
                settings
                    .iter_mut()
                    .filter(|settings| settings.file.dest == dest),
                specific,
            )?;
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
                )?;
            }
            if let Some(specific) = processing {
                update_processing(
                    settings
                        .iter_mut()
                        .filter(|settings| settings.file.dest == dest)
                        .filter(|settings| settings.extract.name == var),
                    specific,
                )?;
            }
        }
    }

    Ok(Template {
        project,
        rename,
        files,
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
                <_ as Deserialize>::deserialize(impulse_filter.to_owned().into_deserializer())?;
        }
        if let Some(impulse_tuning) = specific.get("impulse_tuning") {
            settings.preprocessing.impulse_tuning =
                <_ as Deserialize>::deserialize(impulse_tuning.to_owned().into_deserializer())?;
        }
        if let Some(masks) = specific.get("masks") {
            settings.preprocessing.masks =
                <_ as Deserialize>::deserialize(masks.to_owned().into_deserializer())?;
        }
        if let Some(trim_left) = specific.get("trim_left") {
            settings.preprocessing.trim_left = Some(<_ as Deserialize>::deserialize(
                trim_left.to_owned().into_deserializer(),
            )?);
        }
        if let Some(trim_right) = specific.get("trim_right") {
            settings.preprocessing.trim_right = Some(<_ as Deserialize>::deserialize(
                trim_right.to_owned().into_deserializer(),
            )?);
        }
        if let Some(prefactor_x) = specific.get("prefactor_x") {
            settings.preprocessing.prefactor_x =
                <_ as Deserialize>::deserialize(prefactor_x.to_owned().into_deserializer())?;
        }
        if let Some(prefactor_y) = specific.get("prefactor_y") {
            settings.preprocessing.prefactor_y =
                <_ as Deserialize>::deserialize(prefactor_y.to_owned().into_deserializer())?;
        }
        if let Some(invert_x) = specific.get("invert_x") {
            settings.preprocessing.invert_x =
                <_ as Deserialize>::deserialize(invert_x.to_owned().into_deserializer())?;
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
            settings.processing.interpolation = Some(<_ as Deserialize>::deserialize(
                interpolation.to_owned().into_deserializer(),
            )?);
        }
        if let Some(interpolation_n) = specific.get("interpolation_n") {
            settings.processing.interpolation_n = Some(<_ as Deserialize>::deserialize(
                interpolation_n.to_owned().into_deserializer(),
            )?);
        }
        if let Some(derivative) = specific.get("derivative") {
            settings.processing.derivative =
                <_ as Deserialize>::deserialize(derivative.to_owned().into_deserializer())?;
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct Template {
    pub project: Project,
    pub rename: Vec<Rename>,
    pub files: Vec<File>,
    pub settings: Vec<Settings>,
    pub fft: Option<Fft>,
}

impl Template {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        load(path)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Project {
    pub title: String,
    #[serde(default = "_true")]
    pub plot: bool,
    #[serde(default)]
    pub format: Format,
    #[serde(default = "_true")]
    pub threading: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Deserialize)]
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

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub file: File,
    pub extract: Extract,
    pub preprocessing: Preprocessing,
    pub processing: Processing,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Deserialize)]
pub struct File {
    pub source: String,
    pub dest: String,
}

#[derive(Copy, Clone, Debug, PartialEq, Deserialize)]
pub struct Mask {
    pub left: f64,
    pub right: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Rename {
    pub from: String,
    pub to: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Extract {
    pub name: String,
    pub x: String,
    pub y: String,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Preprocessing {
    #[serde(default)]
    pub impulse_filter: u32,
    #[serde(default = "one")]
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

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Processing {
    pub interpolation: Option<InterpolationAlgorithm>,
    pub interpolation_n: Option<String>,
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
    #[serde(default = "_true")]
    pub center: bool,
    #[serde(default = "_true")]
    pub hann: bool,
    pub truncate_lower: Option<f64>,
    pub truncate_upper: Option<f64>,
    #[serde(default)]
    pub sweep: FftSweep,
    pub sweep_steps: Option<usize>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Deserialize)]
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
