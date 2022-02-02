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

pub fn load<P: AsRef<Path>>(path: P) -> Result<Template> {
    let s = std::fs::read_to_string(path)?;
    let root = toml::from_str::<Value>(&s)?;

    let project = root
        .get("project")
        .ok_or(anyhow!("project settings missing"))?;
    let project = <Project as Deserialize>::deserialize(project.to_owned().into_deserializer())?;

    let preprocessing_default = toml::from_str::<'_, Preprocessing>("")
        .expect("preprocessing fields dont all have defaults");
    let processing_default =
        toml::from_str::<'_, Processing>("").expect("processing fields dont all have defaults");

    let preprocessing_global = root
        .get("preprocessing")
        .and_then(|preprocessing| preprocessing.get("global"));
    let preprocessing_global = match preprocessing_global {
        Some(preprocessing_global) => <Preprocessing as Deserialize>::deserialize(
            preprocessing_global.to_owned().into_deserializer(),
        )?,
        None => preprocessing_default.clone(),
    };

    let processing_global = root
        .get("processing")
        .and_then(|processing| processing.get("global"));
    let processing_global = match processing_global {
        Some(processing_global) => <Processing as Deserialize>::deserialize(
            processing_global.to_owned().into_deserializer(),
        )?,
        None => processing_default.clone(),
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
        let preprocessing = if_chain::if_chain! {
            if let Some(preprocessing) = root.get("preprocessing");
            if let Some(specific) = preprocessing.get(var);
            then {
                Some(<Preprocessing as Deserialize>::deserialize(
                    specific.to_owned().into_deserializer(),
                )?)
            } else {
                None
            }
        };

        let processing = if_chain::if_chain! {
            if let Some(processing) = root.get("processing");
            if let Some(specific) = processing.get(var);
            then {
                Some(<Processing as Deserialize>::deserialize(
                    specific.to_owned().into_deserializer(),
                )?)
            } else {
                None
            }
        };

        if let Some(specific) = preprocessing {
            update_preprocessing(
                settings
                    .iter_mut()
                    .filter(|settings| settings.extract.name == var),
                &specific,
                &preprocessing_default,
            );
        }
        if let Some(specific) = processing {
            update_processing(
                settings
                    .iter_mut()
                    .filter(|settings| settings.extract.name == var),
                &specific,
                &processing_default,
            );
        }
    }

    // [processing."file.013.dat"]
    for source in files.iter().map(|file| file.source.as_str()) {
        let preprocessing = if_chain::if_chain! {
            if let Some(preprocessing) = root.get("preprocessing");
            if let Some(specific) = preprocessing.get(source);
            then {
                Some(<Preprocessing as Deserialize>::deserialize(
                    specific.to_owned().into_deserializer(),
                )?)
            } else {
                None
            }
        };

        let processing = if_chain::if_chain! {
            if let Some(processing) = root.get("processing");
            if let Some(specific) = processing.get(source);
            then {
                Some(<Processing as Deserialize>::deserialize(
                    specific.to_owned().into_deserializer(),
                )?)
            } else {
                None
            }
        };

        if let Some(specific) = preprocessing {
            update_preprocessing(
                settings
                    .iter_mut()
                    .filter(|settings| settings.file.source == source),
                &specific,
                &preprocessing_default,
            );
        }
        if let Some(specific) = processing {
            update_processing(
                settings
                    .iter_mut()
                    .filter(|settings| settings.file.source == source),
                &specific,
                &processing_default,
            );
        }
    }

    // [processing."file.013.dat".Vxx]
    for source in files.iter().map(|file| file.source.as_str()) {
        for var in extract.iter().map(|extract| extract.name.as_str()) {
            let preprocessing = if_chain::if_chain! {
                if let Some(preprocessing) = root.get("preprocessing");
                if let Some(table) = preprocessing.get(source);
                if let Some(specific) = table.get(var);
                then {
                    Some(<Preprocessing as Deserialize>::deserialize(
                        specific.to_owned().into_deserializer(),
                    )?)
                } else {
                    None
                }
            };

            let processing = if_chain::if_chain! {
                if let Some(processing) = root.get("processing");
                if let Some(table) = processing.get(source);
                if let Some(specific) = table.get(var);
                then {
                    Some(<Processing as Deserialize>::deserialize(
                        specific.to_owned().into_deserializer(),
                    )?)
                } else {
                    None
                }
            };

            if let Some(specific) = preprocessing {
                update_preprocessing(
                    settings
                        .iter_mut()
                        .filter(|settings| settings.file.source == source)
                        .filter(|settings| settings.extract.name == var),
                    &specific,
                    &preprocessing_default,
                );
            }
            if let Some(specific) = processing {
                update_processing(
                    settings
                        .iter_mut()
                        .filter(|settings| settings.file.source == source)
                        .filter(|settings| settings.extract.name == var),
                    &specific,
                    &processing_default,
                );
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
    specific: &Preprocessing,
    default: &Preprocessing,
) {
    let Preprocessing {
        impulse_filter,
        impulse_tuning,
        masks,
        trim_left,
        trim_right,
        prefactor_x,
        prefactor_y,
        invert_x,
    } = specific;

    for settings in iter {
        if impulse_filter != &default.impulse_filter {
            settings.preprocessing.impulse_filter = impulse_filter.to_owned();
        }
        if impulse_tuning != &default.impulse_tuning {
            settings.preprocessing.impulse_tuning = impulse_tuning.to_owned();
        }
        if masks != &default.masks {
            settings.preprocessing.masks = masks.to_owned();
        }
        if trim_left != &default.trim_left {
            settings.preprocessing.trim_left = trim_left.to_owned();
        }
        if trim_right != &default.trim_right {
            settings.preprocessing.trim_right = trim_right.to_owned();
        }
        if prefactor_x != &default.prefactor_x {
            settings.preprocessing.prefactor_x = prefactor_x.to_owned();
        }
        if prefactor_y != &default.prefactor_y {
            settings.preprocessing.prefactor_y = prefactor_y.to_owned();
        }
        if invert_x != &default.invert_x {
            settings.preprocessing.invert_x = invert_x.to_owned();
        }
    }
}

fn update_processing<'a>(
    iter: impl Iterator<Item = &'a mut Settings>,
    specific: &Processing,
    default: &Processing,
) {
    let Processing {
        interpolation,
        interpolation_n,
        derivative,
    } = specific;

    for settings in iter {
        if interpolation != &default.interpolation {
            settings.processing.interpolation = interpolation.to_owned();
        }
        if interpolation_n != &default.interpolation_n {
            settings.processing.interpolation_n = interpolation_n.to_owned();
        }
        if derivative != &default.derivative {
            settings.processing.derivative = derivative.to_owned();
        }
    }
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
    pub gnuplot: bool,
    #[serde(default = "_true")]
    pub threading: bool,
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
}

fn _true() -> bool {
    true
}

fn one() -> f64 {
    1.0
}
