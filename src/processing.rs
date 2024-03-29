/*
    processing.rs
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

#![allow(dead_code)]

use crate::data::Data;
use crate::data::MonotonicXY;
use crate::data::XY;
use crate::output::save;
use crate::output::SaveRecord;
use crate::settings::Settings;
use crate::settings::*;
use anyhow::bail;
use anyhow::ensure;
use anyhow::Context;
use anyhow::Result;
use gsl_rust::interpolation::interpolate_monotonic;
use gsl_rust::interpolation::Derivative;
use gsl_rust::stats;
use log::*;
use num_complex::Complex;
use num_traits::AsPrimitive;
use num_traits::Float;
use serde_json as json;
use std::sync::Arc;

impl Settings {
    pub fn prepare(self: Arc<Self>, data: &Data, project: Project) -> Result<Option<Prepared>> {
        let settings = self;
        let Extract { name, x, y } = &settings.extract;
        let src = settings.file.source.as_str();

        debug!("Preparing file '{src}': dataset '{name}'");

        // Check if the given names actually correspond to existing data columns
        if !data.contains(x) {
            warn!("specified x column '{x}' for dataset '{name}' does not exist in file '{src}'");
            return Ok(None);
        }
        if !data.contains(y) {
            warn!("specified y column '{y}' for dataset '{name}' does not exist in file '{src}'");
            return Ok(None);
        }

        // Extract the columns
        debug!("Extracting dataset '{name}' (x='{x}', y='{y}')");
        let xy = match data.clone_xy(x, y) {
            Some(xy) => xy,
            None => {
                bail!("Dataset '{name}' (x='{x}', y='{y}') has less than 2 values");
            }
        };

        // Check for NaN and infinities
        if !xy.is_finite() {
            bail!("File '{src}':'{name}' contains non-finite values");
        }

        // Prepare labels
        let x_label = x.to_owned();
        let y_label = format!("{} ({})", name, y);

        Ok(Some(Prepared {
            project,
            settings,
            x_label,
            y_label,
            xy,
        }))
    }
}

impl Prepared {
    // pub fn xy(&self) -> &XY {
    //     &self.xy
    // }

    pub fn preprocess(self) -> Result<(Preprocessed, Vec<SaveRecord>)> {
        let Self {
            project,
            settings,
            x_label,
            y_label,
            xy,
        } = self;

        let Extract { name, x, y: _ } = &settings.extract;
        let src = settings.file.source.as_str();

        debug!("Preprocessing file '{src}': dataset '{name}'");
        let mut saves = vec![];

        let n_log2 = (xy.len() as f64).log2();
        trace!("Dataset '{src}':'{name}' length: ~2^{n_log2:.2}");

        // Output unprocessed data
        trace!("Storing unprocessed data for '{src}':'{name}'");
        saves.push(
            save(
                &project,
                name,
                "unprocessed",
                &settings.file.dest,
                &x_label,
                &y_label,
                false, // Can't plot non monotonic data
                xy.x(),
                xy.y(),
                json::json!({
                    "tags": settings.file.metadata,
                }),
            )
            .with_context(|| format!("Failed to store unprocessed data for '{src}':'{name}'"))?,
        );

        // Monotonicity
        trace!("Making '{src}':'{name}' monotonic");
        let mut xy = match xy.into_monotonic() {
            Some(xy) => xy,
            None => {
                bail!("Dataset is too small after deduplication. Is your x variable changing?");
            }
        };

        // Output raw data
        trace!("Storing raw data for '{src}':'{name}'");
        saves.push(
            save(
                &project,
                name,
                "raw",
                &settings.file.dest,
                &x_label,
                &y_label,
                project.plot,
                xy.x(),
                xy.y(),
                json::json!({
                    "tags": settings.file.metadata,
                }),
            )
            .with_context(|| format!("Failed to store raw data for '{src}':'{name}'"))?,
        );

        // Impulse filtering
        // Do this before trimming, such that edge artifacts may be cut off afterwards
        if settings.preprocessing.impulse_filter > 0 {
            let width = settings.preprocessing.impulse_filter;
            let tuning = settings.preprocessing.impulse_tuning;
            trace!(
                "Applying impulse filter of width {width} and tuning {tuning} to '{src}':'{name}'"
            );
            xy.impulse_filter(width as usize, tuning);
        }

        // Masking
        for &Mask { left, right } in &settings.preprocessing.masks {
            ensure!(
                left >= xy.left_x(),
                "Mask {left} is outside domain {:?} for '{src}':'{name}'",
                xy.left_x()
            );
            ensure!(
                right <= xy.right_x(),
                "Mask {right} is outside domain {:?} for '{src}':'{name}'",
                xy.right_x()
            );

            trace!("Masking '{src}':'{name}' from {} to {}", left, right);
            xy.mask(left, right);
        }

        // Trimming
        let mut trim_left = settings
            .preprocessing
            .trim_left
            .unwrap_or_else(|| xy.left_x());
        let mut trim_right = settings
            .preprocessing
            .trim_right
            .unwrap_or_else(|| xy.right_x());
        ensure!(
            trim_left < trim_right,
            "Trimmed domain is empty or negative"
        );

        // Automagically swap trim sign if needed to deal with negative x data
        if xy.right_x() <= 0.0 && trim_right >= 0.0 {
            trace!("Flipping trim domain around zero for '{src}':'{name}'");
            std::mem::swap(&mut trim_left, &mut trim_right);
            trim_left *= -1.0;
            trim_right *= -1.0;
        }

        if settings.preprocessing.invert_x {
            ensure!(
                trim_left != 0.0,
                "Can't trim to zero when inverting x for '{src}':'{name}'"
            );
            ensure!(
                trim_right != 0.0,
                "Can't trim to zero when inverting x for '{src}':'{name}'"
            );
        }

        ensure!(
            xy.left_x() <= trim_left,
            "Left trim {trim_left} is outside domain {:?} for '{src}':'{name}'",
            xy.left_x()
        );
        ensure!(
            xy.right_x() >= trim_right,
            "Right trim {trim_right} is outside domain {:?} for '{src}':'{name}'",
            xy.right_x()
        );
        trace!("Desired data domain: [{trim_left},{trim_right}]");
        xy.trim(trim_left, trim_right);
        trace!("Actual data domain: [{},{}]", xy.left_x(), xy.right_x());

        // Premultiplication
        // Use local override prefactor if set
        let mx = settings.preprocessing.prefactor_x;
        let my = settings.preprocessing.prefactor_y;
        if mx != 1.0 {
            trace!("Multiplying '{src}':'{name}' x by {mx}");
            xy.multiply_x(mx);
        }
        if my != 1.0 {
            trace!("Multiplying '{src}':'{name}' y by {my}");
            xy.multiply_y(my);
        }

        // Output preprocessed data
        trace!("Storing preprocessed data for '{src}':'{name}'");
        saves.push(
            save(
                &project,
                name,
                "preprocessed",
                &settings.file.dest,
                &x_label,
                &y_label,
                project.plot,
                xy.x(),
                xy.y(),
                json::json!({
                    "tags": settings.file.metadata,
                    "settings": settings.preprocessing,
                }),
            )
            .with_context(|| format!("Failed to store preprocessed data for '{src}':'{name}'"))?,
        );

        // 1/x
        if settings.preprocessing.invert_x {
            trace!("Inverting '{src}':'{name}' x");
            xy.invert_x();
            trace!("Inverted data domain: [{},{}]", xy.left_x(), xy.right_x());

            // Output inverted data
            trace!("Storing inverted data for '{src}':'{name}'");
            saves.push(
                save(
                    &project,
                    name,
                    "inverted",
                    &settings.file.dest,
                    &x_label,
                    &y_label,
                    false,
                    xy.x(),
                    xy.y(),
                    json::json!({
                        "tags": settings.file.metadata,
                    }),
                )
                .with_context(|| format!("Failed to store inverted data for '{src}':'{name}'"))?,
            );
        }

        let x_label = if settings.preprocessing.invert_x {
            format!("1/{x}")
        } else {
            x_label
        };

        Ok((
            Preprocessed {
                project,
                settings,
                x_label,
                y_label,
                xy,
            },
            saves,
        ))
    }
}

impl Preprocessed {
    pub fn xy(&self) -> &MonotonicXY {
        &self.xy
    }

    pub fn process(
        self,
        n_interp_global: Option<u64>,
        n_interp_var: Option<u64>,
    ) -> Result<(Processed, Vec<SaveRecord>)> {
        let Self {
            project,
            settings,
            x_label,
            y_label,
            xy,
        } = self;
        let name = &settings.extract.name;
        let src = settings.file.source.as_str();

        debug!("Processing file '{src}': dataset '{name}'");
        let mut saves = vec![];

        // Interpolation
        let (x, y) = match settings.processing.interpolation {
            Some(algorithm) => {
                let n_interp = match settings.processing.interpolation_n {
                    InterpolationLength::MinimumPerVariable => {
                        // Invariant: the interpolation length statistics routine will have returned Some
                        n_interp_var.expect("should have calculated n_interp_var")
                    }
                    InterpolationLength::Minimum => {
                        // Invariant: the interpolation length statistics routine will have returned Some
                        n_interp_global.expect("should have calculated n_interp_global")
                    }
                    InterpolationLength::Amount(n) => n,
                };

                let deriv = settings.processing.derivative;
                let deriv_str = match deriv {
                    Derivative::None => "function",
                    Derivative::First => "1st derivative",
                    Derivative::Second => "2nd derivative",
                };

                debug!("Interpolating '{src}':'{name}' using {deriv_str} at {n_interp} points using {algorithm} interpolation");

                let dx = xy.domain_len() / n_interp as f64;
                let x_eval = (0..n_interp)
                    .map(|i| (i as f64) * dx + xy.left_x())
                    .collect::<Vec<_>>();
                let y_eval =
                    interpolate_monotonic(algorithm.into(), deriv, xy.x(), xy.y(), &x_eval)
                        .with_context(|| format!("Failed to make '{src}':'{name}' monotonic"))?;

                // Output processed data
                trace!("Storing processed data for '{src}':'{name}'");
                saves.push(
                    save(
                        &project,
                        name,
                        "processed",
                        &settings.file.dest,
                        &x_label,
                        &y_label,
                        false,
                        &x_eval,
                        &y_eval,
                        json::json!({
                            "tags": settings.file.metadata,
                            "settings": settings.processing,
                            "interpolation_n": n_interp
                        }),
                    )
                    .with_context(|| {
                        format!("Failed to store processed data for '{src}':'{name}'")
                    })?,
                );

                (x_eval, y_eval)
            }
            None => xy.take_xy(),
        };

        let xy = match MonotonicXY::new(x, y) {
            Some(xy) => xy,
            None => {
                bail!("Dataset is too small after interpolation");
            }
        };

        Ok((
            Processed {
                project,
                settings,
                // x_label,
                // y_label,
                xy,
            },
            saves,
        ))
    }
}

impl Processed {
    pub fn xy(&self) -> &MonotonicXY {
        &self.xy
    }

    pub fn prepare_fft(
        self,
        fft_settings: Fft,
        domain_left: f64,
        domain_right: f64,
        sweep_index: Option<usize>,
    ) -> Result<PreparedFft> {
        let Self {
            project,
            settings,
            // x_label,
            // y_label,
            mut xy,
        } = self;
        let name = &settings.extract.name;
        let src = settings.file.source.as_str();

        let window_label = sweep_index
            .map(|i| format!("window #{i}"))
            .unwrap_or_else(|| "(full window)".to_owned());
        debug!("Preparing file '{src}': dataset '{name}' for FFT {window_label}");

        // Trim domain
        // The domain is calculated internally so we may assume its boundaries are correct
        trace!(
            "Trimming '{src}':'{name}' for FFT {window_label} to [{domain_left}, {domain_right}]"
        );
        xy.trim(domain_left, domain_right);

        // Amount of nonzero points
        let n_data = xy.len();
        // Extract requested total FFT length
        let desired_n = fft_settings.zero_pad;
        // Amount of padded zero points
        let n_pad = desired_n.saturating_sub(n_data);

        if fft_settings.center {
            let mean = stats::mean(xy.y());

            trace!("Removing mean value {mean} from '{src}':'{name}'");
            xy.y_mut().iter_mut().for_each(|y| {
                *y -= mean;
            });
        }

        if fft_settings.hann {
            trace!("Applying Hann window to '{src}':'{name}'");
            xy.y_mut().iter_mut().enumerate().for_each(|(i, y)| {
                *y *= (i as f64 * std::f64::consts::PI / (n_data - 1) as f64)
                    .sin()
                    .powi(2);
            });

            // Avoid numerical error and leaking at the edges
            // Invariant: x/y can not be empty
            *xy.y_mut().first_mut().unwrap() = 0.0;
            *xy.y_mut().last_mut().unwrap() = 0.0;
        }

        // Zero padding is delayed until manual application on PreparedFft
        // Transferring the zeroes to the GPU is redundant, but required for CPU FFT

        // Prepare frequency space cutoffs
        let lower_cutoff = fft_settings.truncate_lower.unwrap_or(0.0);
        let upper_cutoff = fft_settings.truncate_upper.unwrap_or(f64::INFINITY);
        if lower_cutoff != 0.0 || upper_cutoff.is_finite() {
            trace!("Truncating FFT to {lower_cutoff}..{upper_cutoff} 'Hz' for '{src}':'{name}'");
        }

        // Invariant: x has a length of >= 2
        // We have to assume the data is uniformly sampled or interpolated here
        let dt = xy.x()[1] - xy.x()[0];
        let nyquist = 0.5 / dt;
        let df = nyquist / (desired_n as f64);
        // This is not necessarily equal for all files, even after interpolation,
        // because the trimmed domains are not exactly the same length
        // This can't really be solved in a consistent way, perhaps by interpolating the full data domain,
        // and then resampling on the trimmed domain. The issue is that it requires information outside the trimmed domain,
        // meaning we can either ignore it, extrapolate, use "fake" frequencies on the domain, or resample.
        // In practise the difference should be less than 2 dt, with df vanishing when using some zero padding.
        trace!("FFT dt = {dt}, df = {df}, Nyquist = {nyquist} for '{src}':'{name}'");

        // Calculate clamped frequency space cutoff indices
        // Note: this includes zero padding
        let max_fft_len = desired_n / 2 + 1;
        let start_idx = (lower_cutoff * dt * desired_n as f64).floor() as usize;
        ensure!(
            start_idx < max_fft_len,
            "FFT lower cutoff {lower_cutoff} is higher than Nyquist frequency {nyquist:.2} for '{src}':'{name}'"
        );
        let end_idx = {
            let end_idx = (upper_cutoff * dt * desired_n as f64).ceil() as usize;
            if end_idx > max_fft_len {
                warn!("FFT upper cutoff {upper_cutoff} is higher than Nyquist frequency {nyquist:.2} for '{src}':'{name}'");
                // End index is exclusive
                max_fft_len
            } else {
                end_idx
            }
        };
        ensure!(
            start_idx != end_idx,
            "FFT trimmed domain is empty for '{src}':'{name}'"
        );

        // Sampled frequencies : k/(N dt)
        let freq_normalisation = 1.0 / (dt * desired_n as f64);
        let frequencies = (start_idx..end_idx)
            .map(|i| i as f64 * freq_normalisation)
            .collect::<Vec<_>>();

        // Take y
        let (_, y) = xy.take_xy();

        Ok(PreparedFft {
            inner: PreparedFftNoData {
                project,
                settings,
                fft_settings,
                sweep_index,
                // x_label,
                // y_label,
                n_data,
                n_pad,
                domain_left,
                domain_right,
                start_idx,
                end_idx,
                frequencies,
            },
            y,
        })
    }
}

impl PreparedFft {
    pub fn zero_pad(&mut self) -> Result<()> {
        let Self {
            inner:
                PreparedFftNoData {
                    settings,
                    n_pad,
                    fft_settings,
                    ..
                },
            y,
        } = self;
        let name = &settings.extract.name;
        let src = settings.file.source.as_str();

        let desired_n = fft_settings.zero_pad;
        let desired_n_log2 = (desired_n as f64).log2().floor() as usize;

        if *n_pad > 0 {
            assert!(y.len() != desired_n, "Tried to zero pad twice");
            trace!("Zero padding '{src}':'{name}' by {n_pad} to reach 2^{desired_n_log2}");
            y.resize(desired_n, 0.0);
        } else {
            ensure!(
                y.len().is_power_of_two(),
                "Data length is larger than 2^{desired_n_log2}, so no padding can take place, but the length is not a power of two."
            );
        }

        Ok(())
    }

    pub fn minimum_fft_len(&self) -> usize {
        // We just ignore the values starting from 0 and request all FFT values up to our end index
        self.inner.end_idx
    }

    pub fn split_data(self) -> (PreparedFftNoData, Vec<f64>) {
        let Self { inner, y } = self;
        (inner, y)
    }
}

impl PreparedFftNoData {
    pub fn finish<F: Float + AsPrimitive<f64>>(
        self,
        // May be cut off above end_idx (GPU optimization)
        mut fft: Vec<Complex<F>>,
    ) -> Result<Vec<SaveRecord>> {
        let Self {
            project,
            settings,
            fft_settings,
            sweep_index,
            // x_label: _,
            // y_label: _,
            n_data,
            n_pad: _,
            domain_left,
            domain_right,
            start_idx,
            end_idx,
            frequencies,
        } = self;
        let name = &settings.extract.name;
        let src = settings.file.source.as_str();

        debug!("FFT postprocessing file '{src}': dataset '{name}'");
        let mut saves = vec![];

        // Truncate FFT
        assert!(
            start_idx < fft.len(),
            "FFT start index {} > FFT length {}",
            start_idx,
            fft.len()
        );
        assert!(
            end_idx <= fft.len(),
            "FFT end index {} > FFT length {}",
            end_idx,
            fft.len()
        );
        let fft = fft.drain(start_idx..end_idx);

        // Take absolute value and normalize
        // NB. Do this after truncation to save a huge amount of work
        trace!("Normalizing FFT by 1/{n_data}");
        let normalisation = 1.0 / n_data as f64;
        let fft = fft
            .map(|y| y.norm().as_() * normalisation)
            .collect::<Vec<_>>();

        // Output FFT
        debug!("Storing FFT for '{src}':'{name}'");
        saves.push(
            save(
                &project,
                name,
                &match fft_settings.sweep {
                    FftSweep::Full => "fft".to_owned(),
                    FftSweep::Lower { .. } => format!("fft_sweep_lower_{}", sweep_index.unwrap()),
                    FftSweep::Upper { .. } => format!("fft_sweep_upper_{}", sweep_index.unwrap()),
                    FftSweep::Windows { .. } => {
                        format!("fft_sweep_window_{}", sweep_index.unwrap())
                    }
                },
                &settings.file.dest,
                "Frequency",
                "FFT Amplitude",
                false,
                &frequencies,
                &fft,
                json::json!({
                    "tags": settings.file.metadata,
                    "fft_domain": (domain_left, domain_right),
                    "settings": fft_settings,
                }),
            )
            .with_context(|| format!("Failed to store FFT for '{src}':'{name}'"))?,
        );

        Ok(saves)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Prepared {
    pub project: Project,
    pub settings: Arc<Settings>,
    x_label: String,
    y_label: String,
    xy: XY,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Preprocessed {
    pub project: Project,
    pub settings: Arc<Settings>,
    x_label: String,
    y_label: String,
    xy: MonotonicXY,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Processed {
    pub project: Project,
    pub settings: Arc<Settings>,
    // x_label: String,
    // y_label: String,
    xy: MonotonicXY,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedFft {
    inner: PreparedFftNoData,
    y: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedFftNoData {
    pub project: Project,
    pub settings: Arc<Settings>,
    fft_settings: Fft,
    sweep_index: Option<usize>,
    // x_label: String,
    // y_label: String,
    n_data: usize,
    n_pad: usize,
    domain_left: f64,
    domain_right: f64,
    start_idx: usize,
    end_idx: usize,
    frequencies: Vec<f64>,
}
