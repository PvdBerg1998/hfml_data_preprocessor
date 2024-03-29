/*
    data.rs
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
use anyhow::bail;
use anyhow::ensure;
use anyhow::Error;
use anyhow::Result;
use gsl_rust::filter;
use gsl_rust::sorting;
use itertools::Itertools;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

// Up to this amount can be stored per line without another heap allocation
const HEADER_GUESS: usize = 16;

#[derive(Clone, Debug, PartialEq)]
pub struct Data {
    data: HashMap<String, Vec<f64>>,
}

impl FromStr for Data {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Values are tab separated
        let mut headers = vec![];
        let mut header_lines = 0;

        for line in s.lines() {
            header_lines += 1;

            // Check if line contains only non-numerical values
            let split = line
                .trim()
                .split('\t')
                .map(|header| header.trim())
                .filter(|header| !header.is_empty())
                .collect::<Vec<_>>();
            if split
                .iter()
                .all(|&entry| fast_float::parse::<f64, _>(entry).is_err())
            {
                headers.extend_from_slice(&split);
            } else {
                break;
            }
        }
        let headers = headers;
        let header_lines = header_lines;

        ensure!(
            !has_dup(&headers),
            "header contains duplicate values: {:?}",
            headers.iter().duplicates().collect::<Vec<_>>()
        );

        let data = s
            .lines()
            .skip(header_lines)
            .map(|line| line.trim().split_ascii_whitespace().collect())
            .collect::<Vec<SmallVec<[&str; HEADER_GUESS]>>>();

        for (i, line) in data.iter().enumerate() {
            match line.len().cmp(&headers.len()) {
                Ordering::Less => bail!("missing column at line {i}"),
                Ordering::Greater => {
                    if i == 0 {
                        bail!("missing header(s)");
                    } else {
                        bail!("missing header(s) or extra column(s) at line {i}");
                    }
                }
                Ordering::Equal => {}
            }
        }

        // Split data into columns
        let mut out = HashMap::new();
        for (i, &header) in headers.iter().enumerate() {
            let column = data
                .iter()
                .map(|line| unsafe {
                    // Safety: we checked for missing columns
                    line.get_unchecked(i)
                })
                .map(|entry| fast_float::parse::<f64, _>(entry).map_err(|e| e.into()))
                .collect::<Result<Vec<f64>>>()?;
            out.insert(String::from(header), column);
        }

        Ok(Data { data: out })
    }
}

impl fmt::Display for Data {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Data: {{")?;
        for (k, v) in self
            .columns()
            .iter()
            .map(|&column| (column, &self.data[column]))
        {
            writeln!(
                f,
                "   {}: {:?},... ({} values)",
                k,
                v.iter().take(3).collect::<Vec<_>>(),
                v.len()
            )?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

#[allow(dead_code)]
impl Data {
    pub fn iter_columns(&self) -> impl Iterator<Item = &'_ str> {
        self.data.keys().map(|x| x.as_str())
    }

    pub fn columns(&self) -> Box<[&'_ str]> {
        let mut c = self.iter_columns().collect::<Vec<_>>();
        c.sort_unstable_by(|a, b| natord::compare(a, b));
        c.into_boxed_slice()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.data.contains_key(name)
    }

    /// ### Panics
    /// - When no column with given name exists.
    pub fn column(&self, name: &str) -> &[f64] {
        &self.data[name]
    }

    /// Tries to change the name of a column, if it exists.
    /// ### Panics
    /// - When target column already exists (i.e. overwriting is not allowed)
    pub fn rename_column(&mut self, from: &str, to: &str) {
        if let Some(from_val) = self.data.remove(from) {
            assert!(
                !self.data.contains_key(to),
                "Tried renaming column to existing column"
            );
            self.data.insert(String::from(to), from_val);
        }
    }

    /// Clones an x/y data pair.
    /// Returns None if the dataset is smaller than 2 values.
    /// ### Panics
    /// - When columns do not exist
    /// - When columns do not have the same length
    /// - When columns have less than 2 elements
    pub fn clone_xy(&self, x: &str, y: &str) -> Option<XY> {
        let x = self.data.get(x).expect("x column does not exist");
        let y = self.data.get(y).expect("y column does not exist");
        assert_eq!(x.len(), y.len(), "XY columns have a different length");

        if x.len() < 2 {
            None
        } else {
            Some(XY::new(x.clone(), y.clone()))
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct XY {
    x: Vec<f64>,
    y: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MonotonicXY {
    x: Vec<f64>,
    y: Vec<f64>,
}

#[allow(dead_code)]
impl XY {
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        assert!(x.len() >= 2);
        assert!(y.len() >= 2);
        Self { x, y }
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_finite(&self) -> bool {
        self.x.iter().all(|x| x.is_finite()) && self.y.iter().all(|y| y.is_finite())
    }

    pub fn into_monotonic(self) -> Option<MonotonicXY> {
        let XY { mut x, mut y } = self;

        // Heap sort may not be the best for almost sorted data,
        // but it's plenty fast and has good worst case scaling
        sorting::sort_xy(&mut x, &mut y);

        // Safe to unwrap as this only returns Err when the lengths are not equal
        let (x, y) = sorting::dedup_x_mean(&x, &y).unwrap();

        // Note: if all x values are equal, the length of x and y will now be 1,
        // so MonotonicXY::new can return None.
        MonotonicXY::new(x, y)
    }

    pub fn x(&self) -> &[f64] {
        &self.x
    }

    pub fn y(&self) -> &[f64] {
        &self.y
    }

    pub fn y_mut(&mut self) -> &mut Vec<f64> {
        &mut self.y
    }
}

#[allow(dead_code)]
impl MonotonicXY {
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Option<Self> {
        if x.len() < 2 || y.len() < 2 {
            return None;
        }
        Some(MonotonicXY { x, y })
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn left_x(&self) -> f64 {
        // Invariant: x/y can not be empty
        *self.x.first().unwrap()
    }

    pub fn right_x(&self) -> f64 {
        // Invariant: x/y can not be empty
        *self.x.last().unwrap()
    }

    pub fn x(&self) -> &[f64] {
        &self.x
    }

    pub fn y(&self) -> &[f64] {
        &self.y
    }

    pub fn y_mut(&mut self) -> &mut Vec<f64> {
        &mut self.y
    }

    pub fn domain_len(&self) -> f64 {
        self.right_x() - self.left_x()
    }

    pub fn xy(&self) -> (&[f64], &[f64]) {
        (&self.x, &self.y)
    }

    pub fn take_xy(self) -> (Vec<f64>, Vec<f64>) {
        (self.x, self.y)
    }

    pub fn multiply_x(&mut self, v: f64) {
        self.x.iter_mut().for_each(|x| {
            *x *= v;
        });
        if v < 0.0 {
            self.x.reverse();
            self.y.reverse();
        }
    }

    pub fn multiply_y(&mut self, v: f64) {
        self.y.iter_mut().for_each(|y| {
            *y *= v;
        });
    }

    pub fn invert_x(&mut self) {
        self.x.iter_mut().for_each(|x| {
            *x = 1.0 / *x;
        });
        self.x.reverse();
        self.y.reverse();
    }

    /// Applies a median filter of given width.
    /// ### Panics
    /// - When the width is zero
    pub fn median_filter(&mut self, width: usize) {
        filter::median(width as usize, &mut self.y).unwrap();
    }

    /// Applies an impulse filter of given width and tuning parameter.
    /// ### Panics
    /// - When the width is zero
    /// - The tuning parameter is negative
    pub fn impulse_filter(&mut self, width: usize, tuning: f64) {
        filter::impulse(
            width,
            tuning,
            filter::ImpulseFilterScale::SnStatistic,
            &mut self.y,
        )
        .unwrap();
    }

    fn robust_x_to_idx(&self, lower_x: f64, upper_x: f64) -> (usize, usize) {
        let lower = if lower_x <= self.left_x() {
            0
        } else if lower_x >= self.right_x() {
            self.x.len() - 1
        } else {
            self.x
                .iter()
                .position(|&x| x >= lower_x)
                .unwrap_or(self.x.len() - 1)
        };

        let upper = if upper_x <= self.left_x() {
            0
        } else if upper_x >= self.right_x() {
            self.x.len() - 1
        } else {
            self.x
                .iter()
                .position(|&x| x >= upper_x)
                .unwrap_or(self.x.len() - 1)
        };

        (lower, upper)
    }

    /// Removes a range of data
    /// ### Panics
    /// - When the resulting dataset is empty
    pub fn mask(&mut self, lower_x: f64, upper_x: f64) {
        let (lower, upper) = self.robust_x_to_idx(lower_x, upper_x);

        let mut i = 0usize;
        self.x.retain(|_| {
            let remove = (lower..=upper).contains(&i);
            i += 1;
            !remove
        });

        let mut i = 0usize;
        self.y.retain(|_| {
            let remove = (lower..=upper).contains(&i);
            i += 1;
            !remove
        });

        assert!(!self.x.is_empty(), "Mask resulted in an empty domain");
    }

    /// Truncates the stored values to be inside or equal to the given boundaries
    /// ### Panics
    /// - When the resulting dataset is empty
    pub fn trim(&mut self, lower_x: f64, upper_x: f64) {
        let (lower, upper) = self.robust_x_to_idx(lower_x, upper_x);

        self.x.drain(0..lower);
        self.y.drain(0..lower);
        self.x.truncate(upper - lower);
        self.y.truncate(upper - lower);

        assert!(!self.x.is_empty(), "Trim resulted in an empty domain");
    }
}
