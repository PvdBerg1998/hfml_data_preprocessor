/*
    data.rs
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

use anyhow::ensure;
use anyhow::Error;
use gsl_rust::sorting;
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

#[derive(Clone, Debug, PartialEq)]
pub struct Data {
    data: HashMap<String, Vec<f64>>,
}

impl FromStr for Data {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Values are tab separated
        let headers = s
            .lines()
            .map(|line| line.trim().split('\t'))
            .flatten()
            .take_while(|&entry| fast_float::parse::<f64, _>(entry).is_err())
            .collect::<Vec<_>>();

        ensure!(!has_dup(&headers), "header contains duplicate values");

        // Split data into columns
        let mut data = HashMap::new();
        for (i, &header) in headers.iter().enumerate() {
            let column = s
                .lines()
                .flat_map(|line| line.trim().split('\t'))
                .filter_map(|entry| fast_float::parse::<f64, _>(entry).ok())
                .skip(i)
                .step_by(headers.len())
                .collect::<Vec<f64>>();
            data.insert(String::from(header), column);
        }

        Ok(Data { data })
    }
}

fn has_dup<T: PartialEq>(slice: &[T]) -> bool {
    for i in 1..slice.len() {
        if slice[i..].contains(&slice[i - 1]) {
            return true;
        }
    }
    false
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
        c.sort_unstable();
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
    pub fn rename_column(&mut self, old: &str, new: &str) {
        if let Some(old_val) = self.data.remove(old) {
            assert!(
                !self.data.contains_key(new),
                "Tried renaming column to existing column"
            );
            self.data.insert(String::from(new), old_val);
        }
    }

    /// Clones an x/y data pair.
    /// ### Panics
    /// - When columns do not exist
    /// - When columns do not have the same length
    /// - When columns are empty
    pub fn clone_xy(&mut self, x: &str, y: &str) -> XY {
        let x = self.data.get(x).unwrap();
        let y = self.data.get(y).unwrap();
        assert_eq!(x.len(), y.len(), "XY columns have a different length");
        assert!(!x.is_empty(), "XY columns are empty");
        XY {
            x: x.clone(),
            y: y.clone(),
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
    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_finite(&self) -> bool {
        self.x.iter().all(|x| x.is_finite()) && self.y.iter().all(|y| y.is_finite())
    }

    pub fn multiply_x(&mut self, v: f64) {
        self.x.iter_mut().for_each(|x| {
            *x *= v;
        });
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
    }

    pub fn into_monotonic(self) -> MonotonicXY {
        let XY { mut x, mut y } = self;

        sorting::sort_xy(&mut x, &mut y);

        // Safe to unwrap as this only returns Err when the lengths are not equal
        let (x, y) = sorting::dedup_x_mean(&x, &y).unwrap();

        MonotonicXY { x, y }
    }

    pub fn x(&self) -> &[f64] {
        &self.x
    }

    pub fn y(&self) -> &[f64] {
        &self.y
    }
}

#[allow(dead_code)]
impl MonotonicXY {
    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn min_x(&self) -> f64 {
        *self.x.first().unwrap()
    }

    pub fn max_x(&self) -> f64 {
        *self.x.last().unwrap()
    }

    pub fn x(&self) -> &[f64] {
        &self.x
    }

    pub fn y(&self) -> &[f64] {
        &self.y
    }

    pub fn domain_len(&self) -> f64 {
        self.max_x() - self.min_x()
    }

    pub fn xy(&self) -> (&[f64], &[f64]) {
        (&self.x, &self.y)
    }

    pub fn take_xy(self) -> (Vec<f64>, Vec<f64>) {
        (self.x, self.y)
    }

    /// Truncates the stored values to be inside or equal to the given boundaries
    /// ### Panics
    /// - When a boundary is higher than the largest x value in the data
    pub fn trim(&mut self, lower_x: f64, upper_x: f64) {
        if lower_x == self.min_x() && upper_x == self.max_x() {
            return;
        }

        let lower = self.x.iter().position(|&x| x >= lower_x).unwrap();
        let upper = self.x.iter().position(|&x| x >= upper_x).unwrap();
        self.x.drain(0..lower);
        self.y.drain(0..lower);
        self.x.truncate(upper - lower);
        self.y.truncate(upper - lower);
    }
}
