use gsl_rust::sorting;
use std::collections::HashMap;
use std::convert::Infallible;
use std::fmt;
use std::str::FromStr;

#[derive(Clone, Debug, PartialEq)]
pub struct Data {
    data: HashMap<String, Box<[f64]>>,
}

impl FromStr for Data {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Values are tab separated
        let headers = s
            .lines()
            .map(|line| line.trim().split('\t'))
            .flatten()
            .take_while(|entry| entry.parse::<f64>().is_err())
            .collect::<Vec<_>>();

        // Split data into columns
        let mut data = HashMap::new();
        for (i, &header) in headers.iter().enumerate() {
            let column = s
                .lines()
                .flat_map(|line| line.trim().split('\t'))
                .filter_map(|entry| entry.parse::<f64>().ok())
                .skip(i)
                .step_by(headers.len())
                .collect::<Box<[f64]>>();
            data.insert(String::from(header), column);
        }

        Ok(Data { data })
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
    pub fn xy(&self, x: &str, y: &str) -> XY {
        let x = self.column(x);
        let y = self.column(y);
        assert_eq!(x.len(), y.len(), "XY columns have a different length");
        assert!(!x.is_empty(), "XY columns are empty");
        XY {
            x: x.to_vec(),
            y: y.to_vec(),
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
    x: Box<[f64]>,
    y: Box<[f64]>,
}

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

impl MonotonicXY {
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

    pub fn xy(&self) -> (&[f64], &[f64]) {
        (&self.x, &self.y)
    }

    /// Returns a slice, truncated to x values given.
    /// ### Panics
    /// - When a boundary is higher than the largest x value in the data
    pub fn trimmed_xy(&self, lower_x: f64, upper_x: f64) -> (&[f64], &[f64]) {
        let lower = self.x.iter().position(|&b| b >= lower_x).unwrap();
        let upper = self.x.iter().position(|&b| b >= upper_x).unwrap();

        let x = &self.x[lower..upper];
        let y = &self.y[lower..upper];

        (x, y)
    }
}
