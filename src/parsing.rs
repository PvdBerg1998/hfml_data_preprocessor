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
        let mut s = f.debug_map();
        for (k, v) in &self.data {
            s.entry(
                k,
                &format!(
                    "{:?},... ({} values)",
                    v.iter().take(3).collect::<Vec<_>>(),
                    v.len()
                ),
            );
        }
        s.finish()?;
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

    /// Gets an x/y data pair.
    /// ### Panics
    /// - When columns do not exist
    /// - When columns do not have the same length
    /// - When columns are empty
    pub fn xy(&self, x: &str, y: &str) -> XY<'_> {
        let x = self.column(x);
        let y = self.column(y);
        assert_eq!(x.len(), y.len(), "XY columns have a different length");
        assert!(x.len() > 0, "XY columns are empty");

        XY { x, y }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct XY<'a> {
    x: &'a [f64],
    y: &'a [f64],
}

#[derive(Clone, Debug, PartialEq)]
pub struct MonotonicXY {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl<'a> XY<'a> {
    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_finite(&self) -> bool {
        self.x.iter().all(|x| x.is_finite()) && self.y.iter().all(|y| y.is_finite())
    }

    pub fn to_monotonic(&self) -> MonotonicXY {
        let XY { x, y } = *self;

        // Sort
        let perm = permutation::sort_by_key(x.as_ref(), |x| float_ord::FloatOrd(*x));
        let x = perm.apply_slice(x.as_ref());
        let y = perm.apply_slice(y.as_ref());

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

    pub fn y_mut(&mut self) -> &mut [f64] {
        &mut self.y
    }

    pub fn xy_mut(&mut self) -> (&[f64], &mut [f64]) {
        (&self.x, &mut self.y)
    }

    /// Returns a new MonotonicXY, truncated to x values given.
    /// ### Panics
    /// - When a boundary is higher than the largest x value in the data
    pub fn trim(&mut self, lower_x: f64, upper_x: f64) -> Self {
        let lower = self.x.iter().position(|&b| b >= lower_x).unwrap();
        let upper = self.x.iter().position(|&b| b >= upper_x).unwrap();

        let x = &self.x[lower..upper];
        let y = &self.y[lower..upper];

        MonotonicXY {
            x: x.to_vec(),
            y: y.to_vec(),
        }
    }
}
