//! Naming element `i` of a flat tensor: a row-major list of named dims.

/// One dim of a [`Layout`].
#[derive(Clone, Debug, PartialEq)]
pub struct Dim {
    pub name: &'static str,
    /// 0 = unbounded (only meaningful for the leading dim, e.g. tokens).
    pub size: usize,
    /// When set, index `k` prints as `labels[k]` instead of `{name}{k}` (e.g. q|k|v).
    pub labels: Option<&'static [&'static str]>,
}

/// A row-major layout descriptor, outermost dim first:
/// `Layout::new().dim("tok", 0).dim("h", nh).dim("d", hd)` names element 13 of a
/// `[t][2][4]` tensor `tok1 h1 d1`. An empty layout names elements `i{n}`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Layout {
    pub dims: Vec<Dim>,
}

impl Layout {
    pub fn new() -> Self {
        Self::default()
    }

    /// Flat: `i{n}`.
    pub fn flat() -> Self {
        Self::default()
    }

    /// Append a dim (inner to the ones before it).
    pub fn dim(mut self, name: &'static str, size: usize) -> Self {
        self.dims.push(Dim {
            name,
            size,
            labels: None,
        });
        self
    }

    /// Append a labelled dim of `labels.len()` entries, printed by label.
    pub fn labeled(mut self, name: &'static str, labels: &'static [&'static str]) -> Self {
        self.dims.push(Dim {
            name,
            size: labels.len(),
            labels: Some(labels),
        });
        self
    }

    /// Elements per step of the leading dim (product of the inner sizes).
    pub fn row(&self) -> usize {
        self.dims.iter().skip(1).map(|d| d.size.max(1)).product()
    }

    /// `tok3 h1 d7`-style name of element `i`.
    pub fn locate(&self, i: usize) -> String {
        if self.dims.is_empty() {
            return format!("i{i}");
        }
        let mut idx = vec![0usize; self.dims.len()];
        let mut rest = i;
        for (k, d) in self.dims.iter().enumerate().rev() {
            if k == 0 || d.size == 0 {
                idx[k] = rest;
                rest = 0;
            } else {
                idx[k] = rest % d.size;
                rest /= d.size;
            }
        }
        let parts: Vec<String> = self
            .dims
            .iter()
            .zip(&idx)
            .map(|(d, &k)| match d.labels.and_then(|l| l.get(k)) {
                Some(l) => (*l).to_string(),
                None => format!("{}{k}", d.name),
            })
            .collect();
        parts.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        let inner = Layout::new().dim("tok", 0).dim("h", 2).dim("d", 4);
        assert_eq!(inner.locate(13), "tok1 h1 d1");
        assert_eq!(
            inner.locate(8 * 7 + 3),
            "tok7 h0 d3",
            "leading dim unbounded"
        );
        let state = Layout::new().dim("h", 0).dim("key", 4).dim("val", 4);
        assert_eq!(state.locate(16 + 4 * 2 + 3), "h1 key2 val3");
        let conv = Layout::new()
            .labeled("sec", &["q", "k", "v"])
            .dim("h", 1)
            .dim("d", 2)
            .dim("tap", 3);
        assert_eq!(conv.locate(3 * 2 + 1), "k h0 d0 tap1");
        assert_eq!(Layout::flat().locate(5), "i5");
        assert_eq!(inner.row(), 8);
    }
}
