//! What a kernel (or a pipeline of kernels) exposes for checking: named taps, their
//! layouts, which taps each one reads, and where it sits in a shared device buffer.

use crate::Layout;

/// Named host copies of tensors from one side of a run (the device's read-backs, or the
/// CPU reference's intermediates). Insertion order is kept (dumps follow it).
#[derive(Clone, Debug, Default)]
pub struct Taps {
    items: Vec<(String, Vec<f32>)>,
}

impl Taps {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record `v` as `name` (replaces an earlier entry of the same name).
    pub fn put(&mut self, name: impl Into<String>, v: Vec<f32>) {
        let name = name.into();
        match self.items.iter_mut().find(|(n, _)| *n == name) {
            Some(slot) => slot.1 = v,
            None => self.items.push((name, v)),
        }
    }

    /// Copy `v` into `name`'s slot at element `at`, growing the slot (with NaN, so gaps
    /// stay visible) as needed: for taps assembled chunk by chunk.
    pub fn put_at(&mut self, name: &str, at: usize, v: &[f32]) {
        if self.get(name).is_none() {
            self.put(name, Vec::new());
        }
        let slot = &mut self.items.iter_mut().find(|(n, _)| n == name).unwrap().1;
        if slot.len() < at + v.len() {
            slot.resize(at + v.len(), f32::NAN);
        }
        slot[at..at + v.len()].copy_from_slice(v);
    }

    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.items
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_slice())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &[f32])> {
        self.items.iter().map(|(n, v)| (n.as_str(), v.as_slice()))
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// Record into an optional tap sink, computing the value only when someone listens:
/// `tap(&mut taps, "qkv", || qkv.clone())`. Zero cost when `taps` is `None`.
pub fn tap(taps: &mut Option<&mut Taps>, name: &str, v: impl FnOnce() -> Vec<f32>) {
    if let Some(t) = taps.as_deref_mut() {
        t.put(name, v());
    }
}

/// Where a tap lives inside a shared device buffer, in ELEMENTS: row `r` of the tap is
/// `buffer[r·stride + base .. r·stride + base + len]`. Taps that share a buffer are
/// siblings: one kernel's misplaced write lands in another's region, and the attribution
/// uses this geometry to name the writer and its actual offset.
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    pub buffer: &'static str,
    pub base: usize,
    pub len: usize,
    pub stride: usize,
}

/// One observation point.
#[derive(Clone, Debug)]
pub struct Tap {
    pub name: &'static str,
    pub layout: Layout,
    /// Taps this one is computed from. A failing tap whose observed inputs all pass is a
    /// first-divergence candidate.
    pub inputs: Vec<&'static str>,
    /// What writes it (kernel names): the attribution's "suspect".
    pub what: &'static str,
    /// When set, the tap's data is this section of buffer `region.buffer` (in [`Taps`]).
    pub region: Option<Region>,
}

impl Tap {
    pub fn new(name: &'static str, layout: Layout) -> Self {
        Self {
            name,
            layout,
            inputs: Vec::new(),
            what: "",
            region: None,
        }
    }

    /// The taps it reads.
    pub fn after(mut self, inputs: &[&'static str]) -> Self {
        self.inputs = inputs.to_vec();
        self
    }

    /// The kernel(s) that write it.
    pub fn what(mut self, what: &'static str) -> Self {
        self.what = what;
        self
    }

    /// A section of shared buffer `buffer`: row r at `r·stride + base`, `len` elements.
    pub fn within(mut self, buffer: &'static str, base: usize, len: usize, stride: usize) -> Self {
        self.region = Some(Region {
            buffer,
            base,
            len,
            stride,
        });
        self
    }

    /// The tap's data from `taps`: its own entry, or its section of its buffer.
    /// `None` when absent or empty (the tap is then skipped).
    pub fn resolve<'a>(&self, taps: &'a Taps) -> Option<std::borrow::Cow<'a, [f32]>> {
        if let Some(v) = taps.get(self.name).filter(|v| !v.is_empty()) {
            return Some(std::borrow::Cow::Borrowed(v));
        }
        let r = self.region.as_ref()?;
        let buf = taps.get(r.buffer).filter(|v| !v.is_empty())?;
        let rows = buf.len() / r.stride.max(1);
        let mut out = Vec::with_capacity(rows * r.len);
        for row in 0..rows {
            let b = row * r.stride + r.base;
            out.extend_from_slice(buf.get(b..b + r.len)?);
        }
        Some(std::borrow::Cow::Owned(out))
    }
}

/// A kernel's (or pipeline's) taps, in pipeline order.
#[derive(Clone, Debug, Default)]
pub struct Spec {
    pub kernel: String,
    pub taps: Vec<Tap>,
}

impl Spec {
    pub fn new(kernel: impl Into<String>) -> Self {
        Self {
            kernel: kernel.into(),
            taps: Vec::new(),
        }
    }

    pub fn tap(mut self, t: Tap) -> Self {
        self.taps.push(t);
        self
    }

    pub fn push(&mut self, t: Tap) {
        self.taps.push(t);
    }

    /// Each of `sections` as a sibling region of one stacked buffer `[rows][Σ len]`:
    /// `stacked("qkv", &[("qkv.q", di), ("qkv.k", di), ("qkv.v", di)], layout, what)`.
    pub fn stacked(
        mut self,
        buffer: &'static str,
        sections: &[(&'static str, usize, &'static str)],
        layout: Layout,
        inputs: &[&'static str],
    ) -> Self {
        let stride: usize = sections.iter().map(|s| s.1).sum();
        let mut base = 0;
        for &(name, len, what) in sections {
            self.taps.push(
                Tap::new(name, layout.clone())
                    .after(inputs)
                    .what(what)
                    .within(buffer, base, len, stride),
            );
            base += len;
        }
        self
    }

    pub fn get(&self, name: &str) -> Option<&Tap> {
        self.taps.iter().find(|t| t.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sections_resolve() {
        // rows=2, [q0 q1 k0 k1 v0 v1] per row
        let mut t = Taps::new();
        t.put("qkv", (0..12).map(|i| i as f32).collect());
        let s = Spec::new("x").stacked(
            "qkv",
            &[("q", 2, ""), ("k", 2, ""), ("v", 2, "")],
            Layout::flat(),
            &[],
        );
        assert_eq!(
            &*s.get("k").unwrap().resolve(&t).unwrap(),
            &[2.0, 3.0, 8.0, 9.0]
        );
        assert!(s.get("k").unwrap().resolve(&Taps::new()).is_none());
    }

    #[test]
    fn put_at_grows_with_nan() {
        let mut t = Taps::new();
        t.put_at("c", 2, &[1.0]);
        let v = t.get("c").unwrap();
        assert!(v[0].is_nan() && v[1].is_nan() && v[2] == 1.0);
    }
}
