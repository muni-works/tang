//! First-divergence attribution: which kernel is wrong, given every tap's stats.
//!
//! The naive rule (the earliest failing stage, in pipeline order, whose inputs pass)
//! blamed `attn_q` for the GLM-5.3 KDA bug: k and v were written at the wrong offset INTO
//! the q block, so q was the first failing tap — but q's own kernel was fine. The fix
//! weighs evidence the stats already carry:
//!
//! 1. **provenance**: for a candidate with finite-but-wrong values, look for those values
//!    in every tap's CPU reference. Found in a sibling → that sibling's writer put them
//!    there ("`qkv.k` wrote into `qkv.q`"), and when both live in one buffer the geometry
//!    gives the writer's actual offset (×4 off = an element offset used as a byte address).
//! 2. **unwritten**: a candidate with GPU NaN beyond the reference's (poisoned scratch)
//!    has regions its writer never reached — direct evidence against that writer.
//!    A finite-wrong candidate whose SIBLING (same buffer) is unwritten is most likely that
//!    sibling's misdirected write, not its own kernel's bug.
//! 3. **zeroed**: notably more exact zeros than the reference.
//! 4. only then pipeline order.
use crate::stat::elem_bad;
use crate::{Layout, Spec, Stat, Tap, Taps};

/// One compared tap.
#[derive(Clone, Debug)]
pub struct Row {
    pub name: &'static str,
    pub stat: Stat,
    pub layout: Layout,
    pub bad: bool,
}

/// Why the suspect is suspected.
#[derive(Clone, Debug, PartialEq)]
pub enum Cause {
    /// Values that belong to `writer`'s tap were found in `victim`. `actual_base` /
    /// `expected_base` are the writer's offset in their shared buffer (elements), when known.
    Misplaced {
        victim: &'static str,
        at: String,
        from: String,
        actual_base: Option<i64>,
        expected_base: Option<usize>,
        hint: String,
    },
    /// GPU NaN/Inf beyond the reference's (poisoned: never written).
    Unwritten { count: usize, n: usize },
    /// More exact zeros than the reference.
    Zeroed { count: usize, n: usize },
    /// Wrong values, no structural evidence.
    Wrong,
    /// Failures, but every failing tap has a failing input (or none observed).
    NoCleanInput,
}

/// The attribution of one failing window.
#[derive(Clone, Debug)]
pub struct Verdict {
    /// The tap whose writer is blamed ("" for [`Cause::NoCleanInput`]).
    pub suspect: &'static str,
    /// That tap's `what` (the kernel to look at).
    pub what: &'static str,
    pub cause: Cause,
    /// Every candidate and what was found about it, one line each.
    pub evidence: Vec<String>,
}

impl Verdict {
    pub fn message(&self) -> String {
        let what = if self.what.is_empty() {
            String::new()
        } else {
            format!(" → suspect: {}", self.what)
        };
        match &self.cause {
            Cause::Misplaced {
                victim,
                at,
                from,
                actual_base,
                expected_base,
                hint,
            } => {
                let _ = (actual_base, expected_base);
                let geo = if hint.is_empty() {
                    String::new()
                } else {
                    format!(" ({hint})")
                };
                if at.is_empty() {
                    return format!("`{}` likely wrote into `{victim}`{geo}{what}", self.suspect);
                }
                format!(
                    "`{}` wrote into `{victim}`: {victim}@{at} holds {}@{from}{geo}{what}",
                    self.suspect, self.suspect
                )
            }
            Cause::Unwritten { count, n } => format!(
                "`{}` left {count}/{n} elements unwritten (poison NaN survived){what}",
                self.suspect
            ),
            Cause::Zeroed { count, n } => {
                format!("`{}` has {count}/{n} extra exact zeros{what}", self.suspect)
            }
            Cause::Wrong => format!(
                "stage `{}` fails with its inputs passing{what}",
                self.suspect
            ),
            Cause::NoCleanInput => "failures with no clean-input stage (see the BAD rows)".into(),
        }
    }
}

const RUN: usize = 8;
/// Search budget (elements scanned per target) so a huge tensor can't stall a debug run.
const SCAN_CAP: usize = 1 << 24;

struct Found {
    source: &'static str,
    i: usize,
    j: usize,
}

/// Where do `victim`'s wrong values come from? Scans the CPU reference of every tap
/// (siblings in the same buffer first) for a run of RUN consecutive bad GPU values.
fn provenance(
    victim: &Tap,
    gv: &[f32],
    cv: &[f32],
    spec: &Spec,
    cpu: &Taps,
    tol: f64,
) -> Option<Found> {
    let rms_v = rms(cv);
    // up to 16 candidate runs: the first ones may be constant or NaN-broken
    let mut starts = Vec::new();
    let mut i = 0;
    while i + RUN <= gv.len().min(cv.len()) && starts.len() < 16 {
        let ok =
            (0..RUN).all(|k| gv[i + k].is_finite() && elem_bad(gv[i + k], cv[i + k], tol, rms_v));
        if ok {
            let run = &gv[i..i + RUN];
            let (lo, hi) = run
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
            if hi - lo > 1e-6 * (hi.abs() + lo.abs()) && hi.abs() + lo.abs() > 0.0 {
                starts.push(i);
                i += RUN * 64; // spread the probes
                continue;
            }
        }
        i += 1;
    }
    let same_buf = |t: &Tap| match (&t.region, &victim.region) {
        (Some(a), Some(b)) => a.buffer == b.buffer,
        _ => false,
    };
    let mut order: Vec<&Tap> = spec
        .taps
        .iter()
        .filter(|t| same_buf(t) && t.name != victim.name)
        .collect();
    order.extend(
        spec.taps
            .iter()
            .filter(|t| !same_buf(t) && t.name != victim.name),
    );
    order.push(victim);
    let resolved: Vec<_> = order.iter().map(|t| (t, t.resolve(cpu))).collect();
    for &i in &starts {
        let run = &gv[i..i + RUN];
        for (t, ct) in &resolved {
            let Some(ct) = ct else { continue };
            let rms_t = rms(ct);
            let n = ct.len().saturating_sub(RUN - 1).min(SCAN_CAP);
            for j in 0..n {
                if t.name == victim.name && j == i {
                    continue;
                }
                if (0..RUN).all(|k| !elem_bad(run[k], ct[j + k], tol, rms_t)) {
                    return Some(Found {
                        source: t.name,
                        i,
                        j,
                    });
                }
            }
        }
    }
    None
}

fn rms(v: &[f32]) -> f64 {
    let (mut s, mut n) = (0f64, 0usize);
    for &x in v.iter().filter(|x| x.is_finite()) {
        s += (x as f64) * (x as f64);
        n += 1;
    }
    if n > 0 {
        (s / n as f64).sqrt()
    } else {
        0.0
    }
}

/// The writer's actual base vs its expected one, when victim and source share a buffer.
fn geometry(
    victim: &Tap,
    source: &Tap,
    i: usize,
    j: usize,
) -> (Option<i64>, Option<usize>, String) {
    let (Some(rv), Some(rs)) = (&victim.region, &source.region) else {
        if victim.name == source.name {
            return (
                None,
                None,
                format!("shifted by {} elements", i as i64 - j as i64),
            );
        }
        return (None, None, String::new());
    };
    if rv.buffer != rs.buffer || rv.stride != rs.stride {
        return (None, None, String::new());
    }
    let pv = (i / rv.len) * rv.stride + rv.base + i % rv.len;
    let ps = (j / rs.len) * rs.stride + rs.base + j % rs.len;
    let actual = rs.base as i64 + pv as i64 - ps as i64;
    let e = rs.base as i64;
    let why = if actual > 0 && actual * 4 == e {
        "expected = 4 × actual: an ELEMENT offset added to a BYTE address (size_of::<f32>())"
            .to_string()
    } else if actual > 0 && actual * 2 == e {
        "expected = 2 × actual: an element offset used as bytes for a 2-byte type".to_string()
    } else if e > 0 && actual == e * 4 {
        "actual = 4 × expected: a BYTE offset used as an element index".to_string()
    } else {
        format!("displaced by {} elements", actual - e)
    };
    let hint = format!(
        "writer's base offset in `{}` is {actual}, expected {e}: {why}",
        rs.buffer
    );
    (Some(actual), Some(rs.base), hint)
}

/// Attribute a failing window. `rows` are the compared taps (spec order).
pub fn attribute(
    spec: &Spec,
    rows: &[Row],
    gpu: &Taps,
    cpu: &Taps,
    tol: f64,
    poisoned: bool,
) -> Verdict {
    let row = |n: &str| rows.iter().find(|r| r.name == n);
    let bad = |n: &str| row(n).is_some_and(|r| r.bad);
    let cands: Vec<(&Tap, &Row)> = spec
        .taps
        .iter()
        .filter_map(|t| row(t.name).map(|r| (t, r)))
        .filter(|(t, r)| r.bad && t.inputs.iter().all(|i| !bad(i)))
        .collect();
    if cands.is_empty() {
        return Verdict {
            suspect: "",
            what: "",
            cause: Cause::NoCleanInput,
            evidence: Vec::new(),
        };
    }
    let unwritten_word = if poisoned {
        "unwritten (poison NaN)"
    } else {
        "non-finite"
    };
    let mut evidence = Vec::new();
    let mut misplaced: Option<Verdict> = None;
    for (t, r) in &cands {
        let s = &r.stat;
        let mut line = format!("candidate `{}`: rel {:.3e}", t.name, s.rel_l2);
        if s.extra_nonfinite() > 0 {
            line += &format!(", {} {unwritten_word}", s.extra_nonfinite());
        }
        if s.extra_zeros() > s.n / 100 {
            line += &format!(", {} extra zeros", s.extra_zeros());
        }
        let (Some(gv), Some(cv)) = (t.resolve(gpu), t.resolve(cpu)) else {
            evidence.push(line);
            continue;
        };
        if let Some(f) = provenance(t, &gv, &cv, spec, cpu, tol) {
            let src = spec.get(f.source).unwrap();
            let (actual, expected, hint) = geometry(t, src, f.i, f.j);
            let at = t.layout.locate(f.i);
            let from = src.layout.locate(f.j);
            line += &format!(", values at {at} = CPU `{}`@{from}", f.source);
            if !hint.is_empty() {
                line += &format!(" ({hint})");
            }
            if misplaced.is_none() {
                misplaced = Some(Verdict {
                    suspect: src.name,
                    what: src.what,
                    cause: Cause::Misplaced {
                        victim: t.name,
                        at,
                        from,
                        actual_base: actual,
                        expected_base: expected,
                        hint,
                    },
                    evidence: Vec::new(),
                });
            }
        }
        evidence.push(line);
    }
    let finish = |suspect, what, cause| Verdict {
        suspect,
        what,
        cause,
        evidence: evidence.clone(),
    };
    if let Some(mut v) = misplaced {
        v.evidence = evidence;
        return v;
    }
    // a finite-wrong candidate with an unwritten sibling: the sibling's writer misfired
    let sib = |a: &Tap, b: &Tap| match (&a.region, &b.region) {
        (Some(x), Some(y)) => x.buffer == y.buffer && a.name != b.name,
        _ => false,
    };
    for (t, r) in &cands {
        if r.stat.extra_nonfinite() > 0 {
            continue;
        }
        if let Some((s, sr)) = cands
            .iter()
            .find(|(s, sr)| sib(t, s) && sr.stat.extra_nonfinite() > 0)
        {
            return finish(s.name, s.what, Cause::Misplaced {
                    victim: t.name,
                    at: String::new(),
                    from: String::new(),
                    actual_base: None,
                    expected_base: None,
                    hint: format!(
                        "sibling `{}` has {} {unwritten_word} while `{}` is finite but wrong (values not traced)",
                        s.name,
                        sr.stat.extra_nonfinite(),
                        t.name
                    ),
                },
            );
        }
    }
    if let Some((t, r)) = cands.iter().find(|(_, r)| r.stat.extra_nonfinite() > 0) {
        return finish(
            t.name,
            t.what,
            Cause::Unwritten {
                count: r.stat.extra_nonfinite(),
                n: r.stat.n,
            },
        );
    }
    if let Some((t, r)) = cands
        .iter()
        .find(|(_, r)| r.stat.extra_zeros() > r.stat.n / 100)
    {
        return finish(
            t.name,
            t.what,
            Cause::Zeroed {
                count: r.stat.extra_zeros(),
                n: r.stat.n,
            },
        );
    }
    let (t, _) = cands[0];
    finish(t.name, t.what, Cause::Wrong)
}
