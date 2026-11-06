//! Memory planning for a lowered program: buffers assigned to shared
//! storage slots by liveness over the kernel order, plus in-place updates.
//!
//! A [`WgslProgram`] (and the CUDA/MSL programs built the same way) has one
//! buffer per graph input and per materialized comp. Over a whole training
//! step most intermediates are dead long before the step ends, so
//! [`TensorGraph::plan_memory`] walks the kernels in order and hands each
//! output a *slot* — a storage buffer that some earlier, now-dead
//! intermediate used — growing or adding slots only when nothing fits.
//!
//! In place: `donate` lists `(output, input)` pairs where the output may
//! overwrite the input's buffer (an optimizer's `p ← p − lr·u` writing into
//! `p`). It is taken only when legal: the kernel writes one value per
//! point, reads the donor point-for-point at its own coordinates, and every
//! other reader of the donor has already run. Kernels with a donation and
//! no readers of their own are moved to just after the donor's last other
//! reader, which is what makes the update legal in a forward-backward-update
//! program where the weight is still read by backward kernels emitted later.
//!
//! The plan is pure bookkeeping: runtimes allocate `slots` and bind buffer
//! `i` of the program to `slots[slot[i]]`. A backend that cannot bind one
//! buffer as both read-only and read-write in one dispatch (wgpu) must pass
//! no donations.

use super::wgsl::WgslProgram;
use super::{Format, TensorGraph, TensorId};

/// Bytes per element word: every slot is a run of 32-bit words.
fn words(len: usize, format: Format) -> usize {
    match format {
        Format::F32 => len.max(1),
        Format::F16 | Format::BF16 => len.div_ceil(2).max(1),
        f => f.buffer_bytes(len).div_ceil(4).max(1),
    }
}

/// Byte counts of a plan. Persistent: graph inputs (weights, optimizer
/// state, data). Intermediate: everything else.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryStats {
    pub persistent_bytes: usize,
    /// One buffer per materialized node.
    pub naive_bytes: usize,
    /// Sum of the planned intermediate slots.
    pub planned_bytes: usize,
    /// Most intermediate bytes live at once (a lower bound for any plan
    /// over this kernel order).
    pub live_peak_bytes: usize,
    /// Intermediate bytes saved by writing outputs into donated inputs.
    pub inplace_bytes: usize,
}

pub struct MemoryPlan {
    /// Kernel indices in execution order (a valid topological order).
    pub order: Vec<usize>,
    /// `slot[i]` holds `program.buffers[i]`.
    pub slot: Vec<usize>,
    /// Length of each slot in 32-bit words.
    pub slot_words: Vec<usize>,
    /// `Some(k)` when the slot is graph input `k`'s own buffer.
    pub slot_input: Vec<Option<usize>>,
    /// Donations taken: `(output, input whose buffer it overwrites)`.
    pub inplace: Vec<(TensorId, TensorId)>,
    pub stats: MemoryStats,
}

impl MemoryPlan {
    /// The trivial plan: one slot per buffer, program order.
    pub fn naive(prog: &WgslProgram) -> Self {
        let slot_words: Vec<usize> = prog
            .buffers
            .iter()
            .map(|b| words(b.len, b.format))
            .collect();
        let persistent: usize = prog
            .buffers
            .iter()
            .filter(|b| b.input.is_some())
            .map(|b| words(b.len, b.format) * 4)
            .sum();
        let naive = slot_words.iter().sum::<usize>() * 4 - persistent;
        Self {
            order: (0..prog.kernels.len()).collect(),
            slot: (0..prog.buffers.len()).collect(),
            slot_input: prog.buffers.iter().map(|b| b.input).collect(),
            slot_words,
            inplace: vec![],
            stats: MemoryStats {
                persistent_bytes: persistent,
                naive_bytes: naive,
                planned_bytes: naive,
                live_peak_bytes: naive,
                inplace_bytes: 0,
            },
        }
    }

    /// Total bytes the plan allocates (persistent + intermediate slots).
    pub fn total_bytes(&self) -> usize {
        self.slot_words.iter().sum::<usize>() * 4
    }
}

impl TensorGraph {
    /// Whether kernel `root` may write its output over `donor`'s buffer:
    /// one value per point from a point-schedule map that reads `donor`
    /// only directly, through the identity, at its own coordinates.
    fn inplace_legal(&self, prog: &WgslProgram, root: usize, donor: TensorId) -> bool {
        let node = &self.nodes[root];
        let Some(c) = self.comp_at(root) else {
            return false;
        };
        if node.split != 0 || node.tile != (0, 0, 0) || !c.red.is_empty() || c.width() != 1 {
            return false;
        }
        let spec = |t: TensorId| prog.buffers.iter().find(|b| b.node == t);
        let (Some(o), Some(d)) = (spec(TensorId(root as u32)), spec(donor)) else {
            return false;
        };
        if o.len != d.len || o.format != d.format {
            return false;
        }
        let points = c.points().to_vec();
        if self.nodes[donor.n()].shape != points {
            return false;
        }
        let rank = points.len();
        let identity = super::View::identity(rank, rank);
        // reads of the donor anywhere in the kernel: only root's own
        // args, only the identity
        for j in self.inline_closure(root) {
            // materialized deps are read from their buffers, not recomputed
            if j != root && !self.nodes[j].inline {
                continue;
            }
            let Some(cj) = self.comp_at(j) else {
                continue;
            };
            for a in &cj.args {
                if self.storage(a.src) != donor {
                    continue;
                }
                if j != root || a.src != donor || a.view.guard || a.view != identity {
                    return false;
                }
            }
        }
        true
    }

    /// Plan `prog` (lowered from `self`) — see the module docs. `outputs`
    /// stay live to the end; `donate` offers in-place updates.
    pub fn plan_memory(
        &self,
        prog: &WgslProgram,
        outputs: &[TensorId],
        donate: &[(TensorId, TensorId)],
    ) -> MemoryPlan {
        let nb = prog.buffers.len();
        let nk = prog.kernels.len();
        let buf_of = |t: TensorId| {
            let s = prog.storage(t);
            prog.buffers.iter().position(|b| b.node == s)
        };
        let kernel_reads: Vec<Vec<usize>> = prog
            .kernels
            .iter()
            .map(|k| k.reads.iter().filter_map(|&t| buf_of(t)).collect())
            .collect();
        let kernel_out: Vec<usize> = prog
            .kernels
            .iter()
            .map(|k| buf_of(k.node).expect("kernel output buffer"))
            .collect();

        // donations that could be legal at all, by kernel
        let mut donation: Vec<Option<usize>> = vec![None; nk];
        for &(out, donor) in donate {
            let (Some(ob), Some(db)) = (buf_of(out), buf_of(donor)) else {
                continue;
            };
            let Some(k) = kernel_out.iter().position(|&b| b == ob) else {
                continue;
            };
            if prog.buffers[db].input.is_none()
                || !kernel_reads[k].contains(&db)
                || !self.inplace_legal(prog, prog.kernels[k].node.n(), prog.buffers[db].node)
            {
                continue;
            }
            donation[k] = Some(db);
        }

        // order: sink each donating kernel past the donor's other readers,
        // as long as that keeps it before the readers of its own output
        let mut order: Vec<usize> = (0..nk).collect();
        for k in 0..nk {
            let Some(db) = donation[k] else { continue };
            let ob = kernel_out[k];
            let pos = order.iter().position(|&x| x == k).unwrap();
            let last_other = order
                .iter()
                .enumerate()
                .filter(|&(_, &x)| x != k && kernel_reads[x].contains(&db))
                .map(|(p, _)| p)
                .max();
            let first_reader = order
                .iter()
                .position(|&x| kernel_reads[x].contains(&ob))
                .unwrap_or(usize::MAX);
            if let Some(last) = last_other {
                if last > pos && last < first_reader {
                    order.remove(pos);
                    order.insert(last, k);
                }
            }
        }
        // a donor may only be overwritten once, by its last reader
        let pos_of: Vec<usize> = {
            let mut p = vec![0; nk];
            for (i, &k) in order.iter().enumerate() {
                p[k] = i;
            }
            p
        };
        let mut taken = vec![false; nb];
        for k in 0..nk {
            let Some(db) = donation[k] else { continue };
            let last = (0..nk)
                .filter(|&x| kernel_reads[x].contains(&db))
                .map(|x| pos_of[x])
                .max()
                .unwrap();
            if last != pos_of[k] || taken[db] {
                donation[k] = None;
            } else {
                taken[db] = true;
            }
        }

        // liveness over the order
        let output_bufs: Vec<usize> = outputs.iter().filter_map(|&t| buf_of(t)).collect();
        let mut last_use = vec![None::<usize>; nb];
        for (p, &k) in order.iter().enumerate() {
            for &b in kernel_reads[k].iter().chain([&kernel_out[k]]) {
                last_use[b] = Some(last_use[b].map_or(p, |l: usize| l.max(p)));
            }
        }
        for &b in &output_bufs {
            last_use[b] = Some(usize::MAX);
        }

        let w: Vec<usize> = prog
            .buffers
            .iter()
            .map(|b| words(b.len, b.format))
            .collect();
        let mut slot = vec![usize::MAX; nb];
        let mut slot_words = Vec::new();
        let mut slot_input = Vec::new();
        for (i, b) in prog.buffers.iter().enumerate() {
            if b.input.is_some() {
                slot[i] = slot_words.len();
                slot_words.push(w[i]);
                slot_input.push(b.input);
            }
        }
        let mut free: Vec<usize> = Vec::new();
        let mut live_words = 0usize;
        let mut peak = 0usize;
        let mut inplace = Vec::new();
        let mut inplace_words = 0usize;
        // buffers dying at each position
        let mut dying: Vec<Vec<usize>> = vec![Vec::new(); nk];
        for b in 0..nb {
            if prog.buffers[b].input.is_some() {
                continue;
            }
            if let Some(l) = last_use[b] {
                if l != usize::MAX {
                    dying[l].push(b);
                }
            }
        }
        for (p, &k) in order.iter().enumerate() {
            let ob = kernel_out[k];
            if slot[ob] == usize::MAX {
                if let Some(db) = donation[k] {
                    slot[ob] = slot[db];
                    inplace.push((prog.buffers[ob].node, prog.buffers[db].node));
                    inplace_words += w[ob];
                } else {
                    let need = w[ob];
                    let fit = free
                        .iter()
                        .enumerate()
                        .filter(|&(_, &s)| slot_words[s] >= need)
                        .min_by_key(|&(_, &s)| slot_words[s])
                        .map(|(i, _)| i);
                    let pick = fit.or_else(|| {
                        free.iter()
                            .enumerate()
                            .max_by_key(|&(_, &s)| slot_words[s])
                            .map(|(i, _)| i)
                    });
                    let s = match pick {
                        Some(i) => free.swap_remove(i),
                        None => {
                            slot_words.push(0);
                            slot_input.push(None);
                            slot_words.len() - 1
                        }
                    };
                    slot_words[s] = slot_words[s].max(need);
                    slot[ob] = s;
                    live_words += need;
                }
            }
            peak = peak.max(live_words);
            for &b in &dying[p] {
                if donation[k].is_some() && b == ob {
                    continue;
                }
                if slot[b] != usize::MAX && !inplace.iter().any(|&(o, _)| o == prog.buffers[b].node)
                {
                    free.push(slot[b]);
                    live_words -= w[b];
                }
            }
        }
        // buffers no kernel touches (unused inputs are already slotted)
        for b in 0..nb {
            if slot[b] == usize::MAX {
                slot[b] = slot_words.len();
                slot_words.push(w[b]);
                slot_input.push(None);
            }
        }

        let persistent: usize = prog
            .buffers
            .iter()
            .enumerate()
            .filter(|(_, b)| b.input.is_some())
            .map(|(i, _)| w[i] * 4)
            .sum();
        let naive = w.iter().sum::<usize>() * 4 - persistent;
        let planned = slot_words
            .iter()
            .zip(&slot_input)
            .filter(|(_, i)| i.is_none())
            .map(|(w, _)| w * 4)
            .sum();
        MemoryPlan {
            order,
            slot,
            slot_words,
            slot_input,
            inplace,
            stats: MemoryStats {
                persistent_bytes: persistent,
                naive_bytes: naive,
                planned_bytes: planned,
                live_peak_bytes: peak * 4,
                inplace_bytes: inplace_words * 4,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    /// A chain `x → a → b → c → out` needs two slots, not four, and an
    /// update `p' = p - g` writes into `p`.
    #[test]
    fn chain_reuses_slots_and_updates_in_place() {
        let mut g = TensorGraph::new();
        let x = g.input(&[64]);
        let p = g.input(&[64]);
        let mut h = x;
        for _ in 0..4 {
            let r = g.reduce(h, 0, Monoid::sum()).unwrap();
            let rb = g.broadcast(r, &[64]).unwrap();
            h = g.map(&[h, rb], |v| v[0] * v[1]).unwrap();
        }
        let p2 = g.map(&[p, h], |v| v[0] - v[1]).unwrap();
        let prog = g.to_wgsl(&[p2]);
        let naive = super::MemoryPlan::naive(&prog);
        let plan = g.plan_memory(&prog, &[p2], &[(p2, p)]);
        assert_eq!(plan.inplace, vec![(p2, p)]);
        assert!(plan.stats.planned_bytes < naive.stats.planned_bytes);
        // slots shared by buffers that are live at once must not collide
        for (pos, &k) in plan.order.iter().enumerate() {
            let kern = &prog.kernels[k];
            let bi = |t: TensorId| {
                prog.buffers
                    .iter()
                    .position(|b| b.node == prog.storage(t))
                    .unwrap()
            };
            let out = plan.slot[bi(kern.node)];
            for &r in &kern.reads {
                if r == p {
                    continue;
                }
                assert_ne!(plan.slot[bi(r)], out, "kernel at {pos} reads its own slot");
            }
        }
    }
}
