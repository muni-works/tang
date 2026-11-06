//! Control-loop graphs shared by the device exporters (`opencl_export`,
//! `cpu_export`): ipse-world's GRU world-model step and a locomotion-style
//! MLP policy.
use tang::Scalar;
use tang_compile::tensor::{TensorGraph, TensorId};
use tang_expr::ExprId;

/// One step of a GRU cell (PyTorch gate convention, as ipse-world's
/// `GruCell`): inputs `x [batch, obs]`, `h [batch, hidden]`, then per gate
/// r, z, n: `W_i [obs, hidden]`, `W_h [hidden, hidden]`, `b_i`, `b_h
/// [hidden]`; output `h' [batch, hidden]`.
pub fn gru_step(batch: usize, obs: usize, hidden: usize) -> (TensorGraph, Vec<TensorId>) {
    let mut g = TensorGraph::new();
    let x = g.input(&[batch, obs]);
    let h = g.input(&[batch, hidden]);
    let gate = |g: &mut TensorGraph| {
        let w_i = g.input(&[obs, hidden]);
        let w_h = g.input(&[hidden, hidden]);
        let b_i = g.input(&[1, hidden]);
        let b_h = g.input(&[1, hidden]);
        let gi = g.matmul(x, w_i).unwrap();
        let gh = g.matmul(h, w_h).unwrap();
        let b_i = g.broadcast(b_i, &[batch, hidden]).unwrap();
        let b_h = g.broadcast(b_h, &[batch, hidden]).unwrap();
        let gi = g.map(&[gi, b_i], |v| v[0] + v[1]).unwrap();
        let gh = g.map(&[gh, b_h], |v| v[0] + v[1]).unwrap();
        (gi, gh)
    };
    let (ir, hr) = gate(&mut g);
    let (iz, hz) = gate(&mut g);
    let (inn, hn) = gate(&mut g);
    let one = || ExprId::from_f64(1.0);
    let sigmoid = |e: ExprId| one() / (one() + (-e).exp());
    let h_next = g
        .map(&[ir, hr, iz, hz, inn, hn, h], |v| {
            let r = sigmoid(v[0] + v[1]);
            let z = sigmoid(v[2] + v[3]);
            let n = (v[4] + r * v[5]).tanh();
            (one() - z) * n + z * v[6]
        })
        .unwrap();
    let (g, outs) = g.substitute_maps(&[h_next]);
    let (mut g, outs) = g.eliminate_common(&outs);
    g.schedule_auto(&outs);
    (g, outs)
}

/// The same GRU step with the input and hidden contractions concatenated
/// along one reduce axis of `obs + hidden` (guarded views read zero past
/// each operand's end), so all four gate sums `r, z, i_n, h_n` share a
/// reduce space; `merge_siblings` makes them one tuple reduction whose
/// epilogue is the gate math: one kernel per step. Same inputs as
/// [`gru_step`].
pub fn gru_fused(batch: usize, obs: usize, hidden: usize) -> (TensorGraph, Vec<TensorId>) {
    use tang_compile::tensor::{Arg, Form, Monoid, View};
    let mut g = TensorGraph::new();
    let x = g.input(&[batch, obs]);
    let h = g.input(&[batch, hidden]);
    let mut w = Vec::new();
    for _ in 0..3 {
        let w_i = g.input(&[obs, hidden]);
        let w_h = g.input(&[hidden, hidden]);
        let b_i = g.input(&[1, hidden]);
        let b_h = g.input(&[1, hidden]);
        w.push((w_i, w_h, b_i, b_h));
    }
    let k = obs + hidden;
    let shifted = |d: usize| Form {
        offset: -(obs as i64),
        ..Form::unit(3, d)
    };
    let xv = || View {
        gather: None,
        forms: vec![Form::unit(3, 0), Form::unit(3, 2)],
        guard: true,
    };
    let hv = || View {
        gather: None,
        forms: vec![Form::unit(3, 0), shifted(2)],
        guard: true,
    };
    let wiv = || View {
        gather: None,
        forms: vec![Form::unit(3, 2), Form::unit(3, 1)],
        guard: true,
    };
    let whv = || View {
        gather: None,
        forms: vec![shifted(2), Form::unit(3, 1)],
        guard: true,
    };
    let sum = |g: &mut TensorGraph, args: Vec<Arg>| {
        let two = args.len() == 4;
        g.comp(&[batch, hidden], &[k], args, Some(Monoid::sum()), |v| {
            if two {
                vec![v[0] * v[1] + v[2] * v[3]]
            } else {
                vec![v[0] * v[1]]
            }
        })
        .unwrap()
    };
    let a = |src, view| Arg { src, view };
    let r = sum(
        &mut g,
        vec![a(x, xv()), a(w[0].0, wiv()), a(h, hv()), a(w[0].1, whv())],
    );
    let z = sum(
        &mut g,
        vec![a(x, xv()), a(w[1].0, wiv()), a(h, hv()), a(w[1].1, whv())],
    );
    let i_n = sum(&mut g, vec![a(x, xv()), a(w[2].0, wiv())]);
    let h_n = sum(&mut g, vec![a(h, hv()), a(w[2].1, whv())]);
    let bias: Vec<TensorId> = w
        .iter()
        .flat_map(|&(_, _, b_i, b_h)| [b_i, b_h])
        .map(|b| g.broadcast(b, &[batch, hidden]).unwrap())
        .collect();
    let one = || ExprId::from_f64(1.0);
    let sigmoid = |e: ExprId| one() / (one() + (-e).exp());
    let mut ins = vec![r, z, i_n, h_n, h];
    ins.extend(&bias);
    let h_next = g
        .map(&ins, |v| {
            let r = sigmoid(v[0] + v[5] + v[6]);
            let z = sigmoid(v[1] + v[7] + v[8]);
            let n = (v[2] + v[9] + r * (v[3] + v[10])).tanh();
            (one() - z) * n + z * v[4]
        })
        .unwrap();
    let (g, outs) = g.substitute_maps(&[h_next]);
    let (g, outs) = g.eliminate_common(&outs);
    let (mut g, outs) = g.merge_siblings(&outs);
    g.schedule_auto(&outs);
    (g, outs)
}

/// A locomotion-style MLP policy: `obs [batch, obs] → hidden → hidden →
/// act`, ELU activations, inputs `x`, then per layer `W [in, out]`, `b [1,
/// out]`; output `a [batch, act]`.
pub fn mlp_policy(
    batch: usize,
    obs: usize,
    hidden: usize,
    layers: usize,
    act: usize,
) -> (TensorGraph, Vec<TensorId>) {
    let mut g = TensorGraph::new();
    let mut y = g.input(&[batch, obs]);
    let mut width = obs;
    for l in 0..=layers {
        let out = if l == layers { act } else { hidden };
        let w = g.input(&[width, out]);
        let b = g.input(&[1, out]);
        let z = g.matmul(y, w).unwrap();
        let b = g.broadcast(b, &[batch, out]).unwrap();
        y = if l == layers {
            g.map(&[z, b], |v| v[0] + v[1]).unwrap()
        } else {
            g.map(&[z, b], |v| {
                let s = v[0] + v[1];
                let neg = s.exp() - ExprId::from_f64(1.0);
                tang_expr::with_graph(|eg| eg.select(s, s, neg))
            })
            .unwrap()
        };
        width = out;
    }
    let (g, outs) = g.substitute_maps(&[y]);
    let (mut g, outs) = g.eliminate_common(&outs);
    g.schedule_auto(&outs);
    (g, outs)
}
