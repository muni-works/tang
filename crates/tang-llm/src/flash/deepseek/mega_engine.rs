//! The engine side of `DS_MEGA=1` (a child of engine.rs, for [`GpuModel`]'s fields): the
//! next block's attention mHC stage is deferred at the end of a block ([`GpuModel::glue_defer`]
//! allocates its outputs, launches nothing) and then either run inside the dense-glue
//! megakernel with the attention's q/kv prologue ([`GpuModel::glue_qkv`]) or, when anything
//! else is about to read its outputs first, as the plain `ds_hc_fused_ks`
//! ([`GpuModel::glue_flush`]).

use super::super::mega::MgArgs;
use super::*;

/// A deferred `ds_hc_fused_ks` (as `GpuModel::hc_fused` with a post-mix), outputs allocated.
pub(super) struct GluePend {
    h: Buf,
    h2: u64,
    res: Buf,
    post: u64,
    comb: u64,
    p: String,
    pre: Buf,
    xo: Buf,
    o: [Buf; 3],
    hin: Buf,
    part: Buf,
}

impl GpuModel {
    /// [`GpuModel::hc_fused`] of the next block's attention stage, deferred: the stream,
    /// the coefficients and the block input, all still to be written.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn glue_defer(
        &mut self,
        h: Buf,
        h2: Option<Buf>,
        res: Buf,
        m: &Mix,
        t: usize,
        p: &str,
        pre: Buf,
    ) -> Result<(Buf, Mix, Buf)> {
        self.glue_flush()?;
        let c = &self.cpu.cfg;
        let (hc, d) = (c.hc_mult, c.dim);
        let rows = self.dw(&format!("{p}.hc_attn_fn"))?.v.rows as usize;
        assert!(
            rows <= 80 && 256 % hc == 0,
            "ds_hc_fused_ks: <= 80 mixes, hc | 256"
        );
        let gx = d.div_ceil(256 / hc);
        let xo = self.fbuf(t * hc * d)?;
        let part = self.fbuf(t * gx * (rows + 2))?;
        let o = [
            self.fbuf(t * hc)?,
            self.fbuf(t * hc)?,
            self.fbuf(t * hc * hc)?,
        ];
        let hin = self.fbuf(t * d)?;
        self.pend_glue = Some(GluePend {
            h,
            h2: h2.map_or(0, |b| b.ptr),
            res,
            post: m.post.ptr,
            comb: m.comb.ptr,
            p: p.to_string(),
            pre,
            xo,
            o,
            hin,
            part,
        });
        let mix = Mix {
            pre: o[0],
            post: o[1],
            comb: o[2],
        };
        Ok((xo, mix, hin))
    }

    /// Run a deferred stage as its own `ds_hc_fused_ks` (what `hc_fused` launches).
    pub(super) fn glue_flush(&mut self) -> Result<()> {
        let Some(g) = self.pend_glue.take() else {
            return Ok(());
        };
        let c = &self.cpu.cfg;
        let (hc, d) = (c.hc_mult, c.dim);
        let t = g.hin.len / (4 * d);
        let w = self.dw(&format!("{}.hc_attn_fn", g.p))?;
        let sc = self.dw(&format!("{}.hc_attn_scale", g.p))?;
        let base = self.dw(&format!("{}.hc_attn_base", g.p))?;
        let nw = self.dw(&format!("{}.attn_norm.weight", g.p))?;
        let gx = d.div_ceil(256 / hc);
        let (ti, hci, di) = (t as i32, hc as i32, d as i32);
        let (it, eps, neps) = (c.hc_sinkhorn_iters as i32, c.hc_eps, c.norm_eps);
        let (wv, nv, scp, bp) = (w.v, nw.v, sc.v.w, base.v.w);
        let cnt = self.hc_cnt.ptr;
        let s = self.span("hc");
        unsafe {
            self.k.go(
                "ds_hc_fused_ks",
                (gx as u32, t.div_ceil(8) as u32, 1),
                B256,
                0,
                &self.s,
                args![
                    g.h.ptr, g.h2, g.res.ptr, g.post, g.comb, g.xo.ptr, wv, g.part.ptr, g.pre.ptr,
                    g.hin.ptr, nv, scp, bp, g.o[0].ptr, g.o[1].ptr, g.o[2].ptr, cnt, ti, hci, di,
                    it, eps, neps, neps, 0i32
                ],
            )?
        };
        self.end(s);
        Ok(())
    }

    /// `qkv_fused`'s launches, with the deferred stage in front, as one `mg_glue` when a
    /// stage is pending for this block input `x` (else `None`, after flushing).
    #[allow(clippy::type_complexity)]
    pub(super) fn glue_qkv(
        &mut self,
        p: &str,
        lid: usize,
        x: Buf,
        t: usize,
        [wqa, wkv, wqb, qn, kn]: [DsW; 5],
    ) -> Result<Option<(Buf, Buf, Buf, usize, usize)>> {
        let ours = matches!(&self.pend_glue, Some(g) if g.hin.ptr == x.ptr && g.p == p);
        if !ours {
            self.glue_flush()?;
            return Ok(None);
        }
        let c = self.cpu.cfg.clone();
        let (h, hd, win, rd) = (c.n_heads, c.head_dim, c.window_size, c.rope_head_dim);
        let (hc, d) = (c.hc_mult, c.dim);
        let w = self.dw(&format!("{p}.hc_attn_fn"))?.v;
        let nw = self.dw(&format!("{p}.attn_norm.weight"))?.v;
        let sc = self.dw(&format!("{p}.hc_attn_scale"))?.v.w;
        let base = self.dw(&format!("{p}.hc_attn_base"))?.v.w;
        let ql = wqa.rows as usize;
        let ring = b(&self.win[lid]);
        let (qr, kv) = (self.fbuf(t * ql)?, self.fbuf(t * hd)?);
        let (kv0, split, mode, all, ncopy) = if t > 1 {
            let all = self.fbuf((win + t) * hd)?;
            (all, win + t, 3, all.ptr, win * hd)
        } else {
            (ring, win, 1, 0, 0)
        };
        let q = self.fbuf(t * h * hd)?;
        let (xc, xs) = self.k.ks_x_ptrs();
        let (qc, qsc) = self.k.wo_x_ptrs();
        let g = self.pend_glue.as_ref().unwrap();
        let a = MgArgs {
            h: g.h.ptr,
            h2: g.h2,
            res: g.res.ptr,
            post: g.post,
            comb: g.comb,
            xo: g.xo.ptr,
            hpart: g.part.ptr,
            pre_i: g.pre.ptr,
            hin: g.hin.ptr,
            scale: sc,
            base,
            pre_o: g.o[0].ptr,
            post_o: g.o[1].ptr,
            comb_o: g.o[2].ptr,
            xc,
            xs,
            ring: ring.ptr,
            all,
            qr: qr.ptr,
            kv: kv.ptr,
            qc,
            qsc,
            tab: self.tabs[lid].ptr,
            p: self.p.ptr,
            q: q.ptr,
            w,
            nw,
            wqa,
            wkv,
            wqb,
            qn,
            kn,
            t: t as i32,
            hc: hc as i32,
            d: d as i32,
            iters: c.hc_sinkhorn_iters as i32,
            gx: d.div_ceil(256 / hc) as i32,
            rd: rd as i32,
            win: win as i32,
            ncopy: ncopy as i32,
            eps: c.hc_eps,
            neps: c.norm_eps,
            reps: c.norm_eps,
            qeps: c.norm_eps,
            ..Default::default()
        };
        let mega = self.mega.as_ref().unwrap();
        if !mega.ok(&a) {
            self.glue_flush()?;
            return Ok(None);
        }
        self.pend_glue = None;
        let s = self.span("glue");
        self.mega.as_ref().unwrap().launch(&self.s, a)?;
        self.end(s);
        Ok(Some((qr, q, kv0, split, mode)))
    }
}
