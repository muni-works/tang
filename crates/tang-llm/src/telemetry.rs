//! Optional OpenTelemetry tracing and metrics (see `docs/observability.md`).
//!
//! With the `otel` feature, each chat request becomes
//! a server span (continuing an incoming W3C `traceparent`) with `queued`, `prefill`,
//! `decode` and `finish` children, and every batched step feeds a few metrics. Spans are
//! built when the request concludes, from timestamps it carried, so nothing runs per token.
//! Export goes through `crates/tang-otel`: OTLP/JSON lines to a file by default, the
//! network only with `TANG_OTEL_LIVE=1`. Without the feature every type here is
//! zero-sized and every call is empty; with it and `OTEL_FILE=off`, each call is one
//! check of an unset `OnceLock`.

use crate::engine::{StepStats, Usage};
use axum::http::HeaderMap;

/// How a request ended, for its `finish` span and the usage log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum End {
    Ok,
    Cancelled,
    Error,
}

impl End {
    pub fn as_str(self) -> &'static str {
        match self {
            End::Ok => "ok",
            End::Cancelled => "cancelled",
            End::Error => "error",
        }
    }
}

/// One span of a finished tree (GLM-5's per-window trace, `flash::glm5::trace`): `parent`
/// indexes an earlier entry of the same slice (`None`: the root).
#[derive(Debug, Clone)]
pub struct TreeSpan {
    pub name: &'static str,
    pub parent: Option<usize>,
    pub start: std::time::SystemTime,
    pub end: std::time::SystemTime,
    pub ints: Vec<(&'static str, i64)>,
    pub floats: Vec<(&'static str, f64)>,
}

#[cfg(not(feature = "otel"))]
mod imp {
    use super::*;

    /// Is telemetry installed? (Never without the feature.)
    #[inline(always)]
    pub fn enabled() -> bool {
        false
    }

    /// Emit a finished span tree (nothing without the feature).
    #[inline(always)]
    pub fn glm5_tree(_: &[TreeSpan], _phase: &'static str) {}

    /// Exporters to flush on drop (none without the feature).
    pub struct Guard;

    pub fn init() -> Option<Guard> {
        None
    }

    /// A request's trace (nothing without the feature).
    #[derive(Debug, Default)]
    pub struct Trace;

    impl Trace {
        /// No trace (a request that wasn't received over HTTP).
        #[inline(always)]
        pub fn none() -> Self {
            Trace
        }
        #[inline(always)]
        pub fn incoming(_: &HeaderMap, _: &str, _: &str, _: Option<&str>) -> Self {
            Trace
        }
        #[inline(always)]
        pub fn dequeued(&mut self, _queue_depth: impl FnOnce() -> usize) {}
        #[inline(always)]
        pub fn finish(self, _: &Usage, _: End, _: u32, _: Option<&str>) {}
    }

    #[inline(always)]
    pub fn step(_: &StepStats) {}

    /// One engine forward window's wall time and width (nothing without the feature).
    #[inline(always)]
    pub fn glm5_window(_: usize, _: f64) {}

    #[inline(always)]
    pub fn queue_depth(_: impl FnOnce() -> usize) {}

    /// Whether DeepSeek forwards are traced (never without the feature).
    #[inline(always)]
    pub fn ds_enabled() -> bool {
        false
    }
    #[cfg(feature = "cuda")]
    #[inline(always)]
    pub fn ds_forward(_: &crate::flash::deepseek::obs::StepRec) {}
    #[cfg(feature = "cuda")]
    #[inline(always)]
    pub fn ds_spec(_: &crate::flash::deepseek::obs::SpecRec) {}
    #[inline(always)]
    pub fn ds_flush_orphans() {}
}

#[cfg(feature = "otel")]
mod imp {
    use super::*;
    use opentelemetry::metrics::{Counter, Gauge, Histogram, Meter, MeterProvider as _};
    use opentelemetry::propagation::{Extractor, TextMapPropagator};
    use opentelemetry::trace::{
        Span as _, SpanKind, Status, TraceContextExt, Tracer as _, TracerProvider as _,
    };
    use opentelemetry::{Context, KeyValue};
    use opentelemetry_sdk::metrics::SdkMeterProvider;
    use opentelemetry_sdk::propagation::TraceContextPropagator;
    use opentelemetry_sdk::trace::{SdkTracer, SdkTracerProvider};
    use std::sync::OnceLock;
    use std::time::{Duration, SystemTime};

    struct Inst {
        tracer: SdkTracer,
        step_secs: Histogram<f64>,
        step_rows: Histogram<u64>,
        chunk_tokens: Histogram<u64>,
        hit_ratio: Histogram<f64>,
        queue_depth: Gauge<u64>,
        tok_s: Histogram<f64>,
        glm5_window: Histogram<f64>,
        ds_forward_ms: Histogram<f64>,
        ds_spec_accepted: Histogram<u64>,
        ds_launches: Counter<u64>,
        ds_syncs: Counter<u64>,
        ds_arena: Gauge<u64>,
        glm5_stage: Histogram<f64>,
        /// `kind` = decode | mixed | prefill, built once.
        kinds: [[KeyValue; 1]; 3],
    }

    static INST: OnceLock<Inst> = OnceLock::new();

    /// The exporters (crates/tang-otel), flushed and shut down on drop.
    pub struct Guard(#[allow(dead_code)] tang_otel::Guard);

    /// Starts telemetry through `tang-otel`: OTLP/JSON lines to `OTEL_FILE`
    /// (default `~/.cache/tang/otel/tang-llm.otel.jsonl`; `OTEL_FILE=off` for none),
    /// and over OTLP/HTTP to `OTEL_EXPORTER_OTLP_ENDPOINT` only with
    /// `TANG_OTEL_LIVE=1` (`serve --otel-live`). Call off any async runtime.
    pub fn init() -> Option<Guard> {
        let g = tang_otel::init("tang-llm")?;
        let (traces, metrics) = tang_otel::providers()?;
        install(traces, metrics);
        Some(Guard(g))
    }

    /// Record to these providers from now on (once per process; later calls are ignored).
    pub fn install(traces: &SdkTracerProvider, metrics: &SdkMeterProvider) {
        let meter: Meter = metrics.meter("tang-llm");
        let _ = INST.set(Inst {
            tracer: traces.tracer("tang-llm"),
            step_secs: meter
                .f64_histogram("tang_llm.step.duration")
                .with_unit("s")
                .with_description(
                    "One batched forward, by kind: decode, mixed (decode + prefill) or prefill",
                )
                .with_boundaries(vec![
                    0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0,
                ])
                .build(),
            step_rows: meter
                .u64_histogram("tang_llm.step.rows")
                .with_description("Streams in one batched forward")
                .with_boundaries(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
                .build(),
            chunk_tokens: meter
                .u64_histogram("tang_llm.prefill.chunk_tokens")
                .with_unit("{token}")
                .with_description("Tokens in one prefill chunk")
                .with_boundaries(vec![
                    32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0,
                ])
                .build(),
            hit_ratio: meter
                .f64_histogram("tang_llm.prefix.hit_ratio")
                .with_description("Of a request's prompt, the share reused from KV caches")
                .with_boundaries(vec![0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
                .build(),
            queue_depth: meter
                .u64_gauge("tang_llm.queue.depth")
                .with_description("Requests waiting to run")
                .build(),
            tok_s: meter
                .f64_histogram("tang_llm.stream.tokens_per_second")
                .with_unit("{token}/s")
                .with_description("A request's decode tokens per second")
                .with_boundaries(vec![
                    5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0,
                ])
                .build(),
            glm5_window: meter
                .f64_histogram("tang_llm.glm5.window")
                .with_unit("s")
                .with_description("One GLM-5 engine window's wall time, by token count")
                .with_boundaries(vec![
                    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
                ])
                .build(),
            ds_forward_ms: meter
                .f64_histogram("deepseek.forward.ms")
                .with_unit("ms")
                .with_description("One DeepSeek forward's device time, by kind and path")
                .with_boundaries(vec![
                    1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 4096.0,
                ])
                .build(),
            ds_spec_accepted: meter
                .u64_histogram("deepseek.spec.accepted")
                .with_description("Drafts accepted per speculative step")
                .with_boundaries(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
                .build(),
            ds_launches: meter
                .u64_counter("deepseek.launches")
                .with_description("Kernel launches in observed forwards")
                .build(),
            ds_syncs: meter
                .u64_counter("deepseek.syncs")
                .with_description("Host syncs in observed forwards")
                .build(),
            ds_arena: meter
                .u64_gauge("deepseek.arena.bytes")
                .with_unit("By")
                .with_description("The step arena's device bytes")
                .build(),
            glm5_stage: meter
                .f64_histogram("tang_llm.glm5.stage.gpu")
                .with_unit("ms")
                .with_description("GPU time of one GLM-5 stage instance (TANG_GLM5_TRACE)")
                .with_boundaries(vec![
                    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 100.0,
                ])
                .build(),
            kinds: [
                [KeyValue::new("kind", "decode")],
                [KeyValue::new("kind", "mixed")],
                [KeyValue::new("kind", "prefill")],
            ],
        });
    }

    struct Headers<'a>(&'a HeaderMap);

    impl Extractor for Headers<'_> {
        fn get(&self, key: &str) -> Option<&str> {
            self.0.get(key).and_then(|v| v.to_str().ok())
        }
        fn keys(&self) -> Vec<&str> {
            self.0.keys().map(|k| k.as_str()).collect()
        }
    }

    struct Live {
        parent: Context,
        received: SystemTime,
        dequeued: Option<SystemTime>,
        priority: &'static str,
        /// `request.id` (if any) and `gen_ai.request.model`, on every span.
        common: Vec<KeyValue>,
    }

    /// A request's trace: its parent context and when it reached each phase.
    #[derive(Default)]
    pub struct Trace(Option<Box<Live>>);

    impl std::fmt::Debug for Trace {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(if self.0.is_some() {
                "Trace(live)"
            } else {
                "Trace(off)"
            })
        }
    }

    impl Trace {
        /// No trace (a request that wasn't received over HTTP).
        pub fn none() -> Self {
            Trace(None)
        }

        pub fn incoming(
            headers: &HeaderMap,
            priority: &str,
            model: &str,
            request_id: Option<&str>,
        ) -> Self {
            if INST.get().is_none() {
                return Trace(None);
            }
            let parent = TraceContextPropagator::new().extract(&Headers(headers));
            let mut common = vec![KeyValue::new("gen_ai.request.model", model.to_string())];
            if let Some(id) = request_id {
                common.push(KeyValue::new("request.id", id.to_string()));
            }
            Trace(Some(Box::new(Live {
                parent,
                received: SystemTime::now(),
                dequeued: None,
                priority: if priority == "background" {
                    "background"
                } else {
                    "interactive"
                },
                common,
            })))
        }

        /// It left the queue (its first run; a yielded request's later runs keep this).
        pub fn dequeued(&mut self, queue_depth: impl FnOnce() -> usize) {
            let Some(inst) = INST.get() else { return };
            inst.queue_depth.record(queue_depth() as u64, &[]);
            if let Some(l) = self.0.as_mut() {
                l.dequeued.get_or_insert_with(SystemTime::now);
            }
        }

        /// Emit the request's spans, ending now. `usage` is as the client sees it.
        pub fn finish(self, usage: &Usage, end: End, yields: u32, error: Option<&str>) {
            let (Some(inst), Some(l)) = (INST.get(), self.0) else {
                return;
            };
            let now = SystemTime::now();
            let dequeued = l.dequeued.unwrap_or(now).max(l.received);
            let decode_secs = if usage.decode_tok_s > 0.0 && usage.completion_tokens > 0 {
                usage.completion_tokens as f64 / usage.decode_tok_s
            } else {
                0.0
            };
            let decode_start = now
                .checked_sub(Duration::from_secs_f64(decode_secs))
                .unwrap_or(now)
                .max(dequeued);
            if usage.prompt_tokens > 0 {
                inst.hit_ratio
                    .record(usage.cached_tokens as f64 / usage.prompt_tokens as f64, &[]);
            }
            if usage.decode_tok_s > 0.0 {
                inst.tok_s.record(usage.decode_tok_s, &[]);
            }

            let t = &inst.tracer;
            let mut root = t
                .span_builder("chat.completions")
                .with_kind(SpanKind::Server)
                .with_start_time(l.received)
                .with_attributes(l.common.clone())
                .start_with_context(t, &l.parent);
            root.set_attribute(KeyValue::new("outcome", end.as_str()));
            if end == End::Error {
                root.set_status(Status::error(error.unwrap_or("error").to_string()));
            }
            let cx = l
                .parent
                .with_remote_span_context(root.span_context().clone());
            let child =
                |name: &'static str, from: SystemTime, to: SystemTime, mut attrs: Vec<KeyValue>| {
                    attrs.extend(l.common.iter().cloned());
                    let mut s = t
                        .span_builder(name)
                        .with_start_time(from)
                        .with_attributes(attrs)
                        .start_with_context(t, &cx);
                    s.end_with_timestamp(to);
                    cx.with_remote_span_context(s.span_context().clone())
                };
            let mut ds_prefill = cx.clone();
            let mut ds_decode = cx.clone();
            let wait = dequeued.duration_since(l.received).unwrap_or_default();
            child(
                "queued",
                l.received,
                dequeued,
                vec![
                    KeyValue::new("priority", l.priority),
                    KeyValue::new("queue.wait_ms", wait.as_secs_f64() * 1e3),
                    KeyValue::new("yields", yields as i64),
                ],
            );
            if l.dequeued.is_some() {
                ds_prefill = child(
                    "prefill",
                    dequeued,
                    decode_start,
                    vec![
                        KeyValue::new("prompt_tokens", usage.prompt_tokens as i64),
                        KeyValue::new("cached_tokens", usage.cached_tokens as i64),
                        KeyValue::new(
                            "new_tokens",
                            usage.prompt_tokens.saturating_sub(usage.cached_tokens) as i64,
                        ),
                        KeyValue::new("chunks", usage.prefill_chunks as i64),
                    ],
                );
            }
            if decode_secs > 0.0 {
                let steps = usage.decode_steps.max(1) as f64;
                ds_decode = child(
                    "decode",
                    decode_start,
                    now,
                    vec![
                        KeyValue::new("completion_tokens", usage.completion_tokens as i64),
                        KeyValue::new("draft_tokens", usage.draft_tokens as i64),
                        KeyValue::new("accepted_draft_tokens", usage.accepted_tokens as i64),
                        KeyValue::new("verify_windows", usage.decode_steps as i64),
                        KeyValue::new(
                            "tokens_per_verify_window",
                            usage.completion_tokens as f64 / steps,
                        ),
                        KeyValue::new("shared_streams_avg", usage.shared_streams as f64 / steps),
                        KeyValue::new("tokens_per_second", usage.decode_tok_s),
                    ],
                );
            }
            ds_emit(&ds_prefill, &ds_decode);
            child(
                "finish",
                now,
                now,
                vec![KeyValue::new("outcome", end.as_str())],
            );
            root.end_with_timestamp(now);
        }
    }

    /// One batched step's metrics (nothing for a step that ran no forward).
    pub fn step(s: &StepStats) {
        let Some(inst) = INST.get() else { return };
        if s.secs == 0.0 {
            return;
        }
        let kind = match (s.decode_streams > 0, s.prefill_chunks > 0) {
            (true, false) => 0,
            (true, true) => 1,
            _ => 2,
        };
        inst.step_secs.record(s.secs, &inst.kinds[kind]);
        inst.step_rows.record(
            (s.decode_streams + s.prefill_chunks) as u64,
            &inst.kinds[kind],
        );
        if let Some(mean) = s.prefill_tokens.checked_div(s.prefill_chunks) {
            inst.chunk_tokens.record(mean as u64, &[]);
        }
    }

    /// One GLM-5 engine window: its token count and wall seconds (the engine's own
    /// per-stage timers live inside `Glm5Engine`; this is the serve-side view).
    pub fn glm5_window(tokens: usize, secs: f64) {
        let Some(inst) = INST.get() else { return };
        if secs <= 0.0 {
            return;
        }
        let bucket = match tokens {
            0..=1 => "decode",
            2..=64 => "prefill_small",
            _ => "prefill",
        };
        inst.glm5_window
            .record(secs, &[KeyValue::new("kind", bucket)]);
    }

    /// Is telemetry installed ([`init`] / [`install`] ran)?
    pub fn enabled() -> bool {
        INST.get().is_some()
    }

    /// Emit a finished span tree: one trace per root (a GLM-5 window), each child under
    /// its parent, with explicit start/end times; every non-root span's GPU ms also feeds
    /// `tang_llm.glm5.stage.gpu` by stage name and `phase` (decode | prefill).
    pub fn glm5_tree(spans: &[TreeSpan], phase: &'static str) {
        let Some(inst) = INST.get() else { return };
        let t = &inst.tracer;
        let mut live: Vec<opentelemetry_sdk::trace::Span> = Vec::with_capacity(spans.len());
        for (i, sp) in spans.iter().enumerate() {
            let mut attrs: Vec<KeyValue> = Vec::with_capacity(sp.ints.len() + sp.floats.len() + 1);
            attrs.push(KeyValue::new("phase", phase));
            attrs.extend(sp.ints.iter().map(|&(k, v)| KeyValue::new(k, v)));
            attrs.extend(sp.floats.iter().map(|&(k, v)| KeyValue::new(k, v)));
            let cx = match sp.parent.filter(|&p| p < i) {
                Some(p) => Context::new().with_remote_span_context(live[p].span_context().clone()),
                None => Context::new(),
            };
            let s = t
                .span_builder(sp.name)
                .with_kind(SpanKind::Internal)
                .with_start_time(sp.start)
                .with_attributes(attrs)
                .start_with_context(t, &cx);
            if sp.parent.is_some() {
                if let Some(&(_, ms)) = sp.floats.iter().find(|(k, _)| *k == "gpu.ms") {
                    inst.glm5_stage.record(
                        ms,
                        &[
                            KeyValue::new("stage", sp.name),
                            KeyValue::new("phase", phase),
                        ],
                    );
                }
            }
            live.push(s);
        }
        for (s, sp) in live.iter_mut().zip(spans) {
            s.end_with_timestamp(sp.end);
        }
    }

    /// Requests waiting, sampled between steps.
    pub fn queue_depth(n: impl FnOnce() -> usize) {
        if let Some(inst) = INST.get() {
            inst.queue_depth.record(n() as u64, &[]);
        }
    }

    // DeepSeek forwards (flash/deepseek/obs.rs): buffered as span trees, emitted under the
    // request's prefill / decode spans when it finishes (or by `ds_flush_orphans`).

    struct Sub {
        name: std::borrow::Cow<'static, str>,
        start: SystemTime,
        end: SystemTime,
        attrs: Vec<KeyValue>,
        children: Vec<Sub>,
        /// prefill-time work (prefill, the drafter's seeding) vs decode
        prefill: bool,
        rank: usize,
        parent: u64,
        step: u64,
    }

    /// Buffered spans; bounded so a process with no requests can't grow it forever.
    static DS: std::sync::Mutex<Vec<Sub>> = std::sync::Mutex::new(Vec::new());
    const DS_MAX: usize = 20_000;

    fn ds_push(s: Sub) {
        let mut v = DS.lock().unwrap();
        if v.len() < DS_MAX {
            v.push(s);
        }
    }

    pub fn ds_enabled() -> bool {
        INST.get().is_some()
    }

    #[cfg(feature = "cuda")]
    pub fn ds_forward(r: &crate::flash::deepseek::obs::StepRec) {
        use crate::flash::deepseek::obs;
        let Some(inst) = INST.get() else { return };
        let kp = [KeyValue::new("kind", r.kind), KeyValue::new("path", r.path)];
        if r.rank == 0 {
            inst.ds_forward_ms.record(r.gpu_ms, &kp);
            inst.ds_launches.add(r.counts.launches, &kp);
            inst.ds_syncs.add(r.counts.syncs, &kp);
            inst.ds_arena.record(r.arena_bytes, &[]);
        }
        // spans: every DS_OBS_EVERY-th step, every step at DS_OBS=2
        if obs::level() < 2 && r.step % obs::every() != 0 {
            return;
        }
        let ms = |m: f64| Duration::from_secs_f64(m.max(0.0) / 1e3);
        // build the span tree from the flat list (parents come first)
        let mut nodes: Vec<Sub> = r
            .spans
            .iter()
            .map(|s| Sub {
                // the leaf: `q`, not `layer.attn.q` (that is the `op` attribute)
                name: s.name.rsplit('.').next().unwrap_or("op").to_string().into(),
                start: r.start + ms(s.start_ms),
                end: r.start + ms(s.start_ms + s.gpu_ms),
                attrs: vec![
                    KeyValue::new("op", s.name.clone()),
                    KeyValue::new("layer", s.layer as i64),
                    KeyValue::new("gpu_ms", s.gpu_ms),
                    KeyValue::new("host_ms", s.host_ms),
                ],
                children: Vec::new(),
                prefill: false,
                rank: r.rank,
                parent: r.parent,
                step: r.step,
            })
            .collect();
        let mut top = Vec::new();
        for i in (0..nodes.len()).rev() {
            let n = nodes.pop().unwrap();
            let p = r.spans[i].parent;
            if p >= 0 {
                nodes[p as usize].children.insert(0, n);
            } else {
                top.insert(0, n);
            }
        }
        let c = &r.counts;
        ds_push(Sub {
            name: "forward".into(),
            start: r.start,
            end: r.start + ms(r.wall_ms),
            attrs: vec![
                KeyValue::new("kind", r.kind),
                KeyValue::new("t", r.t as i64),
                KeyValue::new("sp", r.sp as i64),
                KeyValue::new("path", r.path),
                KeyValue::new("captured", r.captured),
                KeyValue::new("rank", r.rank as i64),
                KeyValue::new("step", r.step as i64),
                KeyValue::new("gpu_ms", r.gpu_ms),
                KeyValue::new("host_ms", r.host_ms),
                KeyValue::new("launches", c.launches as i64),
                KeyValue::new("syncs", c.syncs as i64),
                KeyValue::new("d2h_bytes", c.d2h_bytes as i64),
                KeyValue::new("h2d_bytes", c.h2d_bytes as i64),
                KeyValue::new("allocs", c.allocs as i64),
                KeyValue::new("expert_uploads", c.expert_uploads as i64),
                KeyValue::new("reduce_wait_ms", r.reduce_wait_ms),
            ],
            children: top,
            prefill: r.kind == "prefill" || (r.kind == "draft" && r.sp == 0),
            rank: r.rank,
            parent: r.parent,
            step: r.step,
        });
    }

    #[cfg(feature = "cuda")]
    pub fn ds_spec(r: &crate::flash::deepseek::obs::SpecRec) {
        let Some(inst) = INST.get() else { return };
        if r.rank != 0 {
            return;
        }
        inst.ds_spec_accepted.record(r.accepted as u64, &[]);
        let total = r.draft_ms + r.verify_ms + r.rewind_ms;
        ds_push(Sub {
            name: "spec_step".into(),
            start: r.start,
            end: r.start + Duration::from_secs_f64(total / 1e3),
            attrs: vec![
                KeyValue::new("drafted", r.drafted as i64),
                KeyValue::new("accepted", r.accepted as i64),
                KeyValue::new("draft_ms", r.draft_ms),
                KeyValue::new("verify_ms", r.verify_ms),
                KeyValue::new("rewind_ms", r.rewind_ms),
            ],
            children: Vec::new(),
            prefill: false,
            rank: 0,
            parent: 0,
            step: 0,
        });
    }

    fn ds_span(t: &SdkTracer, s: Sub, cx: &Context) {
        let mut span = t
            .span_builder(s.name)
            .with_start_time(s.start)
            .with_attributes(s.attrs)
            .start_with_context(t, cx);
        let ccx = cx.with_remote_span_context(span.span_context().clone());
        for c in s.children {
            ds_span(t, c, &ccx);
        }
        span.end_with_timestamp(s.end);
    }

    /// Emit the buffered spans: rank 0's forwards under `prefill` / `decode`, other ranks'
    /// forwards under rank 0's forward for the same EP command (`parent`).
    fn ds_emit(prefill: &Context, decode: &Context) {
        let Some(inst) = INST.get() else { return };
        let subs = std::mem::take(&mut *DS.lock().unwrap());
        if subs.is_empty() {
            return;
        }
        let t = &inst.tracer;
        let (mine, others): (Vec<Sub>, Vec<Sub>) = subs.into_iter().partition(|s| s.rank == 0);
        let mut others: Vec<Option<Sub>> = others.into_iter().map(Some).collect();
        for s in mine {
            let cx = if s.prefill { prefill } else { decode };
            let link = s.name == "forward" && s.parent != 0;
            let (parent, step) = (s.parent, s.step);
            let mut span = t
                .span_builder(s.name)
                .with_start_time(s.start)
                .with_attributes(s.attrs)
                .start_with_context(t, cx);
            let ccx = cx.with_remote_span_context(span.span_context().clone());
            for c in s.children {
                ds_span(t, c, &ccx);
            }
            if link {
                for o in others.iter_mut() {
                    if o.as_ref()
                        .is_some_and(|o| o.parent == parent && o.step == step)
                    {
                        ds_span(t, o.take().unwrap(), &ccx);
                    }
                }
            }
            span.end_with_timestamp(s.end);
        }
        for o in others.into_iter().flatten() {
            ds_span(t, o, decode);
        }
    }

    /// Emit buffered DeepSeek spans that no request claimed (the profile command) under one
    /// `deepseek.profile` root.
    pub fn ds_flush_orphans() {
        let Some(inst) = INST.get() else { return };
        let start = DS.lock().unwrap().iter().map(|s| s.start).min();
        let Some(start) = start else { return };
        let t = &inst.tracer;
        let mut root = t
            .span_builder("deepseek.profile")
            .with_start_time(start)
            .start(t);
        let cx = Context::new().with_remote_span_context(root.span_context().clone());
        ds_emit(&cx, &cx);
        root.end();
    }
}

pub use imp::*;
