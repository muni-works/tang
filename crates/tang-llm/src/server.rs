//! OpenAI-compatible HTTP server: `/v1/chat/completions` (streaming and not), `/v1/models`,
//! and llama.cpp-style `/props` so clients can discover the context window. `/v1/prefill`
//! takes a chat completion body and only prefills it, so the request that follows (the same
//! conversation plus a new message) starts from a warm cache.
//!
//! For frog's scheduler (see `docs/node.md`): `GET /node` describes the machine, its model,
//! measured rates, queue and KV blocks; `POST /models/load` and `/models/unload` swap the model
//! (one per process), loading only into free memory. Requests carry `x-frog-priority:
//! interactive | background`: interactive ones run first, and a background prefill gives way
//! between chunks when an interactive request is waiting (see [`crate::queue`]).
//!
//! The model lives on one worker thread (GPU state isn't shareable); requests queue for it.
//! Flash GGUF serving uses the same queue and HTTP routes, with prompt rendering on the
//! worker because it depends on prior generated tokens. Its model is fixed: load/unload
//! return 400, background prefills do not yield, and dense KV block reporting is unavailable.
//!
//! With an API key, every route but `/health` wants `Authorization: Bearer <key>`.

use crate::chat::Piece;
use crate::engine::{Control, Engine, Finish, Outcome, Progress, Prompter, Request, Usage};
use crate::model::Dtype;
use crate::node::{Need, Probe, Rates};
use crate::queue::{Priority, Queue, Ticket};
use crate::sample::Sampling;
use crate::telemetry::{End, Trace};
use axum::extract::{Request as HttpRequest, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tang_compute::ComputeDevice;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;

enum Out {
    Piece(Piece),
    Done(Finish, Usage),
    Error(String),
}

/// A load's or unload's answer: a JSON body, or a status and message.
type Reply = Result<Value, (StatusCode, String)>;

enum Job {
    Complete {
        req: Box<Request>,
        /// Its prompt and token ids, made by `model`'s prompter when it was queued.
        prompt: Option<(String, Vec<u32>)>,
        tx: mpsc::UnboundedSender<Out>,
        /// The model it was accepted for: it fails rather than run on another.
        model: String,
        /// What earlier runs did before giving way.
        before: Option<Before>,
        /// The client's `x-request-id`, logged with the request's usage.
        request_id: Option<String>,
        /// Its OpenTelemetry trace (empty unless the `otel` feature exports).
        trace: Trace,
    },
    Load {
        spec: String,
        reply: oneshot::Sender<Reply>,
    },
    Unload {
        reply: oneshot::Sender<Reply>,
    },
}

/// A request's first run, when it gave way part way through its prefill.
#[derive(Debug, Clone, Copy)]
struct Before {
    cached_tokens: usize,
    prefilled: usize,
    secs: f64,
}

/// What the server reports and is shared between the worker and the handlers.
struct Node {
    id: String,
    queue: Queue<Job>,
    state: Mutex<NodeState>,
    hardware: Probe,
    dtype: Dtype,
    /// Serialize model loads and heavy inference across the judge and image workers.
    resources: Arc<crate::image_server::Gate>,
    image: Option<crate::image_server::ImageModel>,
    /// The node's Ed25519 identity (clients pin its fingerprint).
    identity: Arc<crate::auth::Identity>,
    /// The host name.
    name: String,
    /// A model downloading onto this node (`POST /models/pull`).
    pull: Arc<crate::pull::Puller>,
}

#[derive(Default)]
struct NodeState {
    model: Option<Loaded>,
    /// A model being loaded, and since when.
    loading: Option<(String, Instant)>,
    /// Requests running, oldest first (several when the backend batches).
    running: Vec<Running>,
    /// Why the engine can't run anything more (a device fault mid-step); `/health` reports
    /// it and new requests fail instead of joining a broken engine.
    unhealthy: Option<String>,
}

struct Loaded {
    name: String,
    path: PathBuf,
    ctx: usize,
    /// The model takes images.
    vision: bool,
    /// Unix seconds.
    loaded_at: u64,
    load_secs: f64,
    /// Its weights' estimated device memory.
    need: Option<Need>,
    rates: Rates,
    prompter: Option<Prompter>,
    kv: Kv,
}

/// The loaded model's KV, as of the last request.
#[derive(Default)]
struct Kv {
    /// Sealed blocks in memory.
    blocks: Vec<u64>,
    /// Each `prompt_cache_key` with a cache, and the positions it holds.
    keys: Vec<(String, usize)>,
    /// The disk tier's directory (its `blocks/` are listed per `/node`).
    store: Option<PathBuf>,
    /// Device memory the block pool holds.
    pool_bytes: usize,
    /// The fitted step-time model (see `crate::step_cost`), if the backend keeps one.
    step_cost: Option<Value>,
}

struct Running {
    ticket: Ticket,
    progress: Progress,
    started: Instant,
}

impl Node {
    fn state(&self) -> MutexGuard<'_, NodeState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Requests running or waiting.
    fn busy(&self) -> usize {
        self.queue.len() + self.state().running.len()
    }
}

#[derive(Clone)]
struct App {
    node: Arc<Node>,
}

/// `ok`, or 503 with why the engine stopped (see [`NodeState::unhealthy`]).
async fn health(State(app): State<App>) -> Response {
    match app.node.state().unhealthy.clone() {
        None => "ok".into_response(),
        Some(why) => (StatusCode::SERVICE_UNAVAILABLE, format!("unhealthy: {why}")).into_response(),
    }
}

/// A model owned by the server's worker thread. Backends may keep prompt rendering on
/// that thread when it depends on mutable generation history (as Flash does).
pub trait Backend {
    fn complete(
        &mut self,
        req: &Request,
        on: &mut dyn FnMut(Piece) -> bool,
    ) -> anyhow::Result<(Finish, Usage)>;
    fn context_window(&self) -> usize;
    fn vision(&self) -> bool {
        false
    }
    fn prompter(&self) -> Option<Prompter> {
        None
    }
    fn model_path(&self, spec: &str) -> anyhow::Result<PathBuf> {
        crate::resolve_model(spec)
    }
    /// Backends with a fixed GGUF/settings bundle do not support model swapping.
    fn swappable() -> bool {
        false
    }
    fn complete_with(
        &mut self,
        req: &Request,
        _prompt: Option<(String, Vec<u32>)>,
        on: &mut dyn FnMut(Piece) -> bool,
        _ctl: &mut dyn Control,
    ) -> anyhow::Result<Outcome> {
        self.complete(req, on).map(|(f, u)| Outcome::Done(f, u))
    }
    fn block_hashes(&self) -> Vec<u64> {
        Vec::new()
    }
    fn cached_keys(&self) -> Vec<(String, usize)> {
        Vec::new()
    }
    fn kv_store_dir(&self) -> Option<&std::path::Path> {
        None
    }
    fn kv_bytes(&self) -> usize {
        0
    }
    fn flush_kv_store(&self) {}
    /// Drop prompt-derived state unused for the TTL ([`crate::retention`]); the worker calls
    /// it periodically, between requests.
    fn expire(&mut self) {}
    /// Runs after the reply goes out, including failed requests.
    fn after(&mut self, _req: &Request, _result: &anyhow::Result<(Finish, Usage)>) {}

    /// Most requests it runs at once (see `docs/batching.md`); 1 runs them one at a time
    /// with `complete_with`, and the methods below go unused.
    fn max_batch(&self) -> usize {
        1
    }
    /// A request in conversation `key` could start now (a slot is free and the conversation
    /// isn't running).
    fn can_admit(&self, _key: Option<&str>) -> bool {
        false
    }
    /// The KV pool has room for another stream of `rows` positions.
    fn kv_room(&self, _rows: usize) -> bool {
        true
    }
    /// Start a request in the batch (see [`Engine::admit`]).
    fn admit(
        &mut self,
        _req: &Request,
        _prompt: Option<(String, Vec<u32>)>,
    ) -> anyhow::Result<Inflight> {
        anyhow::bail!("this backend doesn't batch")
    }
    /// One step for every lane (see [`Engine::step`]).
    fn step(&mut self, _lanes: &mut [DynLane<'_>]) -> anyhow::Result<()> {
        anyhow::bail!("this backend doesn't batch")
    }
    /// Done with a stream (see [`Engine::retire`]).
    fn retire(&mut self, _s: Inflight) -> anyhow::Result<Outcome> {
        anyhow::bail!("this backend doesn't batch")
    }
    /// The fitted step-time model, for `/node`.
    fn step_cost(&self) -> Option<Value> {
        None
    }
    /// A stream's state: (done, still prefilling).
    fn stream_state(&self, _s: &Inflight) -> (bool, bool) {
        (true, false)
    }
}

/// A batching backend's stream, opaque to the worker.
pub type Inflight = Box<dyn std::any::Any>;

/// A failed [`Backend::step`] that knows which lanes it hurt: the worker retires only those
/// (indices into the step's `lanes`) and the rest keep decoding. `fatal`: the engine itself
/// is broken (a device fault), so every stream ends and the node reports unhealthy.
/// A step error that isn't one of these ends every stream (see [`step_verdict`]).
#[derive(Debug)]
pub struct StepFailure {
    pub lanes: Vec<usize>,
    pub fatal: bool,
    pub error: String,
}

impl std::fmt::Display for StepFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.error)
    }
}

impl std::error::Error for StepFailure {}

impl StepFailure {
    /// `e` for `lanes`, fatal when it reads as a device fault.
    pub fn lanes(lanes: Vec<usize>, e: anyhow::Error) -> anyhow::Error {
        let error = format!("{e:#}");
        anyhow::Error::new(StepFailure {
            lanes,
            fatal: device_fault(&error),
            error,
        })
    }
}

/// The message reads as a GPU fault: after one the device context is unusable.
pub fn device_fault(msg: &str) -> bool {
    msg.contains("CUDA_ERROR") || msg.contains("CUDA error") || msg.contains("cudaError")
}

/// Which streams a step's outcome ends.
#[derive(Debug, PartialEq)]
enum Verdict {
    Ok,
    /// These lanes, with this error; the others go on.
    Lanes(Vec<usize>, String),
    /// Every stream, with this error; `true`: the engine is broken too.
    All(String, bool),
}

/// A step's result (`Err(None)`: it panicked) as what to retire.
fn step_verdict(result: Result<anyhow::Result<()>, String>) -> Verdict {
    match result {
        Ok(Ok(())) => Verdict::Ok,
        Err(panic) => Verdict::All(format!("the engine failed: {panic}"), true),
        Ok(Err(e)) => match e.downcast_ref::<StepFailure>() {
            Some(f) if f.fatal => Verdict::All(f.error.clone(), true),
            Some(f) => Verdict::Lanes(f.lanes.clone(), f.error.clone()),
            None => {
                let msg = format!("{e:#}");
                let fatal = device_fault(&msg);
                Verdict::All(msg, fatal)
            }
        },
    }
}

/// A stream in a [`Backend::step`], with where its pieces go and who controls it.
pub struct DynLane<'a> {
    pub stream: &'a mut Inflight,
    pub on: &'a mut dyn FnMut(Piece) -> bool,
    pub ctl: &'a mut dyn Control,
}

impl<D: ComputeDevice + 'static> Backend for Engine<D> {
    fn complete(
        &mut self,
        req: &Request,
        on: &mut dyn FnMut(Piece) -> bool,
    ) -> anyhow::Result<(Finish, Usage)> {
        Engine::complete(self, req, on)
    }
    fn context_window(&self) -> usize {
        Engine::context_window(self)
    }
    fn vision(&self) -> bool {
        self.model.sees_images()
    }
    fn prompter(&self) -> Option<Prompter> {
        Some(Engine::prompter(self))
    }
    fn swappable() -> bool {
        true
    }
    fn complete_with(
        &mut self,
        req: &Request,
        prompt: Option<(String, Vec<u32>)>,
        on: &mut dyn FnMut(Piece) -> bool,
        ctl: &mut dyn Control,
    ) -> anyhow::Result<Outcome> {
        Engine::complete_with(self, req, prompt, on, ctl)
    }
    fn block_hashes(&self) -> Vec<u64> {
        Engine::block_hashes(self)
    }
    fn cached_keys(&self) -> Vec<(String, usize)> {
        Engine::cached_keys(self)
    }
    fn kv_store_dir(&self) -> Option<&std::path::Path> {
        Engine::kv_store_dir(self)
    }
    fn kv_bytes(&self) -> usize {
        Engine::kv_bytes(self)
    }
    fn flush_kv_store(&self) {
        Engine::flush_kv_store(self)
    }
    fn expire(&mut self) {
        Engine::expire(self)
    }
    fn after(&mut self, req: &Request, result: &anyhow::Result<(Finish, Usage)>) {
        if let Ok((_, usage)) = result {
            if self.speculation().is_some() {
                eprintln!(
                    "tang-llm: {} tokens at {:.1} tok/s, drafts {}/{} accepted",
                    usage.completion_tokens,
                    usage.decode_tok_s,
                    usage.accepted_tokens,
                    usage.draft_tokens
                );
            }
            self.save(req.cache_key.as_deref());
        }
    }
    fn max_batch(&self) -> usize {
        Engine::max_batch(self)
    }
    fn can_admit(&self, key: Option<&str>) -> bool {
        Engine::can_admit(self, key)
    }
    fn kv_room(&self, rows: usize) -> bool {
        Engine::kv_room(self, rows)
    }
    fn admit(
        &mut self,
        req: &Request,
        prompt: Option<(String, Vec<u32>)>,
    ) -> anyhow::Result<Inflight> {
        Ok(Box::new(Engine::admit(self, req, prompt)?))
    }
    fn step(&mut self, lanes: &mut [DynLane<'_>]) -> anyhow::Result<()> {
        let mut typed: Vec<crate::engine::Lane<'_, D::Buffer>> = lanes
            .iter_mut()
            .map(|l| crate::engine::Lane {
                stream: l
                    .stream
                    .downcast_mut()
                    .expect("a stream this engine admitted"),
                on: &mut *l.on,
                ctl: &mut *l.ctl,
            })
            .collect();
        Engine::step(self, &mut typed)
    }
    fn retire(&mut self, s: Inflight) -> anyhow::Result<Outcome> {
        let s = s
            .downcast::<crate::engine::Stream<D::Buffer>>()
            .expect("a stream this engine admitted");
        Engine::retire(self, *s)
    }
    fn step_cost(&self) -> Option<Value> {
        Some(Engine::step_cost(self).json())
    }
    fn stream_state(&self, s: &Inflight) -> (bool, bool) {
        let s: &crate::engine::Stream<D::Buffer> =
            s.downcast_ref().expect("a stream this engine admitted");
        (s.done(), s.prefilling())
    }
}

/// How to serve.
pub struct Options {
    pub addr: String,
    /// With a key, requests must present it.
    pub key: Option<String>,
    /// The model to load at start (a directory or Hugging Face repo id); `None` starts with
    /// none (load one with `POST /models/load`).
    pub model: Option<String>,
    /// How the loader keeps weights (for the fit check before a load).
    pub dtype: Dtype,
    /// Reads the hardware and its free memory.
    pub hardware: Probe,
}

/// Serve until the process exits. `load` makes an engine for a model (a directory or Hugging
/// Face repo id); it runs on the worker thread, so the GPU device is created where it's used.
pub fn serve<B, F>(opts: Options, load: F) -> anyhow::Result<()>
where
    B: Backend + 'static,
    F: Fn(&str) -> anyhow::Result<B> + Send + 'static,
{
    serve_with_images::<B, tang_compute::CpuDevice, F>(opts, load, None)
}

/// Serve chat/vision and a lazy resident image pipeline in one process. Heavy inference
/// is serialized so both workers share scratch headroom; neither evicts the other's weights.
pub fn serve_with_images<B, D, F>(
    opts: Options,
    load: F,
    image: Option<(PathBuf, fn() -> anyhow::Result<D>)>,
) -> anyhow::Result<()>
where
    B: Backend + 'static,
    D: ComputeDevice + 'static,
    F: Fn(&str) -> anyhow::Result<B> + Send + 'static,
{
    // Before any async runtime: the OTLP exporter's blocking client builds its own.
    let _otel = crate::telemetry::init();
    let resources = Arc::<crate::image_server::Gate>::default();
    let image = image.map(|(root, make)| crate::image_server::mount(root, make, resources.clone()));
    let identity = Arc::new(crate::auth::Identity::load_or_create(
        crate::auth::identity_path().as_deref(),
    )?);
    let node = Arc::new(Node {
        pull: Default::default(),
        id: crate::node::node_id(),
        queue: Queue::new(),
        state: Mutex::new(NodeState::default()),
        hardware: opts.hardware,
        dtype: opts.dtype,
        resources,
        image: image.as_ref().map(|service| service.model.clone()),
        identity,
        name: crate::advertise::host_name(),
    });
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let worker = node.clone();
    let first = opts.model.clone();
    std::thread::spawn(move || {
        let mut w = Worker {
            node: worker,
            engine: None,
            load,
            batch: Vec::new(),
        };
        if let Some(spec) = first {
            let gate = w.node.resources.clone();
            let _guard = gate.lock();
            if let Err(e) = w.load(&spec) {
                let _ = ready_tx.send(Err(e));
                return;
            }
        }
        let _ = ready_tx.send(Ok(()));
        w.run();
    });
    ready_rx.recv()??;

    let advert_node = node.clone();
    let auth = Arc::new(crate::auth::Auth::new(
        crate::advertise::reachable(&opts.addr),
        opts.key,
        crate::auth::keys_path(),
    ));
    if auth.beyond && auth.static_key.is_none() && auth.clients().is_empty() {
        eprintln!(
            "tang-llm: not paired yet: every route but /health and /pair/* answers 401 until a client pairs (`tang-llm pair`)"
        );
    }
    let pairing = Arc::new(crate::pair::Pairing::new(
        auth.clone(),
        node.identity.clone(),
        node.id.clone(),
        node.name.clone(),
    ));
    let router = router(App { node }, auth, pairing.clone(), image.map(|i| i.router));
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(&opts.addr).await?;
        let local = listener.local_addr()?;
        eprintln!("tang-llm: listening on http://{local}");
        let advert = crate::advertise::start(
            &local.to_string(),
            &advert_node.id,
            &(advert_node.hardware)(),
            &advert_node.identity.fingerprint(),
        );
        if let Some(advert) = advert {
            // Keep `pair` in the TXT record current; dropped (withdrawn) with the server.
            tokio::spawn(async move {
                let mut open = false;
                loop {
                    let now = pairing.is_open();
                    if now != open {
                        advert.set_pair(now);
                        open = now;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
            });
        }
        axum::serve(
            listener,
            router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await?;
        anyhow::Ok(())
    })
}

/// Every route: `/health` and `/pair/*` open, the rest behind [`crate::auth::gate`].
fn router(
    app: App,
    auth: Arc<crate::auth::Auth>,
    pairing: Arc<crate::pair::Pairing>,
    image: Option<Router>,
) -> Router {
    let health_app = app.clone();
    let mut api = Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/prefill", post(prefill))
        .route("/v1/models", get(models))
        .route("/props", get(props))
        .route("/node", get(node_info))
        .route("/models/load", post(load_model))
        .route("/models/unload", post(unload_model))
        .route("/models/pull", post(pull_model).delete(cancel_pull))
        .route("/models/delete", post(delete_model))
        .with_state(app);
    if let Some(image) = image {
        api = api.merge(image);
    }
    let api = api.layer(middleware::from_fn_with_state(auth, crate::auth::gate));
    Router::new()
        .route("/health", get(health).with_state(health_app))
        .merge(crate::pair::router(pairing))
        .merge(api)
}

/// The model's thread: runs jobs from the queue, interactive first.
struct Worker<B: Backend, F> {
    node: Arc<Node>,
    engine: Option<(String, B)>,
    load: F,
    /// Requests decoding together, when the backend batches.
    batch: Vec<Active>,
}

/// A request in the batch.
struct Active {
    ticket: Ticket,
    req: Box<Request>,
    tx: mpsc::UnboundedSender<Out>,
    model: String,
    before: Option<Before>,
    request_id: Option<String>,
    trace: Trace,
    /// A background request's prompt, kept in case it gives way.
    kept: Option<(String, Vec<u32>)>,
    stream: Inflight,
}

/// Lets a background request's prefill give way to waiting interactive work, and keeps
/// `/node`'s view of the running request current.
struct Ctl<'a> {
    node: &'a Node,
    background: bool,
    /// Closed once the client goes away: the run stops there.
    tx: &'a mpsc::UnboundedSender<Out>,
    /// Its ticket's id, to find it among the running.
    id: u64,
    /// In a batch: whether interactive work is waiting that couldn't join (decided once per
    /// step). Alone: `None`, and any waiting interactive work counts.
    blocked: Option<bool>,
}

impl Control for Ctl<'_> {
    fn keep_prefilling(&mut self) -> bool {
        let blocked = self
            .blocked
            .unwrap_or_else(|| self.node.queue.interactive_waiting());
        !(self.background && blocked)
    }

    fn progress(&mut self, p: Progress) {
        if let Some(r) = self
            .node
            .state()
            .running
            .iter_mut()
            .find(|r| r.ticket.id == self.id)
        {
            r.progress = p;
        }
    }

    fn cancelled(&mut self) -> bool {
        self.tx.is_closed()
    }
}

impl<B, F> Worker<B, F>
where
    B: Backend,
    F: Fn(&str) -> anyhow::Result<B>,
{
    fn run(&mut self) {
        let every = crate::retention::ttl().map(crate::retention::sweep_every);
        let mut swept = Instant::now();
        loop {
            if every.is_some_and(|e| swept.elapsed() >= e) {
                self.expire();
                swept = Instant::now();
            }
            if !self.batch.is_empty() {
                // Between steps: whoever can join does, then one step for all.
                while let Some((ticket, job)) = self.node.queue.pop_where(|t, j| self.joins(t, j)) {
                    self.dispatch(ticket, job);
                }
                if !self.batch.is_empty() {
                    self.step_batch();
                }
                continue;
            }
            let (ticket, job) = match every {
                // Idle: wake for the next sweep.
                Some(e) => match self
                    .node
                    .queue
                    .pop_timeout(e.saturating_sub(swept.elapsed()))
                {
                    Some(x) => x,
                    None => continue,
                },
                None => self.node.queue.pop(),
            };
            self.dispatch(ticket, job);
        }
    }

    /// Drop what's past the TTL, and show `/node` what's left.
    fn expire(&mut self) {
        let Some((_, e)) = &mut self.engine else {
            return;
        };
        let gate = self.node.resources.clone();
        let _guard = gate.lock();
        e.expire();
        self.refresh_kv();
    }

    /// Whether a waiting job joins the running batch now.
    fn joins(&self, ticket: &Ticket, job: &Job) -> bool {
        let Job::Complete { req, model, tx, .. } = job else {
            // Loads and unloads wait for the batch to drain.
            return false;
        };
        let Some((name, e)) = &self.engine else {
            return true;
        };
        if name != model || tx.is_closed() || self.node.state().unhealthy.is_some() {
            // It fails (or is dropped) without running.
            return true;
        }
        let keys: Vec<&str> = self
            .batch
            .iter()
            .filter_map(|a| a.req.cache_key.as_deref())
            .collect();
        let rows = ticket.prompt_tokens.unwrap_or(0) + req.max_tokens.unwrap_or(4096).min(4096);
        let room = crate::batch::Room {
            running: self.batch.len(),
            max: e.max_batch(),
            kv: e.kv_room(rows),
        };
        crate::batch::joins(req.cache_key.as_deref(), &keys, room)
            && e.can_admit(req.cache_key.as_deref())
    }

    fn dispatch(&mut self, ticket: Ticket, job: Job) {
        {
            let gate = self.node.resources.clone();
            let _guard = gate.lock();
            match job {
                Job::Complete {
                    req,
                    prompt,
                    tx,
                    model,
                    before,
                    request_id,
                    trace,
                } => self.complete(ticket, req, prompt, tx, model, before, request_id, trace),
                Job::Load { spec, reply } => {
                    let _ = reply.send(self.swap(&spec));
                }
                Job::Unload { reply } => {
                    let result = if B::swappable() {
                        Ok(self.unload())
                    } else {
                        Err((
                            StatusCode::BAD_REQUEST,
                            "this backend has a fixed model; restart to change it".into(),
                        ))
                    };
                    let _ = reply.send(result);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn complete(
        &mut self,
        ticket: Ticket,
        req: Box<Request>,
        prompt: Option<(String, Vec<u32>)>,
        tx: mpsc::UnboundedSender<Out>,
        model: String,
        before: Option<Before>,
        request_id: Option<String>,
        mut trace: Trace,
    ) {
        let rid = request_id.as_deref();
        trace.dequeued(|| self.node.queue.len());
        if tx.is_closed() {
            // The client went away while it waited.
            log_usage(rid, &model, &Usage::default(), "cancelled");
            trace.finish(&Usage::default(), End::Cancelled, ticket.yields, None);
            return;
        }
        if let Some(why) = self.node.state().unhealthy.clone() {
            let msg = format!("the engine is unhealthy ({why}); restart the server");
            log_usage(rid, &model, &Usage::default(), "error");
            trace.finish(&Usage::default(), End::Error, ticket.yields, Some(&msg));
            let _ = tx.send(Out::Error(msg));
            return;
        }
        let engine = match &mut self.engine {
            Some((name, e)) if *name == model => e,
            Some((name, _)) => {
                let msg = format!("model {model} was unloaded ({name} is loaded now)");
                log_usage(rid, &model, &Usage::default(), "error");
                trace.finish(&Usage::default(), End::Error, ticket.yields, Some(&msg));
                let _ = tx.send(Out::Error(msg));
                return;
            }
            None => {
                let msg = format!("model {model} was unloaded");
                log_usage(rid, &model, &Usage::default(), "error");
                trace.finish(&Usage::default(), End::Error, ticket.yields, Some(&msg));
                let _ = tx.send(Out::Error(msg));
                return;
            }
        };
        self.node.state().running.push(Running {
            ticket: ticket.clone(),
            progress: Progress {
                prompt_tokens: ticket.prompt_tokens.unwrap_or(0),
                ..Progress::default()
            },
            started: Instant::now(),
        });
        // A background request may give way; it keeps its prompt for when it goes on.
        let kept = match ticket.priority {
            Priority::Background => prompt.clone(),
            Priority::Interactive => None,
        };
        if engine.max_batch() > 1 {
            match engine.admit(&req, prompt) {
                Ok(stream) => self.batch.push(Active {
                    ticket,
                    req,
                    tx,
                    model,
                    before,
                    request_id,
                    trace,
                    kept,
                    stream,
                }),
                Err(e) => self.conclude(
                    ticket,
                    req,
                    tx,
                    model,
                    before,
                    request_id,
                    trace,
                    kept,
                    Err(e),
                ),
            }
            return;
        }
        let mut ctl = Ctl {
            node: &self.node,
            background: ticket.priority == Priority::Background,
            tx: &tx,
            id: ticket.id,
            blocked: None,
        };
        let piece_tx = tx.clone();
        let result = engine.complete_with(
            &req,
            prompt,
            &mut |p| piece_tx.send(Out::Piece(p)).is_ok(),
            &mut ctl,
        );
        self.conclude(
            ticket, req, tx, model, before, request_id, trace, kept, result,
        );
    }

    /// One decode step for the batch, then the finished streams' replies.
    fn step_batch(&mut self) {
        let gate = self.node.resources.clone();
        let _guard = gate.lock();
        let Some((_, engine)) = self.engine.as_mut() else {
            return;
        };
        // Interactive work still waiting now couldn't join this round; a stream giving way
        // helps only if what it lacked was a slot.
        let blocked =
            self.node.queue.interactive_waiting() && self.batch.len() >= engine.max_batch();
        let tickets: Vec<&Ticket> = self.batch.iter().map(|a| &a.ticket).collect();
        let order = crate::batch::lane_order(&tickets);
        let node = &*self.node;
        let mut batch: Vec<Option<&mut Active>> = self.batch.iter_mut().map(Some).collect();
        let mut picked: Vec<&mut Active> =
            order.iter().map(|&i| batch[i].take().unwrap()).collect();
        let txs: Vec<mpsc::UnboundedSender<Out>> = picked.iter().map(|a| a.tx.clone()).collect();
        let mut ons: Vec<Box<dyn FnMut(Piece) -> bool>> = txs
            .iter()
            .map(|tx| {
                let tx = tx.clone();
                Box::new(move |p| tx.send(Out::Piece(p)).is_ok()) as Box<dyn FnMut(Piece) -> bool>
            })
            .collect();
        let mut ctls: Vec<Ctl> = picked
            .iter()
            .zip(&txs)
            .map(|(a, tx)| Ctl {
                node,
                background: a.ticket.priority == Priority::Background,
                tx,
                id: a.ticket.id,
                blocked: Some(blocked),
            })
            .collect();
        let mut lanes: Vec<DynLane> = picked
            .iter_mut()
            .zip(ons.iter_mut())
            .zip(ctls.iter_mut())
            .map(|((a, on), ctl)| DynLane {
                stream: &mut a.stream,
                on: &mut **on,
                ctl,
            })
            .collect();
        // A device fault surfaces as a panic in the compute layer: catch it so the
        // streams get an error and the node reports itself unhealthy.
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| engine.step(&mut lanes)))
                .map_err(|p| {
                    p.downcast_ref::<String>()
                        .cloned()
                        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "panic".into())
                });
        crate::telemetry::queue_depth(|| node.queue.len());
        drop(lanes);
        drop(ctls);
        drop(picked);
        drop(batch);
        // Each batch entry's error, if the step failed it (lanes are in `order`).
        let mut failed: Vec<Option<String>> = vec![None; self.batch.len()];
        match step_verdict(result) {
            Verdict::Ok => {}
            Verdict::Lanes(lanes, e) => {
                eprintln!("tang-llm: a step failed {} stream(s): {e}", lanes.len());
                for j in lanes {
                    if let Some(&i) = order.get(j) {
                        failed[i] = Some(e.clone());
                    }
                }
            }
            Verdict::All(e, fatal) => {
                if fatal {
                    eprintln!("tang-llm: the engine is unhealthy: {e}");
                    self.node.state().unhealthy = Some(e.clone());
                }
                failed.iter_mut().for_each(|f| *f = Some(e.clone()));
            }
        }
        // Retire what's done (and what the step failed), oldest first.
        let mut i = 0;
        while i < self.batch.len() {
            let engine = &self.engine.as_ref().expect("the engine that ran it").1;
            let done = failed[i].is_some() || engine.stream_state(&self.batch[i].stream).0;
            if !done {
                i += 1;
                continue;
            }
            let a = self.batch.remove(i);
            let err = failed.remove(i);
            let engine = &mut self.engine.as_mut().expect("the engine that ran it").1;
            let result = match engine.retire(a.stream) {
                Ok(o) => match err {
                    Some(e) => Err(anyhow::anyhow!("{e}")),
                    None => Ok(o),
                },
                Err(e) => Err(e),
            };
            self.conclude(
                a.ticket,
                a.req,
                a.tx,
                a.model,
                a.before,
                a.request_id,
                a.trace,
                a.kept,
                result,
            );
        }
    }

    /// Reply for a run that's over (or requeue one that gave way).
    #[allow(clippy::too_many_arguments)]
    fn conclude(
        &mut self,
        mut ticket: Ticket,
        req: Box<Request>,
        tx: mpsc::UnboundedSender<Out>,
        model: String,
        before: Option<Before>,
        request_id: Option<String>,
        trace: Trace,
        kept: Option<(String, Vec<u32>)>,
        result: anyhow::Result<Outcome>,
    ) {
        let rid = request_id.as_deref();
        self.node
            .state()
            .running
            .retain(|r| r.ticket.id != ticket.id);
        // Before the reply goes out, so a client that asks `/node` next sees what it left.
        self.refresh_kv();
        match result {
            Ok(Outcome::Yielded {
                prompt_tokens,
                cached_tokens,
                prefilled,
                secs,
            }) => {
                self.with_loaded(|l| l.rates.prefilled(prefilled, secs));
                let before = match before {
                    Some(b) => Before {
                        prefilled: b.prefilled + prefilled,
                        secs: b.secs + secs,
                        ..b
                    },
                    None => Before {
                        cached_tokens,
                        prefilled,
                        secs,
                    },
                };
                ticket.yields += 1;
                ticket.prompt_tokens = Some(prompt_tokens);
                let job = Job::Complete {
                    req,
                    prompt: kept,
                    tx,
                    model,
                    before: Some(before),
                    request_id,
                    trace,
                };
                self.node.queue.push_front(ticket, job);
            }
            Ok(Outcome::Done(finish, mut usage)) => {
                let fresh = usage.prompt_tokens - usage.cached_tokens;
                let secs = if usage.prefill_tok_s > 0.0 {
                    fresh as f64 / usage.prefill_tok_s
                } else {
                    0.0
                };
                self.with_loaded(|l| {
                    l.rates.prefilled(fresh, secs);
                    if !req.prefill_only {
                        l.rates.decoded(usage.completion_tokens, usage.decode_tok_s);
                    }
                });
                // As the client sees it: one request, its prefill spread over its runs.
                if let Some(b) = before {
                    usage.cached_tokens = b.cached_tokens;
                    usage.prefill_tok_s =
                        (usage.prompt_tokens - b.cached_tokens) as f64 / (b.secs + secs).max(1e-9);
                }
                let end = if finish == Finish::Cancelled {
                    End::Cancelled
                } else {
                    End::Ok
                };
                log_usage(rid, &model, &usage, end.as_str());
                trace.finish(&usage, end, ticket.yields, None);
                let engine = &mut self.engine.as_mut().expect("the engine that ran it").1;
                let _ = tx.send(Out::Done(finish, usage.clone()));
                engine.after(&req, &Ok((finish, usage)));
            }
            Err(e) => {
                let msg = format!("{e:#}");
                log_usage(rid, &model, &Usage::default(), "error");
                trace.finish(&Usage::default(), End::Error, ticket.yields, Some(&msg));
                let _ = tx.send(Out::Error(msg));
                self.engine
                    .as_mut()
                    .expect("the engine that ran it")
                    .1
                    .after(&req, &Err(e));
            }
        }
    }

    fn with_loaded(&self, f: impl FnOnce(&mut Loaded)) {
        if let Some(l) = self.node.state().model.as_mut() {
            f(l);
        }
    }

    /// Copy what the engine's KV holds into `/node`'s view.
    fn refresh_kv(&self) {
        let Some((_, e)) = &self.engine else { return };
        let kv = Kv {
            blocks: e.block_hashes(),
            keys: e.cached_keys(),
            store: e.kv_store_dir().map(PathBuf::from),
            pool_bytes: e.kv_bytes(),
            step_cost: e.step_cost(),
        };
        self.with_loaded(|l| l.kv = kv);
    }

    /// Load `spec` (nothing else is loaded).
    fn load(&mut self, spec: &str) -> anyhow::Result<()> {
        self.node.state().loading = Some((spec.to_string(), Instant::now()));
        let t = Instant::now();
        let loaded = (self.load)(spec);
        self.node.state().loading = None;
        let engine = loaded?;
        let path = engine.model_path(spec)?;
        let need = crate::node::need(&path, self.node.dtype).ok();
        let load_secs = t.elapsed().as_secs_f64();
        self.node.state().model = Some(Loaded {
            name: spec.to_string(),
            path,
            ctx: engine.context_window(),
            vision: engine.vision(),
            loaded_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|t| t.as_secs())
                .unwrap_or(0),
            load_secs,
            need,
            rates: Rates::default(),
            prompter: engine.prompter(),
            kv: Kv::default(),
        });
        self.engine = Some((spec.to_string(), engine));
        self.refresh_kv();
        Ok(())
    }

    /// Drop the model, if one is loaded; says what was freed.
    fn unload(&mut self) -> Value {
        let Some((name, e)) = self.engine.take() else {
            return json!({ "unloaded": null });
        };
        let freed = self.footprint(&e);
        e.flush_kv_store();
        drop(e);
        self.node.state().model = None;
        eprintln!("tang-llm: unloaded {name}");
        json!({ "unloaded": name, "freed_bytes": freed })
    }

    /// Device memory the loaded engine holds: its weights (estimated) and KV pool.
    fn footprint(&self, e: &B) -> u64 {
        let weights = self
            .node
            .state()
            .model
            .as_ref()
            .and_then(|l| l.need)
            .map_or(0, |n| n.weights_bytes);
        weights + e.kv_bytes() as u64
    }

    /// Load `spec` in place of the current model, if it fits in free memory plus what the
    /// current one would give back. Never takes memory anything else holds.
    fn swap(&mut self, spec: &str) -> Reply {
        if !B::swappable() {
            return Err((
                StatusCode::BAD_REQUEST,
                "this backend has a fixed model; restart to change it".into(),
            ));
        }
        if self.engine.as_ref().is_some_and(|(n, _)| n == spec) {
            return Ok(json!({ "loaded": spec, "already": true }));
        }
        let path =
            crate::resolve_model(spec).map_err(|e| (StatusCode::NOT_FOUND, format!("{e:#}")))?;
        let need = crate::node::need(&path, self.node.dtype)
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("{spec}: {e:#}")))?;
        let hw = (self.node.hardware)();
        let ours = self.engine.as_ref().map_or(0, |(_, e)| self.footprint(e));
        let room = hw.free_bytes + ours;
        if need.total() > room {
            let gb = |b: u64| b as f64 / 1e9;
            return Err((
                StatusCode::INSUFFICIENT_STORAGE,
                format!(
                    "{spec} needs {:.1} GB ({:.1} GB of weights, {:.1} GB headroom) but only \
                     {:.1} GB is free ({:.1} GB free on the {}{}); loads never take memory \
                     other processes hold",
                    gb(need.total()),
                    gb(need.weights_bytes),
                    gb(need.headroom_bytes),
                    gb(room),
                    gb(hw.free_bytes),
                    hw.kind,
                    if ours > 0 {
                        format!(", {:.1} GB from unloading the current model", gb(ours))
                    } else {
                        String::new()
                    },
                ),
            ));
        }
        let previous = self.engine.as_ref().map(|(n, _)| n.clone());
        self.unload();
        match self.load(spec) {
            Ok(()) => {
                let load_secs = self
                    .node
                    .state()
                    .model
                    .as_ref()
                    .map_or(0.0, |l| l.load_secs);
                eprintln!("tang-llm: loaded {spec} in {load_secs:.1}s");
                Ok(json!({
                    "loaded": spec,
                    "replaced": previous,
                    "load_secs": load_secs,
                    "weights_bytes": need.weights_bytes,
                }))
            }
            Err(e) => {
                let mut msg = format!("loading {spec}: {e:#}");
                if let Some(p) = previous {
                    match self.load(&p) {
                        Ok(()) => msg.push_str(&format!(" ({p} is loaded again)")),
                        Err(e2) => msg.push_str(&format!(" (reloading {p} failed too: {e2:#})")),
                    }
                }
                Err((StatusCode::INTERNAL_SERVER_ERROR, msg))
            }
        }
    }
}

/// Turn away requests without the key (compared in constant time). The image-only server's
/// gate; the chat server's is [`crate::auth::gate`].
pub(crate) async fn require_key(
    State(key): State<std::sync::Arc<String>>,
    req: HttpRequest,
    next: Next,
) -> Response {
    match crate::auth::bearer(req.headers()) {
        Some(given) if crate::auth::same(given.as_bytes(), key.as_bytes()) => next.run(req).await,
        Some(_) => crate::auth::unauthorized("bad key"),
        None => crate::auth::unauthorized("missing key"),
    }
}

async fn models(State(app): State<App>) -> Json<Value> {
    let st = app.node.state();
    let mut data: Vec<Value> = st
        .model
        .iter()
        .map(|m| {
            // `thinking_budget`: requests may cap reasoning tokens (see `engine::ThinkBudget`).
            // `prompt_cache_key`: requests may name their conversation's KV slot (see
            // `crate::slots`). `prefill`: `/v1/prefill` warms a conversation's cache.
            // `priority`: `x-frog-priority` orders requests.
            let mut caps = vec![
                "completion",
                "json_object",
                "json_schema",
                "thinking_budget",
                "prompt_cache_key",
                "prefill",
                "priority",
            ];
            if m.vision {
                caps.push("vision");
            }
            json!({ "id": m.name, "object": "model", "owned_by": "tang", "max_model_len": m.ctx, "capabilities": caps })
        })
        .collect();
    if let Some(image) = &app.node.image {
        data.push(image.json());
    }
    Json(json!({ "object": "list", "data": data }))
}

async fn props(State(app): State<App>) -> Response {
    let ctx = app.node.state().model.as_ref().map(|m| m.ctx);
    match ctx {
        Some(ctx) => Json(json!({ "n_ctx": ctx, "default_generation_settings": { "n_ctx": ctx } }))
            .into_response(),
        None => error(StatusCode::SERVICE_UNAVAILABLE, "no model loaded"),
    }
}

/// `GET /node`: see `docs/node.md` for the shape.
async fn node_info(
    State(app): State<App>,
    caller: Option<axum::Extension<crate::auth::Caller>>,
) -> Json<Value> {
    let node = app.node.clone();
    // `prompt_cache_keys` only for a caller the gate let through (every caller of a node beyond
    // loopback presents a key; on loopback without a key the gate is open by design).
    let keys = caller.is_some();
    let v = tokio::task::spawn_blocking(move || describe(&node, keys))
        .await
        .unwrap_or_else(|e| json!({ "error": e.to_string() }));
    Json(v)
}

fn describe(node: &Node, cache_keys: bool) -> Value {
    let hw = (node.hardware)();
    let hex = |h: &u64| format!("{h:016x}");
    let ms = |t: Instant| t.elapsed().as_millis() as u64;
    let st = node.state();
    let loaded: Vec<Value> = st
        .model
        .iter()
        .map(|m| {
            let mut blocks = m.kv.blocks.clone();
            blocks.sort_unstable();
            let mut disk = m
                .kv
                .store
                .as_deref()
                .map(crate::kvstore::Store::block_hashes)
                .unwrap_or_default();
            disk.sort_unstable();
            let ident = crate::model_id::identify(&m.path, Some(node.dtype));
            json!({
                "id": m.name,
                "path": m.path,
                "weights_hash": ident.weights_hash,
                "quant": ident.quant,
                "context_window": m.ctx,
                "vision": m.vision,
                "loaded_at": m.loaded_at,
                "load_secs": m.load_secs,
                "memory": {
                    "weights_bytes": m.need.map(|n| n.weights_bytes),
                    "kv_bytes": m.kv.pool_bytes,
                },
                "rates": m.rates.json(),
                "step_cost": m.kv.step_cost,
                "kv": {
                    "block_positions": crate::blocks::BLOCK,
                    "memory_blocks": blocks.iter().map(hex).collect::<Vec<_>>(),
                    "disk_blocks": disk.iter().map(hex).collect::<Vec<_>>(),
                    "prompt_cache_keys": cache_keys.then(|| m.kv.keys.iter().map(|(k, n)| json!({ "key": k, "cached_tokens": n })).collect::<Vec<_>>()),
                },
            })
        })
        .collect();
    let loaded_ids: Vec<String> = st.model.iter().map(|m| m.name.clone()).collect();
    let loading = st
        .loading
        .as_ref()
        .map(|(m, t)| json!({ "id": m, "elapsed_ms": ms(*t) }));
    let running: Vec<Value> = st
        .running
        .iter()
        .map(|r| {
            json!({
                "id": r.ticket.id,
                "priority": r.ticket.priority.as_str(),
                "prompt_tokens": r.progress.prompt_tokens,
                "cached_tokens": r.progress.cached_tokens,
                "prefilled_tokens": r.progress.prefilled,
                "generated_tokens": r.progress.generated,
                "yields": r.ticket.yields,
                "queued_ms": ms(r.ticket.since),
                "running_ms": ms(r.started),
            })
        })
        .collect();
    drop(st);
    let waiting: Vec<Value> = node
        .queue
        .waiting()
        .iter()
        .map(|t| {
            json!({
                "id": t.id,
                "priority": t.priority.as_str(),
                "prompt_tokens": t.prompt_tokens,
                "yields": t.yields,
                "queued_ms": ms(t.since),
            })
        })
        .collect();
    let on_disk: Vec<Value> = crate::node::models_on_disk()
        .into_iter()
        .map(|m| {
            let need = crate::node::need(&m.path, node.dtype).ok();
            let ident = crate::model_id::identify(&m.path, None);
            json!({
                "id": m.id,
                "path": m.path,
                "weights_hash": ident.weights_hash,
                "quant": ident.quant,
                "size_bytes": m.size_bytes,
                "load_bytes": need.map(|n| n.total()),
                "loaded": loaded_ids.contains(&m.id),
            })
        })
        .collect();
    let (pulling, last_pull) = node.pull.json();
    json!({
        "schema": crate::node::SCHEMA,
        "node_id": node.id,
        "identity": {
            "fingerprint": node.identity.fingerprint(),
            "public_key": base64::Engine::encode(&base64::engine::general_purpose::STANDARD, node.identity.public_key()),
        },
        "name": node.name,
        "version": env!("CARGO_PKG_VERSION"),
        "hardware": hw.json(),
        "models": { "loaded": loaded, "loading": loading, "on_disk": on_disk, "pulling": pulling, "last_pull": last_pull },
        "queue": { "running": running, "waiting": waiting },
        "image_model": node.image.as_ref().map(|image|json!({
            "id":crate::image_server::MODEL_ID, "path":image.path, "resident":image.resident.load(std::sync::atomic::Ordering::Relaxed),
        })),
    })
}

/// `POST /models/load {"model": "<repo id or directory>"}`: swap the model in.
async fn load_model(State(app): State<App>, Json(body): Json<Value>) -> Response {
    let Some(spec) = body["model"].as_str().map(String::from) else {
        return error(StatusCode::BAD_REQUEST, "model must be a string");
    };
    if spec == crate::image_server::MODEL_ID {
        return error(
            StatusCode::BAD_REQUEST,
            "image models load on first generation, not through /models/load",
        );
    }
    if app
        .node
        .state()
        .model
        .as_ref()
        .is_some_and(|m| m.name == spec)
    {
        return Json(json!({ "loaded": spec, "already": true })).into_response();
    }
    control(&app, |reply| Job::Load { spec, reply }).await
}

/// `POST /models/unload`: free the model's memory.
async fn unload_model(State(app): State<App>) -> Response {
    control(&app, |reply| Job::Unload { reply }).await
}

/// `POST /models/pull {"model", "revision"?, "files": [{"path", "url", "sha256", "size"}]}`:
/// download a model into the Hugging Face cache (see [`crate::pull`]). Answers 202 once it has
/// started; `/node`'s `models.pulling` follows it and `models.last_pull` says how it ended.
async fn pull_model(State(app): State<App>, Json(body): Json<Value>) -> Response {
    let spec: crate::pull::Spec = match serde_json::from_value(body) {
        Ok(s) => s,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let model = spec.model.clone();
    let hw = (app.node.hardware)();
    let dtype = app.node.dtype;
    // Fits the device on its own (the node runs one model; loading swaps the old one out).
    let fits = move |config: Option<&[u8]>, weights: u64| -> Result<(), String> {
        let Some(config) = config else { return Ok(()) };
        let need = crate::node::need_from(config, weights, dtype)
            .map_err(|e| format!("config.json: {e}"))?;
        if need.total() > hw.total_bytes {
            return Err(format!(
                "{model} needs {} bytes of memory; this node has {}",
                need.total(),
                hw.total_bytes
            ));
        }
        Ok(())
    };
    let model = spec.model.clone();
    match app.node.pull.start(spec, fits).await {
        Ok(total) => (
            StatusCode::ACCEPTED,
            Json(json!({ "pulling": model, "total_bytes": total })),
        )
            .into_response(),
        Err(crate::pull::Refused::Bad(e)) => error(StatusCode::BAD_REQUEST, e),
        Err(crate::pull::Refused::Busy(e)) => error(StatusCode::CONFLICT, e),
        Err(crate::pull::Refused::TooBig(e)) => error(StatusCode::INSUFFICIENT_STORAGE, e),
    }
}

/// `DELETE /models/pull`: stop the running pull (what it downloaded stays; pulling the same
/// model again resumes).
async fn cancel_pull(State(app): State<App>) -> Response {
    match app.node.pull.cancel() {
        Some(model) => Json(json!({ "cancelled": model })).into_response(),
        None => error(StatusCode::NOT_FOUND, "nothing is being pulled"),
    }
}

/// `POST /models/delete {"model": "<repo id>"}`: remove a model from the Hugging Face cache.
/// Refused for the loaded (or loading) model and one being pulled.
async fn delete_model(State(app): State<App>, Json(body): Json<Value>) -> Response {
    let Some(model) = body["model"].as_str().map(String::from) else {
        return error(StatusCode::BAD_REQUEST, "model must be a string");
    };
    let check = crate::pull::Spec {
        model: model.clone(),
        revision: None,
        files: vec![crate::pull::File {
            path: "x".into(),
            url: "https://x".into(),
            sha256: "0".repeat(64),
            size: 0,
        }],
    };
    if let Err(e) = check.check() {
        return error(StatusCode::BAD_REQUEST, e);
    }
    {
        let st = app.node.state();
        let busy = st.model.as_ref().map(|m| m.name.as_str()) == Some(model.as_str())
            || st.loading.as_ref().map(|l| l.0.as_str()) == Some(model.as_str());
        if busy {
            return error(
                StatusCode::CONFLICT,
                format!("{model} is loaded; unload it first"),
            );
        }
    }
    if app.node.pull.pulling().as_deref() == Some(model.as_str()) {
        return error(
            StatusCode::CONFLICT,
            format!("{model} is being pulled; cancel it first"),
        );
    }
    match tokio::task::spawn_blocking(move || crate::pull::delete(&model).map(|d| (model, d))).await
    {
        Ok(Ok((model, true))) => Json(json!({ "deleted": model })).into_response(),
        Ok(Ok((model, false))) => error(StatusCode::NOT_FOUND, format!("{model} isn't on disk")),
        Ok(Err(e)) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// Run a load or unload on the worker, when no request is running or waiting.
async fn control(app: &App, job: impl FnOnce(oneshot::Sender<Reply>) -> Job) -> Response {
    let busy = app.node.busy();
    if busy > 0 {
        return error(
            StatusCode::CONFLICT,
            format!("{busy} requests running or waiting; retry when the node is idle"),
        );
    }
    let (tx, rx) = oneshot::channel();
    app.node.queue.push(Priority::Interactive, None, job(tx));
    match rx.await {
        Ok(Ok(v)) => Json(v).into_response(),
        Ok(Err((status, msg))) => error(status, msg),
        Err(_) => error(StatusCode::SERVICE_UNAVAILABLE, "model worker stopped"),
    }
}

/// One line per request, for reconciling with a gateway's records by request id:
/// `tang-llm: usage request_id=… model=… prompt_tokens=… cached_tokens=… completion_tokens=… outcome=ok|cancelled|error`.
/// Tokens are what ran: a cancelled request counts its prompt and what it generated.
fn log_usage(request_id: Option<&str>, model: &str, u: &Usage, outcome: &str) {
    eprintln!("{}", usage_line(request_id, model, u, outcome));
}

fn usage_line(request_id: Option<&str>, model: &str, u: &Usage, outcome: &str) -> String {
    format!(
        "tang-llm: usage request_id={} model={} prompt_tokens={} cached_tokens={} completion_tokens={} outcome={outcome}",
        request_id.unwrap_or("-"),
        model.replace(char::is_whitespace, "_"),
        u.prompt_tokens,
        u.cached_tokens,
        u.completion_tokens,
    )
}

/// `x-request-id`, kept to printable ASCII without spaces (so the log line stays one token
/// per field) and at most 128 bytes.
fn request_id(headers: &HeaderMap) -> Option<String> {
    let id: String = headers
        .get("x-request-id")?
        .to_str()
        .ok()?
        .chars()
        .filter(|c| c.is_ascii_graphic())
        .take(128)
        .collect();
    (!id.is_empty()).then_some(id)
}

/// Queue a request for the loaded model, at its `x-frog-priority`, with its prompt counted.
#[allow(clippy::result_large_err)]
async fn submit(
    app: &App,
    headers: &HeaderMap,
    req: Request,
) -> Result<(String, mpsc::UnboundedReceiver<Out>), Response> {
    let priority = Priority::parse(headers.get("x-frog-priority").and_then(|v| v.to_str().ok()))
        .map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
    let (model, prompter) = {
        let st = app.node.state();
        let m = st
            .model
            .as_ref()
            .ok_or_else(|| error(StatusCode::SERVICE_UNAVAILABLE, "no model loaded"))?;
        (m.name.clone(), m.prompter.clone())
    };
    // Off the model's thread: the queue knows the prompt's length, and the worker needn't
    // tokenize it again.
    let (req, prompt) = tokio::task::spawn_blocking(move || {
        let p = prompter.map(|p| p.prompt(&req)).transpose();
        (req, p)
    })
    .await
    .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let prompt = prompt.map_err(|e| error(StatusCode::BAD_REQUEST, format!("{e:#}")))?;
    let (tx, rx) = mpsc::unbounded_channel();
    let n = prompt.as_ref().map(|p| p.1.len());
    let request_id = request_id(headers);
    let trace = Trace::incoming(headers, priority.as_str(), &model, request_id.as_deref());
    let job = Job::Complete {
        req: Box::new(req),
        prompt,
        tx,
        model: model.clone(),
        before: None,
        request_id,
        trace,
    };
    app.node.queue.push(priority, n, job);
    Ok((model, rx))
}

fn error(status: StatusCode, msg: impl Into<String>) -> Response {
    (
        status,
        Json(json!({ "error": { "message": msg.into(), "type": "invalid_request_error" } })),
    )
        .into_response()
}

/// Images a request may carry, and the most bytes one may be.
const MAX_IMAGES: usize = 16;
const MAX_IMAGE_BYTES: usize = 20 << 20;

/// Replace `https://` image URLs in a chat body with data URLs of what they point to. Only
/// public addresses are fetched (the host is resolved once and pinned, redirects are refused),
/// so a request can't make the server reach its own network. `TANG_IMAGE_FETCH=0` turns
/// fetching off (then only data URLs are accepted).
async fn fetch_images(body: &mut Value) -> Result<(), String> {
    use base64::Engine as _;
    // The image URL slots, in order: `image_url.url` or a bare `image_url` string.
    fn slots(body: &mut Value) -> Vec<&mut Value> {
        let mut out = Vec::new();
        for m in body["messages"].as_array_mut().into_iter().flatten() {
            for p in m["content"].as_array_mut().into_iter().flatten() {
                let slot = if p["image_url"].is_object() {
                    &mut p["image_url"]["url"]
                } else {
                    &mut p["image_url"]
                };
                if slot.as_str().is_some_and(|u| !u.starts_with("data:")) {
                    out.push(slot);
                }
            }
        }
        out
    }
    let urls: Vec<String> = slots(body)
        .iter()
        .map(|s| s.as_str().unwrap_or_default().to_string())
        .collect();
    if urls.is_empty() {
        return Ok(());
    }
    if std::env::var("TANG_IMAGE_FETCH").as_deref() == Ok("0") {
        return Err("images must be data URLs (data:image/...;base64,...)".into());
    }
    if urls.len() > MAX_IMAGES {
        return Err(format!("at most {MAX_IMAGES} images per request"));
    }
    let mut data = Vec::with_capacity(urls.len());
    for url in &urls {
        let bytes = fetch_image(url).await?;
        data.push(base64::engine::general_purpose::STANDARD.encode(&bytes));
    }
    for (slot, d) in slots(body).into_iter().zip(data) {
        *slot = json!(format!("data:image;base64,{d}"));
    }
    Ok(())
}

async fn fetch_image(url: &str) -> Result<Vec<u8>, String> {
    let bad = |e: &dyn std::fmt::Display| format!("image {url}: {e}");
    let parsed = reqwest::Url::parse(url).map_err(|e| bad(&e))?;
    if parsed.scheme() != "https" {
        return Err(bad(&"only https and data URLs are accepted"));
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| bad(&"no host"))?
        .to_string();
    let port = parsed.port_or_known_default().unwrap_or(443);
    let addr = tokio::net::lookup_host((host.as_str(), port))
        .await
        .map_err(|e| bad(&e))?
        .find(|a| public_ip(a.ip()))
        .ok_or_else(|| bad(&"not a public address"))?;
    let client = reqwest::Client::builder()
        .resolve(&host, addr)
        .redirect(reqwest::redirect::Policy::none())
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| bad(&e))?;
    let mut resp = client.get(parsed).send().await.map_err(|e| bad(&e))?;
    if !resp.status().is_success() {
        return Err(bad(&resp.status()));
    }
    let mut out = Vec::new();
    while let Some(chunk) = resp.chunk().await.map_err(|e| bad(&e))? {
        out.extend_from_slice(&chunk);
        if out.len() > MAX_IMAGE_BYTES {
            return Err(bad(&format!("over {} MiB", MAX_IMAGE_BYTES >> 20)));
        }
    }
    Ok(out)
}

/// Whether an address is on the public internet (not loopback, private, link-local, CGNAT,
/// unique-local, documentation, multicast or unspecified).
fn public_ip(ip: std::net::IpAddr) -> bool {
    use std::net::IpAddr;
    match ip {
        IpAddr::V4(a) => {
            let o = a.octets();
            !(a.is_loopback()
                || a.is_private()
                || a.is_link_local()
                || a.is_unspecified()
                || a.is_broadcast()
                || a.is_multicast()
                || a.is_documentation()
                || o[0] == 0
                || (o[0] == 100 && (o[1] & 0xc0) == 64)
                || (o[0] == 192 && o[1] == 0 && o[2] == 0)
                || o[0] >= 240)
        }
        IpAddr::V6(a) => {
            if let Some(v4) = a.to_ipv4_mapped() {
                return public_ip(IpAddr::V4(v4));
            }
            let s = a.segments();
            !(a.is_loopback()
                || a.is_unspecified()
                || a.is_multicast()
                || (s[0] & 0xfe00) == 0xfc00
                || (s[0] & 0xffc0) == 0xfe80
                || s[0] == 0x2001 && s[1] == 0xdb8)
        }
    }
}

/// Make content a string, which templates assume: flatten OpenAI content arrays
/// (`[{type: text, text}]`), and turn a null/missing content (tool-call-only assistant turns)
/// into "" (templates do things like `'</think>' in message.content`). Images (`image_url`
/// parts with data URLs) are returned in order, each leaving an image marker in the text.
fn normalize_messages(messages: &Value) -> Result<(Value, Vec<Vec<u8>>), String> {
    use base64::Engine as _;
    let mut out = messages.clone();
    let mut images = Vec::new();
    for m in out.as_array_mut().into_iter().flatten() {
        if let Some(parts) = m["content"].as_array() {
            let mut text: Vec<String> = Vec::new();
            for p in parts {
                if let Some(t) = p["text"].as_str() {
                    text.push(t.to_string());
                } else if let Some(url) = p["image_url"]["url"].as_str().or(p["image_url"].as_str())
                {
                    let data = url
                        .split_once(";base64,")
                        .map(|(_, d)| d)
                        .ok_or("images must be data URLs (data:image/...;base64,...)")?;
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(data.trim())
                        .map_err(|e| format!("image data: {e}"))?;
                    images.push(bytes);
                    text.push(crate::engine::IMAGE_MARKER.to_string());
                }
            }
            // Parts join with newlines, but an image sits right against its neighbours (as
            // the reference chat templates render image parts).
            let marker = crate::engine::IMAGE_MARKER;
            let mut joined = String::new();
            for (i, t) in text.iter().enumerate() {
                if i > 0 && t != marker && text[i - 1] != marker {
                    joined.push('\n');
                }
                joined.push_str(t);
            }
            m["content"] = json!(joined);
        } else if !m["content"].is_string() && m.is_object() {
            m["content"] = json!("");
        }
    }
    Ok((out, images))
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::{Backend, Outcome, Request};

    struct FixedBackend {
        failed: bool,
    }
    impl Backend for FixedBackend {
        fn context_window(&self) -> usize {
            1024
        }
        fn complete(
            &mut self,
            _: &Request,
            on: &mut dyn FnMut(Piece) -> bool,
        ) -> anyhow::Result<(Finish, Usage)> {
            if self.failed {
                anyhow::bail!("generation failed")
            }
            on(Piece::Text("ok".into()));
            Ok((
                Finish::Stop,
                Usage {
                    prompt_tokens: 2,
                    cached_tokens: 0,
                    completion_tokens: 1,
                    reasoning_tokens: 0,
                    prefill_tok_s: 10.0,
                    decode_tok_s: 20.0,
                    ..Usage::default()
                },
            ))
        }
        fn model_path(&self, _: &str) -> anyhow::Result<std::path::PathBuf> {
            Ok("/weights/model.gguf".into())
        }
    }

    #[test]
    fn sampling_presence_survives_server_integration() {
        let defaults = super::parse(&json!({"messages": []})).unwrap();
        assert!(!defaults.temperature_set && !defaults.top_k_set && !defaults.top_p_set);
        assert_eq!(defaults.presence_penalty, None);
        let explicit = super::parse(&json!({"messages": [], "temperature": 0.0, "top_k": 0, "top_p": 1.0, "presence_penalty": 1.5})).unwrap();
        assert!(explicit.temperature_set && explicit.top_k_set && explicit.top_p_set);
        assert_eq!(explicit.presence_penalty, Some(1.5));
    }

    #[test]
    fn fixed_backend_completion_and_failure_are_forwarded() {
        let req = super::parse(&json!({"messages": []})).unwrap();
        let mut e = FixedBackend { failed: false };
        let mut pieces = Vec::new();
        let outcome = e
            .complete_with(
                &req,
                None,
                &mut |p| {
                    pieces.push(p);
                    true
                },
                &mut (),
            )
            .unwrap();
        assert!(matches!(outcome, Outcome::Done(Finish::Stop, u) if u.completion_tokens == 1));
        assert!(matches!(&pieces[0], Piece::Text(t) if t == "ok"));
        assert!(e.prompter().is_none());
        e.failed = true;
        assert!(e.complete_with(&req, None, &mut |_| true, &mut ()).is_err());
    }

    #[test]
    fn fixed_backend_rejects_swaps_before_resolving_or_loading() {
        let node = std::sync::Arc::new(super::Node {
            pull: Default::default(),
            id: "test".into(),
            queue: crate::queue::Queue::new(),
            state: std::sync::Mutex::new(super::NodeState::default()),
            hardware: std::sync::Arc::new(crate::node::Hardware::default),
            dtype: crate::model::Dtype::F32,
            resources: Default::default(),
            image: None,
            identity: std::sync::Arc::new(crate::auth::Identity::load_or_create(None).unwrap()),
            name: "test".into(),
        });
        let mut worker = super::Worker {
            node,
            engine: Some(("flash".into(), FixedBackend { failed: false })),
            load: |_: &str| -> anyhow::Result<FixedBackend> { panic!("must not load") },
            batch: Vec::new(),
        };
        let err = worker.swap("does-not-exist").unwrap_err();
        assert_eq!(err.0, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(worker.engine.as_ref().unwrap().0, "flash");
    }

    #[test]
    fn only_public_addresses_are_fetched() {
        let p = |s: &str| super::public_ip(s.parse().unwrap());
        assert!(p("1.1.1.1") && p("2606:4700::1111"));
        for a in [
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",
            "100.64.0.1",
            "0.0.0.0",
            "::1",
            "fd00::1",
            "fe80::1",
            "::ffff:127.0.0.1",
            "::ffff:10.0.0.1",
        ] {
            assert!(!p(a), "{a}");
        }
    }

    #[test]
    fn images_sit_against_their_text() {
        let m = serde_json::json!([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "what is this?"},
            {"type": "text", "text": "and this"},
        ]}]);
        let (n, imgs) = super::normalize_messages(&m).unwrap();
        let mk = crate::engine::IMAGE_MARKER;
        assert_eq!(n[0]["content"], format!("{mk}what is this?\nand this"));
        assert_eq!(imgs.len(), 1);
    }

    #[test]
    fn content_is_always_a_string() {
        let m = serde_json::json!([
            {"role": "assistant", "content": null, "tool_calls": []},
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            {"role": "assistant"},
        ]);
        let (n, _) = super::normalize_messages(&m).unwrap();
        assert_eq!(n[0]["content"], "");
        assert_eq!(n[1]["content"], "a\nb");
        assert_eq!(n[2]["content"], "");
    }

    #[test]
    fn thinking_budget_from_the_field_the_kwargs_or_the_effort() {
        use super::thinking_budget as tb;
        assert_eq!(tb(&serde_json::json!({})), None);
        assert_eq!(
            tb(&serde_json::json!({ "thinking_budget": 512 })),
            Some(512)
        );
        assert_eq!(tb(&serde_json::json!({ "thinking_budget": -1 })), None);
        let kw = serde_json::json!({ "chat_template_kwargs": { "thinking_budget": 256 } });
        assert_eq!(tb(&kw), Some(256));
        assert_eq!(
            tb(&serde_json::json!({ "reasoning_effort": "low" })),
            Some(1024)
        );
        assert_eq!(tb(&serde_json::json!({ "reasoning_effort": "high" })), None);
        let both = serde_json::json!({ "thinking_budget": 100, "reasoning_effort": "low" });
        assert_eq!(tb(&both), Some(100));
    }

    use super::{collect_response, stream_response, Out};
    use crate::chat::Piece;
    use crate::engine::{Finish, Usage};
    use crate::telemetry::Trace;
    #[cfg(feature = "otel")]
    use axum::http::HeaderMap;
    use axum::response::IntoResponse;
    use serde_json::{json, Value};
    use tokio::sync::mpsc;

    fn two_calls() -> mpsc::UnboundedReceiver<Out> {
        let (tx, rx) = mpsc::unbounded_channel();
        for name in ["a", "b"] {
            let call = Piece::ToolCall {
                name: name.into(),
                arguments: json!({}),
            };
            tx.send(Out::Piece(call)).unwrap();
        }
        let usage = Usage {
            prompt_tokens: 1,
            cached_tokens: 0,
            completion_tokens: 1,
            reasoning_tokens: 0,
            prefill_tok_s: 0.0,
            decode_tok_s: 0.0,
            ..Usage::default()
        };
        tx.send(Out::Done(Finish::ToolCalls, usage)).unwrap();
        rx
    }

    async fn body(r: axum::response::Response) -> String {
        let b = axum::body::to_bytes(r.into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8(b.to_vec()).unwrap()
    }

    async fn ids_non_streaming() -> Vec<String> {
        let r = collect_response("m".into(), "id".into(), 0, two_calls()).await;
        let v: Value = serde_json::from_str(&body(r).await).unwrap();
        v["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["id"].as_str().unwrap().to_string())
            .collect()
    }

    async fn ids_streaming() -> Vec<String> {
        let r = stream_response("m".into(), "id".into(), 0, false, two_calls()).into_response();
        body(r)
            .await
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter_map(|d| serde_json::from_str::<Value>(d).ok())
            .filter_map(|v| {
                v["choices"][0]["delta"]["tool_calls"][0]["id"]
                    .as_str()
                    .map(String::from)
            })
            .collect()
    }

    #[tokio::test]
    async fn tool_call_ids_are_unique_within_and_across_responses() {
        let mut ids = Vec::new();
        for _ in 0..2 {
            ids.extend(ids_non_streaming().await);
            ids.extend(ids_streaming().await);
        }
        assert_eq!(ids.len(), 8);
        for id in &ids {
            assert!(id.starts_with("call_") && id.len() == 29, "{id}");
        }
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len(), "{ids:?}");
    }

    #[tokio::test]
    async fn include_usage_ends_with_an_openai_usage_chunk() {
        let r = stream_response("m".into(), "id".into(), 7, true, two_calls()).into_response();
        let text = body(r).await;
        let data: Vec<&str> = text
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .collect();
        assert_eq!(data.last(), Some(&"[DONE]"));
        let evs: Vec<Value> = data
            .iter()
            .filter_map(|d| serde_json::from_str(d).ok())
            .collect();
        let (usage, rest) = evs.split_last().unwrap();
        assert_eq!(usage["object"], "chat.completion.chunk");
        assert_eq!(usage["id"], "id");
        assert_eq!(usage["model"], "m");
        assert_eq!(usage["created"], 7);
        assert_eq!(usage["choices"], json!([]));
        assert_eq!(usage["usage"]["prompt_tokens"], 1);
        assert_eq!(usage["usage"]["completion_tokens"], 1);
        assert_eq!(usage["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
        // The finish chunk comes before it, without usage; no other chunk carries usage.
        let finish = rest.last().unwrap();
        assert_eq!(finish["choices"][0]["finish_reason"], "tool_calls");
        assert!(rest.iter().all(|e| e.get("usage").is_none()));
    }

    #[test]
    fn usage_line_and_request_id() {
        let mut h = axum::http::HeaderMap::new();
        assert_eq!(super::request_id(&h), None);
        h.insert("x-request-id", "req 42\tab".parse().unwrap());
        assert_eq!(super::request_id(&h).as_deref(), Some("req42ab"));
        let u = Usage {
            prompt_tokens: 10,
            cached_tokens: 4,
            completion_tokens: 3,
            ..Usage::default()
        };
        assert_eq!(
            super::usage_line(Some("req42"), "qwen 3", &u, "cancelled"),
            "tang-llm: usage request_id=req42 model=qwen_3 prompt_tokens=10 cached_tokens=4 completion_tokens=3 outcome=cancelled"
        );
        assert!(super::usage_line(None, "m", &u, "ok").contains("request_id=- "));
    }

    /// Generates silently (as while a tool call is buffered, or during a long prefill) until
    /// its Control says stop, counting steps.
    struct Endless {
        steps: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }
    const ENDLESS: usize = 100_000;
    impl Backend for Endless {
        fn context_window(&self) -> usize {
            1024
        }
        fn complete(
            &mut self,
            _: &Request,
            _: &mut dyn FnMut(Piece) -> bool,
        ) -> anyhow::Result<(Finish, Usage)> {
            unreachable!()
        }
        fn complete_with(
            &mut self,
            _: &Request,
            _: Option<(String, Vec<u32>)>,
            _: &mut dyn FnMut(Piece) -> bool,
            ctl: &mut dyn super::Control,
        ) -> anyhow::Result<Outcome> {
            use std::sync::atomic::Ordering;
            self.steps.store(0, Ordering::SeqCst);
            let mut finish = Finish::Length;
            for _ in 0..ENDLESS {
                if ctl.cancelled() {
                    finish = Finish::Cancelled;
                    break;
                }
                self.steps.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            Ok(Outcome::Done(
                finish,
                Usage {
                    prompt_tokens: 5,
                    completion_tokens: self.steps.load(Ordering::SeqCst),
                    ..Usage::default()
                },
            ))
        }
        fn model_path(&self, _: &str) -> anyhow::Result<std::path::PathBuf> {
            Ok("/m".into())
        }
    }

    /// Spans and step metrics through the worker, to in-memory exporters.
    #[cfg(feature = "otel")]
    #[test]
    fn otel_spans_continue_the_incoming_trace() {
        use opentelemetry::trace::{SpanKind, Status, TraceId};
        use opentelemetry::Value as V;
        use opentelemetry_sdk::error::OTelSdkResult;
        use opentelemetry_sdk::metrics::data::ResourceMetrics;
        use opentelemetry_sdk::metrics::exporter::PushMetricExporter;
        use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider, Temporality};
        use opentelemetry_sdk::trace::{SdkTracerProvider, SpanData, SpanExporter};
        use std::sync::{Arc, Mutex};

        #[derive(Debug, Clone, Default)]
        struct Spans(Arc<Mutex<Vec<SpanData>>>);
        impl SpanExporter for Spans {
            async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
                self.0.lock().unwrap().extend(batch);
                Ok(())
            }
        }
        #[derive(Clone, Default)]
        struct Metrics(Arc<Mutex<Vec<String>>>);
        impl PushMetricExporter for Metrics {
            async fn export(&self, m: &ResourceMetrics) -> OTelSdkResult {
                let mut names = self.0.lock().unwrap();
                for sm in m.scope_metrics() {
                    names.extend(sm.metrics().map(|m| m.name().to_string()));
                }
                Ok(())
            }
            fn force_flush(&self) -> OTelSdkResult {
                Ok(())
            }
            fn shutdown_with_timeout(&self, _: std::time::Duration) -> OTelSdkResult {
                Ok(())
            }
            fn temporality(&self) -> Temporality {
                Temporality::Cumulative
            }
        }

        struct Traced;
        impl Backend for Traced {
            fn context_window(&self) -> usize {
                1024
            }
            fn complete(
                &mut self,
                _: &Request,
                on: &mut dyn FnMut(Piece) -> bool,
            ) -> anyhow::Result<(Finish, Usage)> {
                // Longer than its decode, so the prefill span has room before it.
                std::thread::sleep(std::time::Duration::from_millis(80));
                on(Piece::Text("ok".into()));
                Ok((
                    Finish::Stop,
                    Usage {
                        prompt_tokens: 32,
                        cached_tokens: 8,
                        completion_tokens: 6,
                        decode_tok_s: 120.0,
                        prefill_tok_s: 1000.0,
                        draft_tokens: 4,
                        accepted_tokens: 2,
                        prefill_chunks: 3,
                        decode_steps: 4,
                        shared_streams: 6,
                        ..Usage::default()
                    },
                ))
            }
            fn model_path(&self, _: &str) -> anyhow::Result<std::path::PathBuf> {
                Ok("/weights/model.gguf".into())
            }
        }

        let spans = Spans::default();
        let metrics = Metrics::default();
        let tp = SdkTracerProvider::builder()
            .with_simple_exporter(spans.clone())
            .build();
        let mp = SdkMeterProvider::builder()
            .with_reader(PeriodicReader::builder(metrics.clone()).build())
            .build();
        crate::telemetry::install(&tp, &mp);

        let node = std::sync::Arc::new(super::Node {
            pull: Default::default(),
            id: "test".into(),
            queue: crate::queue::Queue::new(),
            state: std::sync::Mutex::new(super::NodeState::default()),
            hardware: std::sync::Arc::new(crate::node::Hardware::default),
            dtype: crate::model::Dtype::F32,
            resources: Default::default(),
            image: None,
        });
        let mut worker = super::Worker {
            node,
            engine: Some(("m".into(), Traced)),
            load: |_: &str| -> anyhow::Result<Traced> { unreachable!() },
            batch: Vec::new(),
        };
        let run = |worker: &mut super::Worker<Traced, _>, headers: &HeaderMap, model: &str| {
            let rid = super::request_id(headers);
            let trace = Trace::incoming(headers, "background", model, rid.as_deref());
            let (tx, mut rx) = mpsc::unbounded_channel();
            let ticket = crate::queue::Ticket {
                id: 1,
                priority: crate::queue::Priority::Background,
                prompt_tokens: Some(32),
                since: std::time::Instant::now(),
                yields: 0,
            };
            let req = Box::new(super::parse(&json!({"messages": []})).unwrap());
            worker.complete(ticket, req, None, tx, model.into(), None, rid, trace);
            while rx.try_recv().is_ok() {}
        };
        let mut headers = HeaderMap::new();
        headers.insert(
            "traceparent",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
                .parse()
                .unwrap(),
        );
        headers.insert("x-request-id", "req-7".parse().unwrap());
        run(&mut worker, &headers, "m");
        // A model that's gone: an error, a fresh trace.
        run(&mut worker, &HeaderMap::new(), "other");
        crate::telemetry::step(&crate::engine::StepStats {
            decode_rows: 3,
            decode_streams: 3,
            prefill_tokens: 512,
            prefill_chunks: 1,
            secs: 0.02,
        });
        crate::telemetry::queue_depth(|| 2);

        let got = spans.0.lock().unwrap().clone();
        let attr = |s: &SpanData, k: &str| {
            s.attributes
                .iter()
                .find(|kv| kv.key.as_str() == k)
                .map(|kv| kv.value.clone())
        };
        let named = |n: &str| -> Vec<&SpanData> { got.iter().filter(|s| s.name == n).collect() };
        let trace_id = TraceId::from_hex("4bf92f3577b34da6a3ce929d0e0e4736").unwrap();
        let roots = named("chat.completions");
        assert_eq!(roots.len(), 2);
        let root = roots
            .iter()
            .find(|s| s.span_context.trace_id() == trace_id)
            .expect("the incoming trace continued");
        assert_eq!(root.parent_span_id.to_string(), "00f067aa0ba902b7");
        assert_eq!(root.span_kind, SpanKind::Server);
        assert_eq!(attr(root, "request.id"), Some(V::from("req-7")));
        assert_eq!(attr(root, "gen_ai.request.model"), Some(V::from("m")));
        assert_eq!(attr(root, "outcome"), Some(V::from("ok")));
        let child = |n: &str| {
            *named(n)
                .iter()
                .find(|s| s.span_context.trace_id() == trace_id)
                .unwrap_or_else(|| panic!("no {n} span"))
        };
        for n in ["queued", "prefill", "decode", "finish"] {
            let s = child(n);
            assert_eq!(s.parent_span_id, root.span_context.span_id(), "{n}");
            assert_eq!(attr(s, "request.id"), Some(V::from("req-7")), "{n}");
            assert!(s.start_time <= s.end_time, "{n}");
        }
        let q = child("queued");
        assert_eq!(attr(q, "priority"), Some(V::from("background")));
        assert!(attr(q, "queue.wait_ms").is_some());
        let p = child("prefill");
        assert_eq!(attr(p, "prompt_tokens"), Some(V::I64(32)));
        assert_eq!(attr(p, "cached_tokens"), Some(V::I64(8)));
        assert_eq!(attr(p, "new_tokens"), Some(V::I64(24)));
        assert_eq!(attr(p, "chunks"), Some(V::I64(3)));
        let d = child("decode");
        assert_eq!(attr(d, "completion_tokens"), Some(V::I64(6)));
        assert_eq!(attr(d, "accepted_draft_tokens"), Some(V::I64(2)));
        assert_eq!(attr(d, "tokens_per_verify_window"), Some(V::F64(1.5)));
        assert_eq!(attr(d, "shared_streams_avg"), Some(V::F64(1.5)));
        // 6 tokens at 120/s: the decode span is the last 50 ms.
        let ms = d.end_time.duration_since(d.start_time).unwrap().as_millis();
        assert!((49..=51).contains(&ms), "{ms}");
        assert_eq!(p.end_time, d.start_time);
        assert_eq!(attr(child("finish"), "outcome"), Some(V::from("ok")));

        let failed = roots
            .iter()
            .find(|s| s.span_context.trace_id() != trace_id)
            .unwrap();
        assert_eq!(failed.parent_span_id.to_string(), "0000000000000000");
        assert_eq!(attr(failed, "outcome"), Some(V::from("error")));
        assert!(matches!(failed.status, Status::Error { .. }));
        assert!(attr(failed, "request.id").is_none());

        mp.force_flush().unwrap();
        let names = metrics.0.lock().unwrap().clone();
        for m in [
            "tang_llm.step.duration",
            "tang_llm.step.rows",
            "tang_llm.prefill.chunk_tokens",
            "tang_llm.prefix.hit_ratio",
            "tang_llm.queue.depth",
            "tang_llm.stream.tokens_per_second",
        ] {
            assert!(names.iter().any(|n| n == m), "{m} missing from {names:?}");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_dropped_stream_stops_generation() {
        use std::sync::atomic::Ordering;
        let steps = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let node = std::sync::Arc::new(super::Node {
            pull: Default::default(),
            id: "test".into(),
            queue: crate::queue::Queue::new(),
            state: std::sync::Mutex::new(super::NodeState::default()),
            hardware: std::sync::Arc::new(crate::node::Hardware::default),
            dtype: crate::model::Dtype::F32,
            resources: Default::default(),
            image: None,
            identity: std::sync::Arc::new(crate::auth::Identity::load_or_create(None).unwrap()),
            name: "test".into(),
        });
        let endless = Endless {
            steps: steps.clone(),
        };
        // Built on its thread (a worker holds streams that stay on it).
        let make = move || super::Worker {
            node,
            engine: Some(("m".into(), endless)),
            load: |_: &str| -> anyhow::Result<Endless> { unreachable!() },
            batch: Vec::new(),
        };
        let req = Box::new(super::parse(&json!({"messages": []})).unwrap());
        let (tx, rx) = mpsc::unbounded_channel();
        let ticket = crate::queue::Ticket {
            id: 1,
            priority: crate::queue::Priority::Interactive,
            prompt_tokens: None,
            since: std::time::Instant::now(),
            yields: 0,
        };
        let run = std::thread::spawn(move || {
            let mut worker = make();
            let t = std::time::Instant::now();
            worker.complete(
                ticket,
                req,
                None,
                tx,
                "m".into(),
                None,
                Some("rid".into()),
                Trace::none(),
            );
            let took = t.elapsed();
            drop(worker);
            (took, ())
        });
        // The client reads nothing and goes away while the run is under way.
        let sse = stream_response("m".into(), "id".into(), 0, true, rx).into_response();
        while steps.load(Ordering::SeqCst) < 20 {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        drop(sse);
        let took = tokio::task::spawn_blocking(move || run.join().unwrap().0)
            .await
            .unwrap();
        let n = steps.load(Ordering::SeqCst);
        assert!(n < 2_000, "ran {n} steps after the client left");
        assert!(took < std::time::Duration::from_secs(10), "{took:?}");
    }
}

/// A failed step ends only the streams it names; a device fault ends all and marks the
/// node unhealthy.
#[cfg(test)]
mod step_failures {
    use super::{Backend, DynLane, Inflight, Out, Outcome, Request, StepFailure, Verdict};
    use crate::chat::Piece;
    use crate::engine::{Finish, Usage};
    use crate::telemetry::Trace;
    use serde_json::json;
    use tokio::sync::mpsc;

    /// Streams that never finish on their own; `fail` picks what a step does.
    struct Lanes {
        fail: fn(&[usize]) -> anyhow::Result<()>,
    }
    impl Backend for Lanes {
        fn context_window(&self) -> usize {
            1024
        }
        fn complete(
            &mut self,
            _: &Request,
            _: &mut dyn FnMut(Piece) -> bool,
        ) -> anyhow::Result<(Finish, Usage)> {
            unreachable!()
        }
        fn max_batch(&self) -> usize {
            4
        }
        fn can_admit(&self, _: Option<&str>) -> bool {
            true
        }
        fn admit(
            &mut self,
            req: &Request,
            _: Option<(String, Vec<u32>)>,
        ) -> anyhow::Result<Inflight> {
            Ok(Box::new(req.max_tokens.unwrap_or(0)))
        }
        fn step(&mut self, lanes: &mut [DynLane<'_>]) -> anyhow::Result<()> {
            let ids: Vec<usize> = lanes
                .iter()
                .map(|l| *l.stream.downcast_ref::<usize>().unwrap())
                .collect();
            (self.fail)(&ids)
        }
        fn retire(&mut self, _: Inflight) -> anyhow::Result<Outcome> {
            Ok(Outcome::Done(Finish::Length, Usage::default()))
        }
        fn stream_state(&self, _: &Inflight) -> (bool, bool) {
            (false, false)
        }
    }

    type W = super::Worker<Lanes, fn(&str) -> anyhow::Result<Lanes>>;

    fn worker(fail: fn(&[usize]) -> anyhow::Result<()>) -> W {
        let node = std::sync::Arc::new(super::Node {
            pull: Default::default(),
            id: "test".into(),
            queue: crate::queue::Queue::new(),
            state: std::sync::Mutex::new(super::NodeState::default()),
            hardware: std::sync::Arc::new(crate::node::Hardware::default),
            dtype: crate::model::Dtype::F32,
            resources: Default::default(),
            image: None,
            identity: std::sync::Arc::new(crate::auth::Identity::load_or_create(None).unwrap()),
            name: "test".into(),
        });
        fn load(_: &str) -> anyhow::Result<Lanes> {
            unreachable!()
        }
        super::Worker {
            node,
            engine: Some(("m".into(), Lanes { fail })),
            load,
            batch: Vec::new(),
        }
    }

    /// Admit stream `id` (its `max_tokens`); returns its reply channel.
    fn start(w: &mut W, id: usize) -> mpsc::UnboundedReceiver<Out> {
        let req = Box::new(super::parse(&json!({"messages": [], "max_tokens": id})).unwrap());
        let (tx, rx) = mpsc::unbounded_channel();
        let ticket = crate::queue::Ticket {
            id: id as u64,
            priority: crate::queue::Priority::Interactive,
            prompt_tokens: None,
            since: std::time::Instant::now(),
            yields: 0,
        };
        w.complete(ticket, req, None, tx, "m".into(), None, None, Trace::none());
        rx
    }

    fn errored(rx: &mut mpsc::UnboundedReceiver<Out>) -> bool {
        let mut e = false;
        while let Ok(o) = rx.try_recv() {
            e |= matches!(o, Out::Error(_));
        }
        e
    }

    /// Stream 2's slot overflowed: only it ends.
    fn overflow(ids: &[usize]) -> anyhow::Result<()> {
        match ids.iter().position(|&i| i == 2) {
            Some(j) => Err(StepFailure::lanes(
                vec![j],
                anyhow::anyhow!("slot 2 is full"),
            )),
            None => Ok(()),
        }
    }

    #[test]
    fn an_attributable_failure_ends_only_its_streams() {
        let mut w = worker(overflow);
        let mut rx: Vec<_> = (1..=3).map(|i| start(&mut w, i)).collect();
        assert_eq!(w.batch.len(), 3);
        w.step_batch();
        assert_eq!(w.batch.len(), 2);
        let errs: Vec<bool> = rx.iter_mut().map(errored).collect();
        assert_eq!(errs, [false, true, false]);
        assert!(w.node.state().unhealthy.is_none());
        // The others go on stepping.
        w.step_batch();
        assert_eq!(w.batch.len(), 2);
    }

    fn fault(_: &[usize]) -> anyhow::Result<()> {
        Err(anyhow::anyhow!(
            "DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, \"an illegal memory access\")"
        ))
    }

    fn panics(_: &[usize]) -> anyhow::Result<()> {
        panic!("called `Result::unwrap()` on an `Err` value: CUDA_ERROR_ILLEGAL_ADDRESS")
    }

    #[test]
    fn a_device_fault_ends_everything_and_marks_the_node_unhealthy() {
        for f in [fault as fn(&[usize]) -> anyhow::Result<()>, panics] {
            let mut w = worker(f);
            let mut rx: Vec<_> = (1..=3).map(|i| start(&mut w, i)).collect();
            w.step_batch();
            assert!(w.batch.is_empty());
            assert!(rx.iter_mut().all(errored));
            assert!(w.node.state().unhealthy.is_some());
            // A new request fails instead of joining the broken engine.
            let mut late = start(&mut w, 4);
            assert!(w.batch.is_empty());
            assert!(errored(&mut late));
        }
    }

    #[test]
    fn verdicts() {
        assert_eq!(super::step_verdict(Ok(Ok(()))), Verdict::Ok);
        let v = super::step_verdict(Ok(Err(StepFailure::lanes(vec![1], anyhow::anyhow!("x")))));
        assert_eq!(v, Verdict::Lanes(vec![1], "x".into()));
        // Attributed, but a device fault: everything, and unhealthy.
        let v = super::step_verdict(Ok(Err(StepFailure::lanes(
            vec![1],
            anyhow::anyhow!("CUDA_ERROR_ILLEGAL_ADDRESS"),
        ))));
        assert!(matches!(v, Verdict::All(_, true)));
        // Unattributed, not a device fault: everything, still healthy.
        let v = super::step_verdict(Ok(Err(anyhow::anyhow!("parse error"))));
        assert_eq!(v, Verdict::All("parse error".into(), false));
        assert!(matches!(
            super::step_verdict(Err("boom".into())),
            Verdict::All(_, true)
        ));
    }
}

#[cfg(test)]
mod context_limits {
    use super::{Backend, DynLane, Inflight, Out, Outcome, Request};
    use crate::chat::Piece;
    use crate::engine::{Finish, Usage};
    use crate::telemetry::Trace;
    use axum::http::StatusCode;
    use serde_json::json;
    use tokio::sync::mpsc;

    const CTX: usize = 1000;

    /// A batching engine like `flash serve --batch`: a request's prompt is its first message's
    /// length in characters, admission goes through `batch::token_limit`, and every step
    /// generates one token for each stream until its limit.
    struct Window;
    /// (prompt, generated, limit)
    type S = (usize, usize, usize);
    impl Backend for Window {
        fn context_window(&self) -> usize {
            CTX
        }
        fn complete(
            &mut self,
            _: &Request,
            _: &mut dyn FnMut(Piece) -> bool,
        ) -> anyhow::Result<(Finish, Usage)> {
            unreachable!()
        }
        fn max_batch(&self) -> usize {
            4
        }
        fn can_admit(&self, _: Option<&str>) -> bool {
            true
        }
        fn admit(
            &mut self,
            req: &Request,
            _: Option<(String, Vec<u32>)>,
        ) -> anyhow::Result<Inflight> {
            let prompt = req.messages[0]["content"].as_str().unwrap().len();
            let limit = crate::batch::token_limit(prompt, CTX, req.max_tokens)
                .map_err(|e| anyhow::anyhow!(e))?;
            Ok(Box::new((prompt, 0usize, limit)))
        }
        fn step(&mut self, lanes: &mut [DynLane<'_>]) -> anyhow::Result<()> {
            for l in lanes {
                let s = l.stream.downcast_mut::<S>().unwrap();
                assert!(s.0 + s.1 < CTX, "a stream ran past the context");
                s.1 += 1;
                (l.on)(Piece::Text("x".into()));
            }
            Ok(())
        }
        fn retire(&mut self, s: Inflight) -> anyhow::Result<Outcome> {
            let s = s.downcast::<S>().unwrap();
            let usage = Usage {
                prompt_tokens: s.0,
                completion_tokens: s.1,
                ..Default::default()
            };
            Ok(Outcome::Done(Finish::Length, usage))
        }
        fn stream_state(&self, s: &Inflight) -> (bool, bool) {
            let s = s.downcast_ref::<S>().unwrap();
            (s.1 >= s.2, false)
        }
    }

    type W = super::Worker<Window, fn(&str) -> anyhow::Result<Window>>;

    fn worker() -> W {
        let node = std::sync::Arc::new(super::Node {
            pull: Default::default(),
            id: "test".into(),
            queue: crate::queue::Queue::new(),
            state: std::sync::Mutex::new(super::NodeState::default()),
            hardware: std::sync::Arc::new(crate::node::Hardware::default),
            dtype: crate::model::Dtype::F32,
            resources: Default::default(),
            image: None,
            identity: std::sync::Arc::new(crate::auth::Identity::load_or_create(None).unwrap()),
            name: "test".into(),
        });
        fn load(_: &str) -> anyhow::Result<Window> {
            unreachable!()
        }
        super::Worker {
            node,
            engine: Some(("m".into(), Window)),
            load,
            batch: Vec::new(),
        }
    }

    fn start(w: &mut W, id: u64, prompt: usize, max: usize) -> mpsc::UnboundedReceiver<Out> {
        let body = json!({
            "messages": [{"role": "user", "content": "p".repeat(prompt)}],
            "max_tokens": max,
        });
        let req = Box::new(super::parse(&body).unwrap());
        let (tx, rx) = mpsc::unbounded_channel();
        let ticket = crate::queue::Ticket {
            id,
            priority: crate::queue::Priority::Interactive,
            prompt_tokens: None,
            since: std::time::Instant::now(),
            yields: 0,
        };
        w.complete(ticket, req, None, tx, "m".into(), None, None, Trace::none());
        rx
    }

    /// Drain a reply: (pieces, finish and usage, or the error).
    fn reply(
        rx: &mut mpsc::UnboundedReceiver<Out>,
    ) -> (usize, Option<Result<(Finish, Usage), String>>) {
        let (mut n, mut end) = (0, None);
        while let Ok(o) = rx.try_recv() {
            match o {
                Out::Piece(_) => n += 1,
                Out::Done(f, u) => end = Some(Ok((f, u))),
                Out::Error(e) => end = Some(Err(e)),
            }
        }
        (n, end)
    }

    #[test]
    fn a_prompt_past_the_context_is_refused_at_admission() {
        let mut w = worker();
        let mut rx = start(&mut w, 1, CTX + 5, 10);
        assert!(w.batch.is_empty(), "it never joined the batch");
        let (n, end) = reply(&mut rx);
        assert_eq!(n, 0);
        let e = end.unwrap().unwrap_err();
        assert!(
            e.contains("context_length_exceeded") && e.contains("1005"),
            "{e}"
        );
    }

    #[test]
    fn a_stream_reaching_the_context_stops_with_length_and_others_go_on() {
        let mut w = worker();
        // 990 + max 50 > 1000: it gets 9 tokens. The other wants 30.
        let mut long = start(&mut w, 1, 990, 50);
        let mut other = start(&mut w, 2, 10, 30);
        assert_eq!(w.batch.len(), 2);
        for _ in 0..40 {
            w.step_batch();
        }
        assert!(w.batch.is_empty());
        let (n, end) = reply(&mut long);
        let (f, u) = end.unwrap().unwrap();
        assert_eq!((n, f, u.completion_tokens), (9, Finish::Length, 9));
        assert_eq!(super::finish_reason(f), "length");
        let (n, end) = reply(&mut other);
        let (_, u) = end.unwrap().unwrap();
        assert_eq!((n, u.completion_tokens), (30, 30));
    }

    #[tokio::test]
    async fn an_admission_error_answers_400() {
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(Out::Error(
            crate::batch::token_limit(5000, CTX, None).unwrap_err(),
        ))
        .unwrap();
        let r = super::collect_response("m".into(), "id".into(), 0, rx).await;
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    }
}

/// A chat completion body as an engine request.
pub fn parse(body: &Value) -> Result<Request, String> {
    let messages = body
        .get("messages")
        .filter(|m| m.is_array())
        .ok_or("messages must be an array")?;
    let response_schema = crate::structured::response_schema(body).map_err(|e| e.to_string())?;
    let d = Sampling::default();
    let f = |k: &str| body[k].as_f64();
    let (messages, images) = normalize_messages(messages)?;
    Ok(Request {
        messages,
        images,
        tools: body.get("tools").cloned().filter(|t| !t.is_null()),
        response_schema,
        think: body["chat_template_kwargs"]["enable_thinking"]
            .as_bool()
            .or(body["think"].as_bool())
            .or((body["reasoning_effort"] == "none").then_some(false)),
        thinking_budget: thinking_budget(body),
        temperature_set: f("temperature").is_some(),
        top_k_set: body["top_k"].as_u64().is_some(),
        top_p_set: f("top_p").is_some(),
        presence_penalty: f("presence_penalty").map(|v| v as f32),
        sampling: Sampling {
            temperature: f("temperature").map(|v| v as f32).unwrap_or(d.temperature),
            top_p: f("top_p").map(|v| v as f32).unwrap_or(d.top_p),
            top_k: body["top_k"]
                .as_u64()
                .map(|v| v as usize)
                .unwrap_or(d.top_k),
            seed: body["seed"].as_u64().unwrap_or_else(|| {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|t| t.as_nanos() as u64)
                    .unwrap_or(1)
            }),
        },
        max_tokens: body["max_completion_tokens"]
            .as_u64()
            .or(body["max_tokens"].as_u64())
            .map(|n| n as usize),
        stop: match &body["stop"] {
            Value::String(s) => vec![s.clone()],
            Value::Array(a) => a
                .iter()
                .filter_map(|s| s.as_str().map(String::from))
                .collect(),
            _ => Vec::new(),
        },
        cache_key: body["prompt_cache_key"].as_str().map(String::from),
        tenant: body["user"]
            .as_str()
            .or(body["safety_identifier"].as_str())
            .map(String::from),
        prefill_only: false,
    })
}

/// Reasoning token caps for OpenAI's `reasoning_effort` (`high` is unlimited).
const EFFORT: [(&str, usize); 3] = [("minimal", 0), ("low", 1024), ("medium", 4096)];

/// The cap on reasoning tokens: `thinking_budget` (top level, or in `chat_template_kwargs` as
/// some servers take it), else from `reasoning_effort`. Negative or absent is unlimited.
fn thinking_budget(body: &Value) -> Option<usize> {
    let n = |v: &Value| v.as_u64().map(|n| n as usize);
    n(&body["thinking_budget"])
        .or_else(|| n(&body["chat_template_kwargs"]["thinking_budget"]))
        .or_else(|| {
            let effort = body["reasoning_effort"].as_str()?;
            EFFORT.iter().find(|(e, _)| *e == effort).map(|(_, n)| *n)
        })
}

fn finish_reason(f: Finish) -> &'static str {
    match f {
        Finish::Stop | Finish::Cancelled => "stop",
        Finish::Length => "length",
        Finish::ToolCalls => "tool_calls",
    }
}

fn usage_json(u: &Usage) -> Value {
    json!({
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.prompt_tokens + u.completion_tokens,
        "prompt_tokens_details": { "cached_tokens": u.cached_tokens },
        "completion_tokens_details": { "reasoning_tokens": u.reasoning_tokens },
        // llama.cpp's names for speculative decoding stats.
        "timings": {
            "prompt_per_second": u.prefill_tok_s,
            "predicted_per_second": u.decode_tok_s,
            "draft_n": u.draft_tokens,
            "draft_n_accepted": u.accepted_tokens,
            // decode forwards (verify windows with speculation): tokens per forward is
            // completion_tokens / decode_steps
            "decode_steps": u.decode_steps,
        },
    })
}

/// Prefill a chat completion body without generating: `{usage}` once the cache holds it.
async fn prefill(
    State(app): State<App>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> Response {
    if let Err(e) = fetch_images(&mut body).await {
        return error(StatusCode::BAD_REQUEST, e);
    }
    let mut req = match parse(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    req.prefill_only = true;
    let mut rx = match submit(&app, &headers, req).await {
        Ok((_, rx)) => rx,
        Err(r) => return r,
    };
    while let Some(out) = rx.recv().await {
        match out {
            Out::Piece(_) => {}
            Out::Done(_, u) => {
                return Json(json!({ "object": "prefill", "usage": usage_json(&u) }))
                    .into_response()
            }
            Out::Error(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
        }
    }
    error(StatusCode::SERVICE_UNAVAILABLE, "model worker stopped")
}

async fn chat(State(app): State<App>, headers: HeaderMap, Json(mut body): Json<Value>) -> Response {
    if let Err(e) = fetch_images(&mut body).await {
        return error(StatusCode::BAD_REQUEST, e);
    }
    let req = match parse(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let stream = body["stream"].as_bool().unwrap_or(false);
    let include_usage = body["stream_options"]["include_usage"]
        .as_bool()
        .unwrap_or(false);
    let (model, rx) = match submit(&app, &headers, req).await {
        Ok(x) => x,
        Err(r) => return r,
    };
    let id = format!(
        "chatcmpl-{:x}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|t| t.as_nanos())
            .unwrap_or(0)
    );
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|t| t.as_secs())
        .unwrap_or(0);
    if stream {
        stream_response(model, id, created, include_usage, rx).into_response()
    } else {
        collect_response(model, id, created, rx).await
    }
}

fn stream_response(
    model: String,
    id: String,
    created: u64,
    include_usage: bool,
    mut rx: mpsc::UnboundedReceiver<Out>,
) -> Sse<UnboundedReceiverStream<Result<Event, Infallible>>> {
    let chunk = move |delta: Value, finish: Option<&str>| {
        json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
        })
    };
    let mut calls = 0usize;
    let mut first = true;
    let (sse_tx, sse_rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        loop {
            // Watch for the client going away even while nothing is generated (a long
            // prefill, buffered tool-call text): dropping rx stops the run.
            let out = tokio::select! {
                out = rx.recv() => match out {
                    Some(out) => out,
                    None => return,
                },
                _ = sse_tx.closed() => return,
            };
            let mut evs: Vec<Value> = Vec::new();
            let role = if first {
                json!("assistant")
            } else {
                Value::Null
            };
            first = false;
            match out {
                Out::Piece(Piece::Text(t)) => {
                    evs.push(chunk(json!({ "role": role, "content": t }), None))
                }
                Out::Piece(Piece::Reasoning(t)) => {
                    evs.push(chunk(json!({ "role": role, "reasoning_content": t }), None))
                }
                Out::Piece(Piece::ToolCall { name, arguments }) => {
                    evs.push(chunk(
                        json!({ "role": role, "tool_calls": [{
                            "index": calls, "id": tool_call_id(), "type": "function",
                            "function": { "name": name, "arguments": arguments.to_string() },
                        }]}),
                        None,
                    ));
                    calls += 1;
                }
                Out::Done(finish, usage) => {
                    let mut last = chunk(json!({}), Some(finish_reason(finish)));
                    if include_usage {
                        evs.push(last);
                        let mut u = chunk(json!({}), None);
                        u["choices"] = json!([]);
                        u["usage"] = usage_json(&usage);
                        last = u;
                    } else {
                        last["usage"] = usage_json(&usage);
                    }
                    evs.push(last);
                }
                Out::Error(e) => evs.push(json!({ "error": { "message": e } })),
            }
            let done = evs.iter().any(|e| {
                e["choices"][0]["finish_reason"].is_string()
                    || e["usage"].is_object()
                    || e["error"].is_object()
            });
            for v in evs {
                if sse_tx
                    .send(Ok(Event::default().data(v.to_string())))
                    .is_err()
                {
                    return; // client went away; dropping rx cancels generation
                }
            }
            if done {
                let _ = sse_tx.send(Ok(Event::default().data("[DONE]")));
                return;
            }
        }
    });
    Sse::new(UnboundedReceiverStream::new(sse_rx))
}

/// A tool-call id unique within the response, across requests and across restarts: clients
/// (kiln among them) key tool results by id, so `call_0` on every call collides. A per-process
/// random prefix plus a process-wide counter, as `call_<24 hex>`.
fn tool_call_id() -> String {
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;
    static PREFIX: OnceLock<u64> = OnceLock::new();
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let prefix = *PREFIX.get_or_init(|| {
        // RandomState is seeded from the OS RNG; mix in time and pid for good measure.
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u128(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|t| t.as_nanos())
                .unwrap_or(0),
        );
        h.write_u32(std::process::id());
        h.finish()
    });
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    format!("call_{prefix:016x}{n:08x}")
}

async fn collect_response(
    model: String,
    id: String,
    created: u64,
    mut rx: mpsc::UnboundedReceiver<Out>,
) -> Response {
    let (mut text, mut reasoning, mut calls) = (String::new(), String::new(), Vec::new());
    while let Some(out) = rx.recv().await {
        match out {
            Out::Piece(Piece::Text(t)) => text.push_str(&t),
            Out::Piece(Piece::Reasoning(t)) => reasoning.push_str(&t),
            Out::Piece(Piece::ToolCall { name, arguments }) => calls.push(json!({
                "id": tool_call_id(), "type": "function",
                "function": { "name": name, "arguments": arguments.to_string() },
            })),
            Out::Error(e) => return error(StatusCode::BAD_REQUEST, e),
            Out::Done(finish, usage) => {
                let mut message = json!({ "role": "assistant", "content": if text.is_empty() { Value::Null } else { json!(text) } });
                if !reasoning.is_empty() {
                    message["reasoning_content"] = json!(reasoning);
                }
                if !calls.is_empty() {
                    message["tool_calls"] = json!(calls);
                }
                return Json(json!({
                    "id": id, "object": "chat.completion", "created": created, "model": model,
                    "choices": [{ "index": 0, "message": message, "finish_reason": finish_reason(finish) }],
                    "usage": usage_json(&usage),
                }))
                .into_response();
            }
        }
    }
    error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "generation ended unexpectedly",
    )
}

/// Pairing and the gate end to end, over HTTP, with a Rust client doing the client's side.
#[cfg(test)]
mod node_auth {
    use super::{App, Node, NodeState, Queue};
    use crate::auth::{Auth, Identity};
    use crate::pair::{client, Pairing};
    use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
    use serde_json::{json, Value};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    struct Server {
        url: String,
        pairing: Arc<Pairing>,
        http: reqwest::Client,
        dir: std::path::PathBuf,
    }

    impl Drop for Server {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    async fn server(static_key: Option<&str>) -> Server {
        let dir = std::env::temp_dir().join(format!(
            "tang-node-auth-{}",
            crate::auth::hex(&crate::auth::random_bytes::<8>())
        ));
        let identity =
            Arc::new(Identity::load_or_create(Some(&dir.join("node-identity"))).unwrap());
        let node = Arc::new(Node {
            pull: Default::default(),
            id: "0123456789abcdef0123456789abcdef".into(),
            queue: Queue::new(),
            state: Mutex::new(NodeState::default()),
            hardware: Arc::new(crate::node::Hardware::default),
            dtype: crate::model::Dtype::Bf16,
            resources: Arc::default(),
            image: None,
            identity: identity.clone(),
            name: "testnode".into(),
        });
        // Bound to loopback for the test, but gated as a node beyond it.
        let auth = Arc::new(Auth::new(
            true,
            static_key.map(String::from),
            Some(dir.join("node-keys.json")),
        ));
        let pairing = Arc::new(Pairing::new(
            auth.clone(),
            identity,
            node.id.clone(),
            node.name.clone(),
        ));
        let router = super::router(App { node }, auth, pairing.clone(), None);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            axum::serve(
                listener,
                router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
            )
            .await
            .unwrap();
        });
        Server {
            url,
            pairing,
            http: reqwest::Client::new(),
            dir,
        }
    }

    impl Server {
        async fn req(
            &self,
            method: reqwest::Method,
            path: &str,
            key: Option<&str>,
            body: Option<Value>,
        ) -> (u16, Value) {
            let mut r = self.http.request(method, format!("{}{path}", self.url));
            if let Some(k) = key {
                r = r.header("authorization", format!("Bearer {k}"));
            }
            if let Some(b) = body {
                r = r
                    .header("content-type", "application/json")
                    .body(b.to_string());
            }
            let resp = r.send().await.unwrap();
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap();
            (
                status,
                serde_json::from_str(&text).unwrap_or(Value::String(text)),
            )
        }
        async fn get(&self, path: &str, key: Option<&str>) -> (u16, Value) {
            self.req(reqwest::Method::GET, path, key, None).await
        }
        async fn post(&self, path: &str, body: Value) -> (u16, Value) {
            self.req(reqwest::Method::POST, path, None, Some(body))
                .await
        }

        /// `/pair/start` with `code`: the pending client state, the confirm to send, the answer.
        async fn start(
            &self,
            client_id: &str,
            code: &str,
        ) -> Result<(String, client::Pending, [u8; 32], Value), (u16, Value)> {
            let c = client::start(code);
            let (st, v) = self
                .post(
                    "/pair/start",
                    json!({ "client_id": client_id, "client_name": "cam's mac", "spake": B64.encode(&c.msg) }),
                )
                .await;
            if st != 200 {
                return Err((st, v));
            }
            let node_msg = B64.decode(v["spake"].as_str().unwrap()).unwrap();
            let identity = B64.decode(v["identity"].as_str().unwrap()).unwrap();
            let (pending, confirm) = c
                .answer(
                    client_id,
                    &node_msg,
                    &identity,
                    v["node_id"].as_str().unwrap(),
                )
                .unwrap();
            Ok((
                v["pair_id"].as_str().unwrap().to_string(),
                pending,
                confirm,
                v,
            ))
        }

        async fn finish(&self, pair_id: &str, confirm: &[u8]) -> (u16, Value) {
            self.post(
                "/pair/finish",
                json!({ "pair_id": pair_id, "confirm": B64.encode(confirm) }),
            )
            .await
        }

        /// A whole pairing with the right code: the key.
        async fn pair(&self, client_id: &str) -> String {
            let (st, v) = self.post("/pair/open", json!({})).await;
            assert_eq!(st, 200, "{v}");
            let code = v["code"].as_str().unwrap().to_string();
            let (pair_id, pending, confirm, _) = self.start(client_id, &code).await.unwrap();
            let (st, v) = self.finish(&pair_id, &confirm).await;
            assert_eq!(st, 200, "{v}");
            let node_confirm = B64.decode(v["confirm"].as_str().unwrap()).unwrap();
            let key_box = B64.decode(v["key_box"].as_str().unwrap()).unwrap();
            pending.key(&node_confirm, &key_box).unwrap()
        }
    }

    fn wrong(code: &str) -> String {
        format!("{:06}", (code.parse::<u32>().unwrap() + 1) % 1_000_000)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pairing_end_to_end() {
        let s = server(None).await;
        // Unpaired beyond loopback: gated routes say so, /health and /pair/* stay open.
        assert_eq!(
            s.get("/node", None).await,
            (401, json!({"error": "not paired"}))
        );
        assert_eq!(
            s.get("/v1/models", Some("tnk_x")).await,
            (401, json!({"error": "not paired"}))
        );
        assert_eq!(s.get("/health", None).await.0, 200);
        assert_eq!(
            s.start("c1", "000000").await.err().unwrap(),
            (403, json!({"error": "no pairing window"}))
        );

        // A window: the code, its life, the fingerprint; `pair` would show in Bonjour.
        let (st, v) = s.post("/pair/open", json!({})).await;
        assert_eq!(st, 200);
        let code = v["code"].as_str().unwrap().to_string();
        assert_eq!(v["expires_in"], 120);
        assert_eq!(v["fingerprint"], s.pairing.identity.fingerprint());
        assert!(s.pairing.is_open());

        // One exchange at a time.
        let (pid, _, confirm, answer) = s.start("c1", &wrong(&code)).await.unwrap();
        assert_eq!(answer["name"], "testnode");
        assert_eq!(answer["node_id"], "0123456789abcdef0123456789abcdef");
        assert_eq!(
            B64.decode(answer["identity"].as_str().unwrap()).unwrap(),
            s.pairing.identity.public_key()
        );
        assert_eq!(
            s.start("c2", &code).await.err().unwrap(),
            (409, json!({"error": "pairing busy"}))
        );
        // Wrong code: 401, and it counts.
        assert_eq!(
            s.finish(&pid, &confirm).await,
            (401, json!({"error": "wrong code"}))
        );
        for _ in 0..3 {
            let (pid, _, confirm, _) = s.start("c1", &wrong(&code)).await.unwrap();
            assert_eq!(s.finish(&pid, &confirm).await.0, 401);
        }
        // Four down; the fifth closes the window, even for the right code after it.
        let (pid, _, confirm, _) = s.start("c1", &wrong(&code)).await.unwrap();
        assert_eq!(s.finish(&pid, &confirm).await.0, 401);
        assert!(!s.pairing.is_open());
        assert_eq!(s.start("c1", &code).await.err().unwrap().0, 403);
        assert_eq!(
            s.finish(&pid, &confirm).await,
            (410, json!({"error": "pairing expired"}))
        );

        // Expiry: a window that runs out between start and finish.
        s.pairing
            .open_with("424242".into(), Duration::from_millis(300));
        let (pid, _, confirm, _) = s.start("c1", "424242").await.unwrap();
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert_eq!(
            s.finish(&pid, &confirm).await,
            (410, json!({"error": "pairing expired"}))
        );

        // Four wrong, then the right code still pairs; the key works.
        let (_, v) = s.post("/pair/open", json!({})).await;
        let code = v["code"].as_str().unwrap().to_string();
        for _ in 0..4 {
            let (pid, _, confirm, _) = s.start("c1", &wrong(&code)).await.unwrap();
            assert_eq!(s.finish(&pid, &confirm).await.0, 401);
        }
        let (pid, pending, confirm, _) = s.start("c1", &code).await.unwrap();
        let (st, v) = s.finish(&pid, &confirm).await;
        assert_eq!(st, 200, "{v}");
        let key = pending
            .key(
                &B64.decode(v["confirm"].as_str().unwrap()).unwrap(),
                &B64.decode(v["key_box"].as_str().unwrap()).unwrap(),
            )
            .unwrap();
        assert!(key.starts_with("tnk_"));
        // One pairing per window.
        assert!(!s.pairing.is_open());
        // The file has the hash, never the key.
        let stored = std::fs::read_to_string(s.dir.join("node-keys.json")).unwrap();
        assert!(
            !stored.contains(&key) && stored.contains(&crate::auth::sha256_hex(key.as_bytes()))
        );
        assert!(stored.contains("cam's mac"));

        let (st, node) = s.get("/node", Some(&key)).await;
        assert_eq!(st, 200, "{node}");
        assert_eq!(node["schema"], 2);
        assert_eq!(node["name"], "testnode");
        assert_eq!(
            node["identity"]["fingerprint"],
            s.pairing.identity.fingerprint()
        );
        assert_eq!(
            node["identity"]["public_key"],
            B64.encode(s.pairing.identity.public_key())
        );
        assert_eq!(
            s.get("/node", None).await,
            (401, json!({"error": "missing key"}))
        );
        assert_eq!(
            s.get("/node", Some("tnk_nope")).await,
            (401, json!({"error": "bad key"}))
        );

        // A second client; un-pairing needs the client's own key.
        let key2 = s.pair("c2").await;
        assert_eq!(s.get("/v1/models", Some(&key2)).await.0, 200);
        let del = reqwest::Method::DELETE;
        assert_eq!(
            s.req(del.clone(), "/pair/c1", Some(&key2), None).await,
            (401, json!({"error": "bad key"}))
        );
        assert_eq!(
            s.req(del.clone(), "/pair/c1", None, None).await,
            (401, json!({"error": "missing key"}))
        );
        assert_eq!(s.req(del, "/pair/c1", Some(&key), None).await.0, 204);
        assert_eq!(
            s.get("/node", Some(&key)).await,
            (401, json!({"error": "bad key"}))
        );
        assert_eq!(s.get("/node", Some(&key2)).await.0, 200);
        // `tang-llm unpair all` from another process: the running server follows.
        tokio::time::sleep(Duration::from_millis(20)).await;
        crate::auth::remove_keys(&s.dir.join("node-keys.json"), "all").unwrap();
        assert_eq!(
            s.get("/node", Some(&key2)).await,
            (401, json!({"error": "not paired"}))
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn static_key_counts_as_one_more_key() {
        let s = server(Some("s3cret")).await;
        assert_eq!(
            s.get("/node", None).await,
            (401, json!({"error": "missing key"}))
        );
        assert_eq!(
            s.get("/node", Some("nope")).await,
            (401, json!({"error": "bad key"}))
        );
        let (st, node) = s.get("/node", Some("s3cret")).await;
        assert_eq!((st, node["schema"].clone()), (200, json!(2)));
        let key = s.pair("c1").await;
        assert_eq!(s.get("/v1/models", Some(&key)).await.0, 200);
        assert_eq!(s.get("/v1/models", Some("s3cret")).await.0, 200);
    }
}
