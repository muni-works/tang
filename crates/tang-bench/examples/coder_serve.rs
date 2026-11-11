//! `tang-llm serve` with the derived-step Qwen3-Coder-Next engine available:
//!
//! ```text
//! TANG_CODER_ENGINE=derived coder_serve <coder-next.gguf> [tang-llm serve options: --port, --ctx, --batch, ...]
//! ```
//!
//! Without `TANG_CODER_ENGINE=derived` it serves exactly as `tang-llm serve` (the hand engine).
fn main() -> anyhow::Result<()> {
    tang_bench::coder_serve::register();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let hardware: tang_llm::node::Probe = std::sync::Arc::new(|| {
        let (total, free) = tang_llm::node::host_memory().unwrap_or((0, 0));
        tang_llm::node::Hardware {
            kind: "cuda",
            name: String::new(),
            unified: false,
            total_bytes: total,
            free_bytes: free,
        }
    });
    tang_llm::flash::serve::serve_cli(&args, hardware)
}
