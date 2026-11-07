# tang-all

One dependency for the tang stack. Re-exports `tang` at the root and each
subcrate as a module behind a feature (`la` by default; `sparse`, `ad`,
`expr`, `optim`, `compile`, `tensor`, and the heavier `gpu`, `train`).

```toml
tang-all = { version = "0.1", features = ["optim", "compile"] }
```

```rust
use tang_all::{Vec3, optim::minimize_lbfgs};
```
