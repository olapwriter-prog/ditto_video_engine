# Ditto video engine

An image and video generation engine in Rust for phones running
[agenticOS](https://github.com/olapwriter-prog/agenticOS), built from the
[Rust Inference Engine](https://github.com/olapwriter-prog/Rust_Interference_engine) (RIE).

- It runs **at night**: on the charger, in agenticOS's Deep stage, checkpointing after every
  denoising step, and giving the phone back the moment it is unplugged, warm or needed.
- It makes **mixture-of-experts models out of existing dense ones**, as RIE's `rie moefy`
  does for LLMs, so only a fraction of an image or video model's weights run per step.

**Status: Phase 0 of the plan.** There is a benchmark, not yet a generator: no model is
loaded and no image or video is produced.

```bash
cargo test --workspace
cargo run --release -- bench --experts 8 --used 2     # one block of a Wan-1.3B-class model
cargo run --release -- estimate --gmacs 15            # the plan's table, nothing is run
scripts/build-phone.sh                                # static ARM64 binary for postmarketOS
```

- [docs/PLAN.md](docs/PLAN.md): constraints, what a night can produce, model choices,
  architecture, the expert conversion, agenticOS integration, phases, risks and the
  decisions still open.
- [docs/PROGRESS.md](docs/PROGRESS.md): what has been measured, and on which machine.
