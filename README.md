# Ditto video engine

An image and video generation engine in Rust for phones running
[agenticOS](https://github.com/olapwriter-prog/agenticOS), built from the
[Rust Inference Engine](https://github.com/olapwriter-prog/Rust_Interference_engine) (RIE).

- It runs **at night**: on the charger, in agenticOS's Deep stage, checkpointing after every
  denoising step, and giving the phone back the moment it is unplugged, warm or needed.
- It makes **mixture-of-experts models out of existing dense ones**, as RIE's `rie moefy`
  does for LLMs, so only a fraction of an image or video model's weights run per step.

**Status.** What works today, with tests:

- `ditto look`: what a picture the user took contains, read and measured without a model
  (time, place, camera, colours, sharpness, QR codes), handed to the inference engine as
  tool output or memory.
- `ditto trace`: a screenshot, diagram or logo as SVG, exact to the pixel.
- `ditto bench`, `ditto bench-conv`: cost measurements for the generator and for perception
  models.

What does not exist yet: no model is loaded, so nothing is generated, no text is read and
no object is recognised.

```bash
cargo test --workspace
cargo run --release -- look photo.jpg                 # add --rie http://127.0.0.1:8080 to remember it
cargo run --release -- trace screenshot.png           # writes screenshot.png.svg
cargo run --release -- bench --experts 8 --used 2     # one block of a Wan-1.3B-class model
cargo run --release -- estimate --gmacs 15            # the plan's table, nothing is run
scripts/build-phone.sh                                # static ARM64 binary for postmarketOS
```

- [docs/PLAN.md](docs/PLAN.md): constraints, what a night can produce, model choices,
  architecture, the expert conversion, agenticOS integration, phases, risks and the
  decisions still open.
- [docs/USE_CASES.md](docs/USE_CASES.md): the user's own pictures, screenshots, editing and
  video use cases, what is built, and how results reach the inference engine.
- [docs/PROGRESS.md](docs/PROGRESS.md): what has been measured, and on which machine.
- [docs/DRIVER_ASSIST.md](docs/DRIVER_ASSIST.md): a proposal for reusing the engine's kernels
  in a multi-phone driver warning system (warnings only; India and Europe; Android, iPhone,
  agenticOS). Only a cost benchmark exists: `cargo run --release -- bench-conv`.
