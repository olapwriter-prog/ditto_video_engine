# Ditto video engine

An image and video generation engine in Rust for phones running
[agenticOS](https://github.com/olapwriter-prog/agenticOS), built from the
[Rust Inference Engine](https://github.com/olapwriter-prog/Rust_Interference_engine) (RIE).

- It runs **at night**: on the charger, in agenticOS's Deep stage, checkpointing after every
  denoising step, and giving the phone back the moment it is unplugged, warm or needed.
- It makes **mixture-of-experts models out of existing dense ones**, as RIE's `rie moefy`
  does for LLMs, so only a fraction of an image or video model's weights run per step.

**Status: plan only. There is no code yet.**

Read [docs/PLAN.md](docs/PLAN.md): constraints, what a night can produce, model choices,
architecture, the expert conversion, agenticOS integration, phases, risks and the decisions
still open.
