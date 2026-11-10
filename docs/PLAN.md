# Ditto: plan for a night-time image and video generation engine for agenticOS phones

Status: written 2026-11-10. Phase 0 has started; see [PROGRESS.md](PROGRESS.md) for what is measured.

Sources read for this plan:

- `Rust_Interference_engine` (RIE) at `f3be365`: the engine Ditto is built from.
- `agenticOS`, branch `ccr-5b2c3e9c-xxflh1` (the newest; `main` holds only a README): the OS
  Ditto runs on, in particular its Deep stage and overnight training.

Every speed, memory and time figure below is an **estimate from arithmetic, not a
measurement**, unless it is quoted from one of those two repositories. Phase 0 exists to
replace the estimates with numbers from a real phone before anything else is built.

---

## 1. What Ditto is

A Rust engine that turns a text prompt (later also a photo) into an image or a short video,
entirely on the phone, while the phone is on the charger and nobody is using it.

Three things define it:

1. **It is made from RIE.** It reuses RIE's GGUF loader and writer, quantized SIMD kernels,
   thread pool, layer streaming, converters and GPU backend. It does not fork them.
2. **It makes mixture-of-experts models out of existing dense ones.** RIE's `rie moefy`
   turns a dense LLM into a MoE without training. Ditto does the same for existing
   open image and video diffusion models, so that only a fraction of the weights run
   per denoising step. No model is trained from scratch.
3. **It is a night job, not a service.** agenticOS already runs LoRA training "overnight on
   the charger" in its Deep stage, stopping when the phone is unplugged, warm or needed and
   continuing from a checkpoint. Ditto follows that contract exactly. A clip that takes an
   hour is acceptable; one that drains the battery or delays a phone call is not.

### How "like MoE LLM models, from existing models" is read here

This is my reading of the request. Section 12 asks you to confirm it.

| Level | What becomes an "expert" | Where it comes from | Router |
|---|---|---|---|
| A. Neuron experts | Groups of feed-forward neurons inside each transformer block | Clustering an existing dense model's neurons (a port of `rie moefy`) | Ridge-regressed, per token, plus the noise level |
| B. Noise-level experts | The set of neuron groups used in one stretch of the denoising schedule | The same dense model, calibrated separately per noise bucket | A table lookup on the step. No per-token cost |
| C. Model experts | Low-rank deltas of existing fine-tunes and LoRAs of one base model (styles, motion, subjects) | SVD of `fine-tune − base`, or community LoRAs used as they are | Per job, from the prompt embedding |
| D. Skip expert | "Reuse last step's result for this block or token" | No weights: a cache | Change in the block input since the last step |

Level A is the direct analogue of what RIE does today. Level B is where diffusion models
differ from LLMs and is likely to pay most: what a denoiser's neurons do depends strongly on
the noise level, and published models already exploit this by shipping separate high-noise
and low-noise experts. Level C is the literal reading of "from existing models".

---

## 2. Constraints from agenticOS

| Fact (from the agenticOS branch) | Consequence for Ditto |
|---|---|
| Base is postmarketOS: Alpine, musl, OpenRC, no UI. | One static `aarch64-unknown-linux-musl` binary, as RIE's `build-android-static.sh` already builds. No Python, no Android NDK, no JNI. |
| Well-supported phones are older: OnePlus 6/6T and Poco F1 (Snapdragon 845, 6 to 8 GB), Fairphone 4/5, PinePhone Pro (4 GB), PinePhone (2 to 3 GB). | Design for 6 GB RAM and Armv8.2 `SDOT`. i8mm is a bonus. 4 GB phones get images only; the PinePhone is out of scope. |
| GPU drivers are Mesa (freedreno/turnip, panfrost), not vendor blobs. The headless image deliberately ships no GPU stack. | CPU is the baseline and must be sufficient. GPU is opt-in per device after a self-test. |
| Stage machine `Asleep → Active → Deep → WindingDown`; `background_allowed` only in Deep; `WorkQueued` and `QueueEmpty` events already exist. | Ditto jobs are the background queue. An empty queue winds the phone down. |
| `trainer.rs` `Conditions::blocker`: charger, battery floor, temperature below `cool_c`, model not busy, Deep stage. | Ditto uses the same conditions and the same pattern: child process, checkpoint, stop, resume, ledger line. |
| `prio`: `Mode::Training` gives long jobs the fast cores in the model cgroup, reserves one core for calls, drops the boost at `hot_c` (80 °C) until `cool_c` (70 °C). | Ditto never picks cores itself. It runs on the cores its cgroup allows. |
| agentd starts child binaries (`llama-server`, `agentos-llm train`) and adopts their pid into the model cgroup. | Ditto is a command-line job runner. No daemon, no open port. |
| `models` crate: `[[models]]` with `url` + `sha256`, resumable verified downloads. | Model bundles are fetched the same way. Several files per bundle. |
| Executor tiers; High needs the approval key. Everything goes in the hash-chained ledger. | Queuing a creation is Medium. Sending one to anybody stays High (existing tools). Each job start, pause, finish is a ledger line. |
| `photos` crate: "Still photos only. This OS takes pictures; it does not record, stream or play video." | Generated media gets its own store. The photos policy is not loosened. |
| `notify`: quiet hours 22:00 to 07:00, digest afterwards. | "Your video is ready" goes into the morning digest, never a night alert. |
| agenticOS has its own LLM engine (`crates/infer`, agentos-llm), separate from RIE. | Ditto depends on `rie-core` only. It does not need agentos-llm, and agentd does not need RIE. |

---

## 3. What a night can produce (estimates)

Model for the estimate, video: a Wan-2.1-1.3B-class diffusion transformer (30 blocks, width
1536, feed-forward 8960, 8× spatial and 4× temporal VAE, 2×2 patches).

```
tokens N       = (1 + (frames−1)/4) × (height/16) × (width/16)
MACs per step ≈ N × 30 × (41.7 M linear + 2·N·1536 attention + ~1 M cross-attention)
seconds/step   = MACs / throughput
```

Throughput assumption: **15 G multiply-adds per second** on four big cores with `SDOT`
(Snapdragon 845 class), from RIE's batched int8 path. This is the first number Phase 0 must
measure; everything scales linearly with it.

| Output | Tokens | Per step | 6-step distilled clip | 30 steps with guidance (60 passes) |
|---|---|---|---|---|
| 320×192, 17 frames (1 s) | 1,200 | ~2 min | ~11 min | ~1.9 h |
| 480×272, 33 frames (2 s) | 4,590 | ~9 min | ~52 min | ~8.7 h (two nights) |
| 480×272, 81 frames (5 s) | 10,710 | ~27 min | ~2.7 h | no |
| 832×480, 81 frames (5 s) | 32,760 | ~2.6 h | ~16 h | no |

Images, PixArt-class 0.6 B transformer at 512×512 (1,024 tokens): ~40 s per step, so about
3 minutes for a 4-step distilled model and about 27 minutes for 20 steps with guidance.
Text encoding and VAE decoding add roughly a minute each.

What follows from the table:

- **Few-step (distilled) sampling is mandatory for video.** Without it one 2-second clip
  does not fit in a night.
- A 6 to 7 hour charge with thermal duty-cycling gives perhaps 4 to 5 hours of compute:
  **several 2-second clips, or one 5-second clip, or dozens of images per night.**
- Feed-forward layers are ~48 % of the work at 4,590 tokens. Running 2 of 8 experts
  (level A/B) removes about a third of the step. Past ~10,000 tokens attention dominates,
  so there the gain must come from fewer tokens (level D), not fewer neurons.
- Full 480p at 5 seconds needs the GPU, a newer phone, or generating small and upscaling
  (section 10, Phase 8).

---

## 4. Which existing models

Licences and exact sizes must be checked in Phase 0 against the model cards. They are listed
from memory here and decide what may ship in a default config.

| Role | Candidate | Why | To verify |
|---|---|---|---|
| Video, first target | **Wan 2.1 T2V-1.3B** | Small transformer, permissive licence as I recall it, 16 fps, community GGUF conversions and step-distilled variants exist | Licence; a few-step variant for the 1.3B; its text encoder (UMT5-XXL, ~5 to 6 B parameters) under layer streaming |
| Video, second | **LTX-Video (2B)** | Much stronger latent compression, so far fewer tokens per second of video. Possibly the better phone model | Licence terms; quality at 4-bit; T5-XXL encoder |
| Image, first target | **PixArt-α/Σ class (0.6 B)** | Plain transformer blocks, the closest to code RIE already has | Few-step variant; T5-XXL encoder |
| Image, alternative | **Sana 0.6B** | 32× autoencoder means very few tokens, and its text encoder is Gemma 2 2B, an architecture **RIE already runs** | Licence (may be non-commercial); its convolutional feed-forward needs a different expert split |
| Tiny decoders | TAESD / TAEHV class | Decoding in seconds instead of minutes; previews | Coverage for the chosen VAEs; licence |
| Later | SD 1.5 + AnimateDiff, SDXL | Huge library of fine-tunes and LoRAs (level C) | UNets are convolution-heavy: a second kernel family. Not before Phase 8 |

Out of scope: 10 B+ models (Flux, HunyuanVideo, Wan 14B, Qwen-Image). They do not fit a
6 GB phone's night even streamed.

Recommendation: **build the image path on a PixArt-class model first, then Wan 2.1 1.3B for
video**, and benchmark LTX-Video in Phase 0 as the possible replacement. Transformer-only
denoisers are chosen on purpose: they reuse RIE's kernels and its MoE machinery.

---

## 5. Architecture

```
agentd (agenticOS)                         ditto (this repository)
 intents / MCP: "make a video of ..."       ┌────────────────────────────────────────────┐
   └─ tool make_video (Medium) ─► queue     │ ditto run JOBDIR      one job, resumable   │
 studio tick (Deep, charger, cool):         │  1 encode   text encoder, streamed, once   │
   spawn `ditto run`, prio.adopt(pid) ────► │  2 denoise  DiT / MoE-DiT, step by step    │
   SIGTERM / cgroup freeze on call or chat  │  3 decode   VAE, tiled and chunked         │
   read status.json, ledger, digest   ◄──── │  4 export   PNG / animated WebP / video    │
                                            └──────┬─────────────────────────────────────┘
                                                   │ rie-core (git dependency, pinned)
                                 GGUF · quant · kernels · threadpool · cpu · turbo · convert
```

### 5.1 Crates

| Crate | Contents |
|---|---|
| `ditto-core` | Tensor ops RIE lacks (full attention, conv2d/conv3d, group/layer norm, 2-D and 3-D position encodings), model definitions, samplers, pipeline stages, checkpointing |
| `ditto-convert` | `safetensors → GGUF`, `quantize`, `moefy`, `lora-pack`, calibration capture |
| `ditto-cli` | `ditto run`, `ditto bench`, `ditto convert`, `ditto moefy`, `ditto verify`, `ditto device` |
| `ditto-gpu` | Later: WGSL kernels on top of `rie-gpu` |
| `tools/reference` | Python scripts (development machines only) that dump reference activations and outputs |

### 5.2 What comes from RIE and what is new

| RIE piece | Use in Ditto | Change needed in RIE |
|---|---|---|
| `gguf.rs` + `GgufWriter` | Every model file, with a `ditto.*` metadata namespace | None |
| `quant/`, `tensor::Weight` | 4-bit and 8-bit weights for all linear layers | None |
| `kernels/` + `matmul.rs` (`QuantActs`, `matmul_multi`) | A denoising step is one big batch of N tokens: exactly RIE's batched prefill path (a weight block unpacked once for 4 tokens, 2×2 i8mm tiles) | Raise or remove batch-size assumptions; expose bias add |
| `threadpool.rs`, `cpu.rs` | Unchanged | **Honour the cgroup cpuset** (`sched_getaffinity`) instead of choosing cores, since agentd's `prio` owns that |
| Layer streaming (`set_streaming`) | Text encoder larger than free RAM; denoiser blocks on 4 GB phones | Generalise from `CpuBackend` to any list of weights |
| `convert.rs`: `quantize_model`, `moefy`, `balanced_kmeans`, `solve_spd` | Basis of `ditto quantize` and `ditto moefy` | Make clustering and the router fit public and independent of the LLM `Model` type |
| `model::MoeFfn` (`unit_weights`: with all experts on, output equals the dense block) | Same contract for diffusion blocks | Support the plain `down(act(up·x + b))` MLP with biases, not only the gated form |
| Gemma 2 forward pass, tokenizers | Text encoder for Sana-class models | A way to read **hidden states of all tokens at a chosen layer** (today only the last row is exposed, for the adapters) |
| `rl.rs` low-rank correction experts | Pattern for applying LoRA experts as a side path | None |
| `turbo.rs` (TurboQuant) | Optional: 4-bit cache of text keys/values and of skipped-block residuals | None |
| `rie-gpu` | Matmul and norm shaders | Share shader modules as a library |
| **New in Ditto** | Bidirectional attention over N tokens in chunks; blocked f32/f16 GEMM; conv2d/conv3d (im2col + GEMM); GroupNorm/LayerNorm; T5/UMT5 encoder and SentencePiece-unigram tokenizer; CLIP text encoder; flow-matching and DPM samplers; VAEs; media writers | |

Dependency rule: `rie-core` as a Cargo git dependency pinned to a commit. Every change in
the right-hand column goes to RIE as its own small pull request with tests. No copied code.

### 5.3 Job directory: the whole interface

agentd and Ditto talk only through files and signals, as agentd and `agentos-llm train` do.

```
JOB/job.toml        prompt, negative prompt, kind (image|video), size, frames, fps, steps,
                    seed, model bundle, expert settings, optional input image
JOB/status.json     stage, step i of n, seconds spent, estimate left, last error
JOB/ckpt/           text.emb · latent.bin + sampler state + step index · decoded chunks
JOB/out/            image.png | clip.webp / clip.mp4 · poster.png · meta.json
```

Exit codes: `0` finished, `75` paused with a checkpoint (run again to continue), other
values failed, with the reason in `status.json`.

### 5.4 Stopping and resuming

| Event | What happens | Work lost |
|---|---|---|
| Incoming call, chat, approval key | agentd **freezes** Ditto's cgroup (the `freeze` crate): instant, the cores are free | None |
| Frozen for more than a few minutes, or memory is needed | SIGTERM. Ditto polls a stop flag after every transformer block and exits within seconds | At most one step, optionally none (block-level checkpoint, ~30 MB) |
| Unplugged, battery below the floor, temperature at `cool_c` | SIGTERM, exit `75` | At most one step |
| Power cut, kill -9 | Checkpoints are written to a temporary file, synced, then renamed | At most one step |

A checkpoint is the latent, the sampler state, the step index and the seed. All noise is
derived from `(seed, step)`, so a resumed job on the same backend gives the same file as an
uninterrupted one. This is a test (Phase 3), not a hope.

### 5.5 Memory plan (6 GB phone, 2-second 480×272 clip)

Stages run one after another and nothing from one stage stays resident in the next.

| Stage | File-backed (mmap, reclaimable) | Anonymous peak |
|---|---|---|
| Text encoder, 4-bit, layer-streamed, run once per prompt | ~3 GB on disk, 1 to 2 layers resident (~300 MB) | < 100 MB |
| Denoiser 1.3 B at 4-bit | ~0.8 GB | ~300 MB (hidden 28 MB; feed-forward and attention evaluated in token chunks) |
| VAE decode, tiled in space and chunked in time | ~0.5 GB | ≤ 400 MB, or far less with a tiny decoder |

Target: anonymous memory under 1 GB at all times. agentd unloads the large LLM before a
job (the Deep-stage `ModelBudget` decides that already) and the small one may stay.

### 5.6 Heat and power

- agentd's `prio` remains the authority. Ditto adds a **duty cycle**: it reads the
  hottest thermal zone between blocks and sleeps to hold a target (default 60 °C, below
  `cool_c`), instead of running into the 80 °C cut-off and oscillating.
- Battery floor as for training (`train_min_battery`, default 50 %), separate key.
- If the charger delivers less than the phone draws under load (level falling over
  10 minutes while charging), drop to fewer threads. Never discharge the phone at night.

---

## 6. MoE-fication of diffusion models

### 6.1 Level A: neuron experts (port of `rie moefy`)

1. Run the dense model on a calibration set: ~64 prompts × the sampler's steps, capturing
   each block's feed-forward input (RIE: `capture_ffn_inputs`).
2. Per block: neuron features `|activation| × ‖down column‖`, balanced k-means into `E`
   experts of equal size (`balanced_kmeans`), router by ridge regression (`solve_spd`).
3. Write a GGUF with stacked expert tensors and `ditto.expert_gating = "unit"`.
4. Contract carried over from RIE: **with all experts selected, output equals the dense
   model.** This is the first test of every converted file.

Differences from the LLM case, each a piece of work:

- The router input gets the timestep embedding in addition to the token.
- Plain MLP blocks with biases (the bias is split with its neurons).
- Calibration must cover the whole noise schedule, not one text.
- Feature vectors are tokens × steps × prompts long: subsample or sketch them.

### 6.2 Level B: noise-level experts

Per block and per noise bucket (for example 4 buckets), pick the `k` experts with the most
activation mass on calibration data and store the choice as a table in the GGUF.

- No router cost and no per-token gather: a step runs a plain, smaller dense matmul, which
  keeps the SIMD kernels at full speed.
- With layer streaming, only the selected experts' pages are read from storage.
- A step-distilled model has 4 to 8 steps, so "bucket" can mean "step": one fixed
  sub-network per step.
- Hybrid: table-chosen experts for every token, plus one or two router-chosen experts for
  the tokens the router scores highest.

### 6.3 Level C: experts from existing fine-tunes and LoRAs

- `ditto lora-pack`: take LoRAs as they are, or extract rank-r deltas from full fine-tunes
  (truncated SVD of `fine-tune − base`), and store them next to the base as named experts
  with a short description and a prompt-embedding centroid.
- Routing per job: similarity of the prompt embedding to the centroids, top-k, or named
  explicitly in `job.toml`. Optional per-noise-bucket weights (structure early, style late).
- Applied as a low-rank side path, as RIE's correction experts are, so the 4-bit base is
  never rewritten.

### 6.4 Level D: skip expert (caching)

- Per block: if the block input moved less than a threshold since the previous step, reuse
  the stored residual. For video, per token: static background tokens are skipped.
- Thresholds are calibrated with the same fidelity gate as the experts.

### 6.5 Quality gate

No converted model ships on a feeling. For a fixed set of prompts and seeds:

| Check | Pass |
|---|---|
| All experts on vs dense | Equal within quantization noise |
| `k` of `E` vs dense, same seed | PSNR/SSIM of the final frames above a floor fixed in Phase 5 (decided with you, on real samples) |
| Video: frame-to-frame flicker vs dense | Not worse than dense by more than a set margin |
| Time and bytes read per step | Reported next to the quality, so the trade is visible |

`ditto moefy` prints this report, like RIE's `LayerReport`. If the loss is too high the
fallback is a short distillation fine-tune of the router and experts on a workstation.
That is the one place where training may enter, and only if the gate demands it.

---

## 7. agenticOS integration

Changes live in agenticOS, in a new `crates/agentd/src/studio.rs` modelled on `trainer.rs`.

**Tools (through the executor)**

| Say | Tool | Tier |
|---|---|---|
| `make a video of a paper boat on a river` | `make_video` (queues) | Medium |
| `draw a poster for Diwali` | `make_image` (queues) | Medium |
| `what are you making`, `cancel the boat video` | `list_creations`, `cancel_creation` | Low / Medium |
| `show me last night's video` | `show_creation` (page with poster, path, prompt) | Low |
| `send it to Mum` | existing messaging tools | High: approval key |

The reply to a request is immediate and honest: *"Queued. I'll make it tonight on the
charger; about 50 minutes of work."* The estimate comes from `ditto bench` on that phone.

**Scheduling**

- `studio_tick` next to `train_tick`. Same `Conditions` plus "queue not empty".
- One heavy job at a time. Order per night: LoRA training first (short, gated by
  `train_min_new`), then creations, oldest first. Images before videos when both wait.
- Queue changes raise `WorkQueued` / `QueueEmpty`, so an empty queue winds the phone down.
- `prio.adopt(pid)`, `Mode::Training`. Start, pause, resume, finish and failure each write
  one ledger line and update `agentd ctl studio status`.

**Config** (`agentd.toml`)

```toml
studio_enabled = true
ditto = "/usr/bin/ditto"
studio_min_battery = 50
studio_target_c = 60.0
studio_max_night_hours = 6
[[media_models]]            # like [[models]]: url + sha256 per file, resumable
name = "wan2.1-1.3b-moe"
kind = "video"
files = [ { file = "...-dit-q4.gguf", url = "...", sha256 = "..." }, ... ]
```

**Storage, screen, privacy**

- Outputs under `data_dir/creations/`, with `meta.json`: prompt, seed, model hashes,
  "made by Ditto on this phone". The same label is embedded in the file's metadata.
- `photos` is untouched. `pages` gets an image block; `agentos-ui` shows the poster frame
  and plays the clip. Until then `show_creation` returns the path.
- Nothing leaves the phone. Prompts and outputs are local; downloads are model files only.
- Prompts pass agentd's existing firewall first, with a short refusal list for content that
  must never be generated (sexual content involving minors; sexual or degrading images of
  real, named people). Using a photo of a person as input is allowed only for photos already
  on the phone, and the result is labelled as generated.

**Packaging and tests**

- `os/postmarketos-ditto/APKBUILD`: the static binary, no dependencies. Models are pulled
  by `agentd models pull`, never packaged.
- A fake `ditto` binary (as `fake_llama.rs` is for the LLM) drives agentd's e2e tests:
  queue, start in Deep, pause on unplug, resume, finish, digest line, ledger lines.
- `phonesim` scenario: a call arrives mid-job, the job freezes, the call rings on time.

---

## 8. Output formats

| Kind | Default | Notes |
|---|---|---|
| Image | PNG | Small pure-Rust encoder |
| Video | Decide in Phase 0 | Options: AV1 via a pure-Rust encoder (slow, but it is night), animated WebP, or the SoC's hardware encoder through V4L2 where the mainline kernel has it. Messaging apps decide which is useful, so test what the receiving phones play |
| Always | `poster.png` + `meta.json` | For the digest and the UI |

---

## 9. Verification

- **Reference parity.** `tools/reference` dumps per-block activations and final outputs
  for fixed seeds from the reference Python implementation. Ditto at F32 must match per
  block within tolerance; at 4-bit, final-frame PSNR must be above a floor.
- **Kernels.** New SIMD kernels are tested against scalar on x86 and on emulated ARM CPUs
  (dotprod, i8mm, neither), as RIE does.
- **No downloads in `cargo test`.** Tiny random models are written as GGUF by the tests
  (the pattern of `rie-core/src/testing.rs`) and exercise every stage.
- **Resume.** Kill at random points: output identical to an uninterrupted run.
- **Stop latency.** SIGTERM to exit, measured on the phone, under a fixed bound.
- **Soak.** Three consecutive nights on a real phone: no crash, no discharge, no missed
  call, temperature inside the target.

---

## 10. Phases

Each phase ends with something that runs and a number that was measured.

| Phase | Work | Done when |
|---|---|---|
| **0. Measure and decide** | `ditto bench` with synthetic weights in real DiT shapes (RIE's `bench --synthetic` pattern) on a Snapdragon 845 phone and a Fairphone-class phone: multiply-adds per second, attention and convolution cost, sustained temperature, storage read speed. Check licences. Pick image model, video model, output format. | Section 3's table is rewritten with measured numbers and the models are fixed. **Go / no-go for video.** |
| **1. Foundations** | Workspace, RIE dependency, RIE pull requests (cpuset, hidden states, generic streaming, public clustering). `safetensors → GGUF`. New ops with scalar and SIMD paths. Parity harness. | Ops match reference dumps; RIE pull requests merged. |
| **2. Images on the desktop, then the phone** | Text encoder, image transformer, VAE decode, samplers, PNG. Quantized models. | Same seed gives the reference image within the PSNR floor; one image on the phone, timed. |
| **3. The night runner** | Job directory, checkpoints, stop flag, duty cycle, staged memory, `status.json`. | Kill/resume and stop-latency tests pass; anonymous memory under 1 GB. |
| **4. Video** | 3-D patches and positions, the video transformer, few-step sampler, tiled and chunked 3-D VAE decode, tiny decoder, export. | A 2-second 480×272 clip on the phone in one night, resumed at least once on the way. |
| **5. Experts** | `ditto moefy` levels A and B with the quality report; then C (`lora-pack`) and D (caching). | Converted model passes the gate and is measurably faster; numbers recorded in `docs/PROGRESS.md`. |
| **6. agenticOS** | `studio.rs`, tools, intents, config, package, fake-binary e2e tests, phonesim scenario, digest. | "make a video of …" in the evening, a clip and a digest line in the morning, on a real phone. |
| **7. GPU and speed** | WGSL attention and convolution on `rie-gpu`; per-device self-test against the CPU result before the GPU is trusted; i8mm tiles for the new kernels. | GPU path faster than CPU on at least one Mesa driver, with equal output. |
| **8. More from one night** | Generate small, then upscale and interpolate frames with small dedicated models (a cascade of experts). Photo-to-video. UNet family for its LoRA library. | Decided after Phase 6 from what users ask for. |

Phases 0 to 3 do not depend on video at all and already give the OS on-device image
generation. Phases 5 and 6 can run in parallel once 4 is done.

A `docs/PROGRESS.md` is kept from Phase 0, in the style of agenticOS's: measured numbers
only, with the machine named.

---

## 11. Risks

| Risk | Effect | Answer |
|---|---|---|
| Real throughput is well below 15 G MAC/s | Video does not fit a night | Phase 0 gate. Fallbacks: LTX-class model (fewer tokens), smaller clips, GPU, images only |
| No usable few-step variant of the chosen video model | 5 to 10× more steps | Check in Phase 0; choose the model that has one; multi-night jobs already work |
| MoE-fied diffusion loses too much quality without training | The speciality under-delivers | Level B first (most robust), exactness at `k = E` as the safety net, distillation on a workstation as the fallback |
| The text encoder (5 B+) is slow or thrashes storage when streamed | Minutes added, flash wear | Run once per prompt and cache the embedding; measure read volume; prefer a model with a small encoder |
| Mesa Vulkan drivers fail or are slower than the CPU | No GPU gain | CPU is the baseline by design |
| 4-bit weights hurt diffusion more than LLMs | Visible artefacts | Mixed precision: 8-bit for the first and last blocks and embeddings (as `quantize_model` keeps the output head at Q8_0); gate by PSNR |
| Heat or battery ageing from nightly load | Hardware harm | Duty cycle to 60 °C, never discharge, hours cap per night, soak test |
| Licences forbid redistribution or commercial use | Cannot ship a default | Verified in Phase 0; the config only references URLs, the user pulls the files |
| RIE and Ditto drift apart | Double maintenance | Pinned dependency, changes upstreamed, no copies |

---

## 12. Decisions needed from you

1. **The reading of "MoE from existing models"** in section 1: levels A to D, with B first.
   If you meant something else (for example merging several complete models as experts),
   section 6 changes.
2. **The target phone** for Phase 0. The plan assumes a Snapdragon 845 phone with 6 to 8 GB.
3. **Commercial use or personal use only.** This narrows the model list.
4. **What counts as good enough for the first video:** the plan assumes 2 seconds at
   480×272 and 16 fps.
5. **Who changes agenticOS and RIE.** The plan assumes pull requests to both from this work.
