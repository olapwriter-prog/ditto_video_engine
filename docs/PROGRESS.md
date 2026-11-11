# Progress

Measured numbers only, with the machine named. Estimates live in [PLAN.md](PLAN.md).

## Phase 0: measure and decide (started 2026-11-10)

**No real phone has been measured yet.** Speeds below are from the development machine, an
x86-64 cloud VM (Intel Xeon 2.1 GHz, 4 cores, AVX2, 16 GB). It says the tool works and how
the work divides; it does not say how fast a phone is.

### What exists

- `ditto bench`: times one denoiser block with the shapes of a real model and random
  weights, through `rie-core`'s quantized kernels, and scales it to a step and a job.
- `ditto estimate`: the plan's table from a throughput figure, without running anything.
- New kernels `rie-core` does not have: attention from every token to every token
  (blocked, keys and values transposed), and vectorised `exp`, softmax, GELU and layer norm.
- The feed-forward block split into experts. Test: with every expert on, the result equals
  the dense block.
- Tests pass (`cargo test --workspace`), no downloads.
- `scripts/build-phone.sh` builds a static ARM64 binary (2.1 MB) with Rust's own linker.
  **Built, not run:** there is no ARM machine or emulator here, so the ARM path is
  untested. `scripts/phone-bench.sh` runs the measurements on the phone; it was dry-run
  here with the x86 binary.

### Measured on the x86-64 VM (4 threads, Q4_0 weights, AVX2 kernels)

| Job | Tokens | One step | Whole job, dense | 2 of 8 experts |
|---|---|---|---|---|
| PixArt-class image, 512×512, 4 steps | 1,024 | 11 to 12 s | 44 to 49 s | 35 to 39 s (80 %) |
| Wan-1.3B-class, 320×192, 17 frames, 6 steps | 1,200 | 32 s | 3.1 to 3.2 min | 2.0 to 2.1 min (65 %) |
| Wan-1.3B-class, 480×272, 33 frames, 6 steps | 4,590 | 2.9 min | 17.2 to 17.5 min | 12.5 to 13.5 min (72 to 77 %) |

Ranges are two runs. Denoising only: text encoding, decoding and export are not built.

- **Throughput:** weight matrices 57 to 61 G multiply-adds per second; attention 22 to 27 G/s.
- **Where the time goes at 4,590 tokens:** weights 58 %, self-attention 40 %, the rest 3 %.
  With 2 of 8 experts, self-attention becomes the larger half (54 %).
- **Threads:** the 1,200-token clip takes 11.9 min on one thread and 3.1 min on four (3.8×).
- **Memory:** peak 270 MB at 4,590 tokens for one block's weights and all buffers.

### What this changes in the plan

1. **Attention is the kernel to improve first.** It runs at under half the speed of the
   quantized weight kernels (f32 against int8). Candidates: int8 scores as in `rie-core`'s
   TurboQuant attention, f16 keys and values, a proper GEMM in place of row-wise `axpy`.
2. **Experts save less than the arithmetic says.** The plan expected about 64 % of the dense
   time at 4,590 tokens; measured 72 to 77 %. Each expert quantizes its own activations and
   runs as a separate small product. Merging the selected experts into one product per
   chunk should recover most of the gap.
3. The plan's 15 G multiply-adds per second for a phone is still an assumption. This VM
   does 44 G/s overall, so the assumption puts a Snapdragon 845 at a third of it.

### Convolutions (added for the image decoder and the driver warning proposal)

- `ditto-core::conv`: 2-D convolution as a quantized matrix product, tested against the
  definition (strides, odd sizes, 1×1 to 5×5, more positions than one chunk).
- `ditto bench-conv`, same VM, 4 threads, Q4_0, a stack the size of a small detector's
  backbone with random weights (cost only):

| Input | Work per frame | One frame | Frames per second |
|---|---|---|---|
| 416×416 | 3.23 G multiply-adds | 90 ms | 11.1 |
| 416×416, half the channels | 0.88 G | 43 ms | 23.5 |
| 640×640 | 7.64 G | 201 ms | 5.0 |
| 416×416, one thread | 3.23 G | 215 ms | 4.6 |

- Throughput 36 to 38 G multiply-adds per second at full width, against 57 G/s for plain
  matrix products; four threads give only 2.4× over one. Laying out patches and quantizing
  them is the cost to remove (direct 3×3 kernels, or patches built in int8).

### Pictures the user took, screenshots as SVG, and the hand-over to the inference engine

Details and limits in [USE_CASES.md](USE_CASES.md).

- `ditto look`: time, place, camera, size, colours, sharpness, QR codes, duplicate
  fingerprint, photo or screenshot. No model; it reads no text and names no objects.
- `ditto trace`: flat pictures as SVG, exact to the pixel.
- `integrations/rie/ditto-tools.json`, `--rie URL`, `--memory DIR`: results reach the
  inference engine as tool output and as memory.
- Tests: 46 pass in the workspace (19 kernels, 23 pictures, 4 end to end). The end-to-end
  tests load the shipped manifest with `rie-server`'s loader, run the tools with its runner,
  and read the memory back with `rie-core`'s store.
- Browser check (Chromium, x86-64 VM): a 420×300 page capture traced and rendered again
  gives a byte-identical SVG at 8, 16 and 256 colours. SVG size 29.6 / 45.5 / 67.5 KB
  against a 13.5 KB PNG: anti-aliased text is expensive as outlines.
- Not tested: a live `rie-server` with a model calling these tools; any ARM device.

### Scripts of all Indian languages

- `ditto-vision::scripts` and `ditto scripts TEXT`: splits mixed text (a sign in Telugu,
  Hindi and English) into runs of one script, lists the languages each run could be, and
  writes every script's digits as 0-9. Covers the scripts of India's 22 scheduled languages
  and English, plus Greek and Cyrillic for Europe.
- The script is detected exactly. The **language is not**: eleven scheduled languages can
  be written in Devanagari, so a run lists candidates. Only Assamese is told from Bengali,
  by its own letters.
- This is the routing step for text reading and spoken warnings. It reads no text from a
  picture: there is still no text-reading model. Warning phrases in the 22 languages are
  **not written**; they need speakers to write and check them.

### Emulated phone processors (`scripts/sim-phones.sh`)

Run on the x86-64 VM with QEMU 8.2 user-mode emulation: 41 checks, 0 failed.

| Emulated processor | Stands for | Tests | Kernel chosen at run time |
|---|---|---|---|
| Cortex-A53, 64-bit | 2015-2019 and entry-level phones, PinePhone | 52 pass | `neon` |
| Cortex-A76, 64-bit | 2018-2021 phones (Snapdragon 845-888 class) | 52 pass | `neon-dotprod` |
| Cortex-A710, 64-bit | 2022 and later | 52 pass | `neon-i8mm` |
| "max", 64-bit | Every instruction the emulator knows | 52 pass | `neon-i8mm` |
| Cortex-A7, 32-bit | 32-bit phones before about 2016 | 52 pass | `scalar` |
| "max", 32-bit | 32-bit system on a newer processor | 52 pass | `scalar` |

- On every profile the SVG of the browser capture, the picture facts and the script
  split are byte-identical to the x86 results.
- Compile only (cannot be run here): Android on arm64, 32-bit ARM and x86-64; iPhone and
  the iPhone simulator target. All compile.
- **What this is not.** It is not the Android emulator or Apple's iPhone simulator: no
  Android or iOS system ran, and no app exists to run in them. No camera, GPU, heat or
  speed was simulated; emulated time says nothing about a phone. "All mobile phones" is
  covered as processor classes, not as individual models. 32-bit phones have no fast
  kernel (`scalar`) and would be far too slow for models.

### Generated pictures labelled and checked ([HONEST_PICTURES.md](HONEST_PICTURES.md))

- `ditto label`: the IPTC `trainedAlgorithmicMedia` marker in XMP, a signed record
  (prompt, seed, model, pixel SHA-256, Ed25519), optional visible "AI" badge.
- `ditto verify`, and `ditto look`: intact, edited after generation, tampered, another
  generator's marker, C2PA present (not validated), or "no marker, which does not show the
  picture is real".
- `ditto compare`: faithful / drifted / different outside the area an edit was allowed to
  change, with where it changed. Thresholds not yet tuned on real edits.
- Generated pictures go into the inference engine's memory as "not a photograph, not
  evidence"; new tool `photo.compare_edit`.
- exiftool reads Ditto's marker as the standard digital source type; Pillow opens the files.
- Tests: 67 pass in the workspace, and on all six emulated phone processors (`scripts/sim-phones.sh`, 41 of 41 checks).

### Next

- Run `scripts/phone-bench.sh` on a phone and record the output here. This decides whether
  video is go or no-go.
- Licences and exact shapes of the candidate models, from their model cards.
- The decisions listed in section 12 of the plan.
