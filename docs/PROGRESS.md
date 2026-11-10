# Progress

Measured numbers only, with the machine named. Estimates live in [PLAN.md](PLAN.md).

## Phase 0: measure and decide (started 2026-11-10)

**No phone has been measured yet.** Everything below is from the development machine, an
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
- `scripts/build-phone.sh` builds a static ARM64 binary (750 KB) with Rust's own linker.
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
  definition (strides, odd sizes, 1×1 to 5×5, more positions than one chunk). 19 tests pass.
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

### Next

- Run `scripts/phone-bench.sh` on a phone and record the output here. This decides whether
  video is go or no-go.
- Licences and exact shapes of the candidate models, from their model cards.
- The decisions listed in section 12 of the plan.
