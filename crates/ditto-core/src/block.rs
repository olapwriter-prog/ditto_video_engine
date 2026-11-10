//! One denoiser block with the shapes of a real model and random weights.
//!
//! This is what Phase 0 times: self-attention, cross-attention to the text and the
//! feed-forward block, with weights in the same quantized formats and going through the
//! same `rie-core` kernels a real model will use. Left out because they cost next to
//! nothing: biases, position encodings and the timestep modulation.
//!
//! The feed-forward block comes in two forms: dense, or split into equal experts of which
//! only some run (levels A and B of the plan). With every expert selected the two forms
//! compute the same function, which the tests check.

use crate::attention::{attend, Memory};
use crate::math::{gelu_tanh, layer_norm};
use rie_core::kernels::matmul::{formats_for, matmul, matmul_multi, MatmulJob, QuantActs};
use rie_core::quant::{quantize, GgmlType};
use rie_core::tensor::Weight;
use rie_core::threadpool::ThreadPool;
use std::time::{Duration, Instant};

/// Tokens that go through the feed-forward block together. Bounds its largest buffer
/// (`FFN_CHUNK × ffn` floats) whatever the clip size.
pub const FFN_CHUNK: usize = 256;

pub enum Ffn {
    /// `up`: `ffn × dim`, `down`: `dim × ffn`.
    Dense { up: Weight, down: Weight },
    /// Per expert `up`: `m × dim` and `down`: `dim × m`, with `m = ffn / experts`.
    Experts { up: Vec<Weight>, down: Vec<Weight> },
}

pub struct Block {
    pub dim: usize,
    pub ffn_dim: usize,
    pub heads: usize,
    wq: Weight,
    wk: Weight,
    wv: Weight,
    wo: Weight,
    cq: Weight,
    co: Weight,
    pub ffn: Ffn,
}

/// Where the time of one block went.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timing {
    /// Weight matrices, including quantizing their inputs.
    pub linear: Duration,
    pub self_attention: Duration,
    pub cross_attention: Duration,
    /// Norms, the activation function, residual adds.
    pub other: Duration,
}

impl Timing {
    pub fn total(&self) -> Duration {
        self.linear + self.self_attention + self.cross_attention + self.other
    }
}

fn random(n: usize, scale: f32, seed: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            ((*seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * scale
        })
        .collect()
}

fn encode(ty: GgmlType, rows: usize, cols: usize, x: &[f32]) -> Weight {
    let data = match ty {
        GgmlType::Q4_0 => quantize::q4_0(x),
        GgmlType::Q8_0 => quantize::q8_0(x),
        GgmlType::F16 => quantize::f16(x),
        GgmlType::F32 => quantize::f32(x),
        other => panic!("synthetic weights in {} are not supported", other.name()),
    };
    Weight::owned(ty, rows, cols, data)
}

impl Block {
    /// Random weights of type `ty`. The same `seed` gives the same underlying values
    /// whether the feed-forward block is dense (`experts == None`) or split.
    pub fn synthetic(
        dim: usize,
        ffn_dim: usize,
        heads: usize,
        ty: GgmlType,
        experts: Option<usize>,
        seed: u64,
    ) -> Block {
        assert!(dim % heads == 0);
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let sq = |s: &mut u64| {
            encode(
                ty,
                dim,
                dim,
                &random(dim * dim, 1.0 / (dim as f32).sqrt(), s),
            )
        };
        let (wq, wk, wv, wo, cq, co) = (
            sq(&mut s),
            sq(&mut s),
            sq(&mut s),
            sq(&mut s),
            sq(&mut s),
            sq(&mut s),
        );
        let up = random(ffn_dim * dim, 1.0 / (dim as f32).sqrt(), &mut s);
        let down = random(dim * ffn_dim, 1.0 / (ffn_dim as f32).sqrt(), &mut s);
        let ffn = match experts {
            None => Ffn::Dense {
                up: encode(ty, ffn_dim, dim, &up),
                down: encode(ty, dim, ffn_dim, &down),
            },
            Some(e) => {
                assert!(
                    e >= 1 && ffn_dim % e == 0,
                    "feed-forward width {ffn_dim} is not divisible by {e} experts"
                );
                let m = ffn_dim / e;
                let ups = (0..e)
                    .map(|x| encode(ty, m, dim, &up[x * m * dim..(x + 1) * m * dim]))
                    .collect();
                let downs = (0..e)
                    .map(|x| {
                        // columns x·m .. (x+1)·m of `down`
                        let mut d = Vec::with_capacity(dim * m);
                        for r in 0..dim {
                            d.extend_from_slice(
                                &down[r * ffn_dim + x * m..r * ffn_dim + (x + 1) * m],
                            );
                        }
                        encode(ty, dim, m, &d)
                    })
                    .collect();
                Ffn::Experts {
                    up: ups,
                    down: downs,
                }
            }
        };
        Block {
            dim,
            ffn_dim,
            heads,
            wq,
            wk,
            wv,
            wo,
            cq,
            co,
            ffn,
        }
    }

    pub fn experts(&self) -> usize {
        match &self.ffn {
            Ffn::Dense { .. } => 1,
            Ffn::Experts { up, .. } => up.len(),
        }
    }

    /// Bytes of weights.
    pub fn size_bytes(&self) -> usize {
        let mut n: usize = [&self.wq, &self.wk, &self.wv, &self.wo, &self.cq, &self.co]
            .iter()
            .map(|w| w.size_bytes())
            .sum();
        match &self.ffn {
            Ffn::Dense { up, down } => n += up.size_bytes() + down.size_bytes(),
            Ffn::Experts { up, down } => {
                n += up.iter().chain(down).map(|w| w.size_bytes()).sum::<usize>()
            }
        }
        n
    }

    /// Run the block on `x` (`n × dim`, updated in place). `text` holds the prompt's keys
    /// and values. `used` lists the experts to run; it is ignored by a dense block.
    pub fn forward(
        &self,
        pool: &ThreadPool,
        x: &mut [f32],
        n: usize,
        text: &Memory,
        used: &[usize],
    ) -> Timing {
        let d = self.dim;
        assert!(x.len() >= n * d);
        let mut t = Timing::default();
        let mut acts = QuantActs::new();
        let normed = |x: &[f32]| {
            let mut h = x[..n * d].to_vec();
            let p = SharedMut(h.as_mut_ptr());
            pool.for_each(n, 16, |i, _| {
                // SAFETY: each token is its own row.
                layer_norm(unsafe { std::slice::from_raw_parts_mut(p.get().add(i * d), d) })
            });
            h
        };
        let add = |x: &mut [f32], y: &[f32]| x.iter_mut().zip(y).for_each(|(a, b)| *a += *b);

        // self-attention
        let c = Instant::now();
        let h = normed(x);
        t.other += c.elapsed();
        let c = Instant::now();
        let (mut q, mut k, mut v) = (vec![0.0; n * d], vec![0.0; n * d], vec![0.0; n * d]);
        acts.prepare(
            pool,
            &h,
            n,
            d,
            &formats_for(&[&self.wq, &self.wk, &self.wv]),
        );
        matmul_multi(
            pool,
            &acts,
            &mut [
                MatmulJob {
                    w: &self.wq,
                    out: &mut q,
                    ld: d,
                },
                MatmulJob {
                    w: &self.wk,
                    out: &mut k,
                    ld: d,
                },
                MatmulJob {
                    w: &self.wv,
                    out: &mut v,
                    ld: d,
                },
            ],
        );
        t.linear += c.elapsed();
        let c = Instant::now();
        let mem = Memory::new(pool, &k, &v, n, d, self.heads);
        drop((k, v));
        let mut a = vec![0.0; n * d];
        attend(pool, &q, n, &mem, &mut a);
        drop(mem);
        t.self_attention += c.elapsed();
        let c = Instant::now();
        acts.prepare(pool, &a, n, d, &formats_for(&[&self.wo]));
        matmul(pool, &acts, &self.wo, &mut q);
        t.linear += c.elapsed();
        let c = Instant::now();
        add(x, &q);

        // cross-attention to the text
        let h = normed(x);
        t.other += c.elapsed();
        let c = Instant::now();
        acts.prepare(pool, &h, n, d, &formats_for(&[&self.cq]));
        matmul(pool, &acts, &self.cq, &mut q);
        t.linear += c.elapsed();
        let c = Instant::now();
        attend(pool, &q, n, text, &mut a);
        t.cross_attention += c.elapsed();
        let c = Instant::now();
        acts.prepare(pool, &a, n, d, &formats_for(&[&self.co]));
        matmul(pool, &acts, &self.co, &mut q);
        t.linear += c.elapsed();
        let c = Instant::now();
        add(x, &q);
        drop((q, a));

        // feed-forward, a chunk of tokens at a time
        let h = normed(x);
        t.other += c.elapsed();
        let width = match &self.ffn {
            Ffn::Dense { .. } => self.ffn_dim,
            Ffn::Experts { .. } => self.ffn_dim / self.experts(),
        };
        let mut u = vec![0.0f32; FFN_CHUNK.min(n) * width];
        let mut y = vec![0.0f32; FFN_CHUNK.min(n) * d];
        let mut acts2 = QuantActs::new();
        let gelu = |u: &mut [f32]| {
            let p = SharedMut(u.as_mut_ptr());
            let len = u.len();
            pool.for_each(len.div_ceil(4096), 1, |i, _| {
                let (a, b) = (i * 4096, ((i + 1) * 4096).min(len));
                // SAFETY: disjoint ranges.
                gelu_tanh(unsafe { std::slice::from_raw_parts_mut(p.get().add(a), b - a) })
            });
        };
        let mut t0 = 0;
        while t0 < n {
            let m = FFN_CHUNK.min(n - t0);
            let hc = &h[t0 * d..(t0 + m) * d];
            let xc = &mut x[t0 * d..(t0 + m) * d];
            let mut one = |up: &Weight, down: &Weight, acts: &QuantActs, t: &mut Timing| {
                let c = Instant::now();
                matmul(pool, acts, up, &mut u[..m * width]);
                t.linear += c.elapsed();
                let c = Instant::now();
                gelu(&mut u[..m * width]);
                t.other += c.elapsed();
                let c = Instant::now();
                acts2.prepare(pool, &u, m, width, &formats_for(&[down]));
                matmul(pool, &acts2, down, &mut y[..m * d]);
                t.linear += c.elapsed();
                let c = Instant::now();
                add(xc, &y[..m * d]);
                t.other += c.elapsed();
            };
            let c = Instant::now();
            match &self.ffn {
                Ffn::Dense { up, down } => {
                    acts.prepare(pool, hc, m, d, &formats_for(&[up]));
                    t.linear += c.elapsed();
                    one(up, down, &acts, &mut t);
                }
                Ffn::Experts { up, down } => {
                    acts.prepare(pool, hc, m, d, &formats_for(&[&up[0]]));
                    t.linear += c.elapsed();
                    for &e in used {
                        one(&up[e], &down[e], &acts, &mut t);
                    }
                }
            }
            t0 += m;
        }
        t
    }
}

#[derive(Clone, Copy)]
struct SharedMut(*mut f32);
unsafe impl Send for SharedMut {}
unsafe impl Sync for SharedMut {}
impl SharedMut {
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::tests::{noise, pool};

    const DIM: usize = 64;
    const FFN: usize = 256;

    fn run(block: &Block, n: usize, used: &[usize], threads: usize) -> Vec<f32> {
        let p = pool(threads);
        let text = Memory::new(
            &p,
            &noise(5 * DIM, 11),
            &noise(5 * DIM, 12),
            5,
            DIM,
            block.heads,
        );
        let mut x = noise(n * DIM, 7);
        block.forward(&p, &mut x, n, &text, used);
        assert!(x.iter().all(|v| v.is_finite()));
        x
    }

    fn worst(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// The contract carried over from `rie moefy`: all experts on equals the dense block.
    #[test]
    fn all_experts_equal_the_dense_block() {
        let dense = Block::synthetic(DIM, FFN, 4, GgmlType::F32, None, 3);
        let split = Block::synthetic(DIM, FFN, 4, GgmlType::F32, Some(8), 3);
        let n = 21;
        let a = run(&dense, n, &[], 2);
        let b = run(&split, n, &[0, 1, 2, 3, 4, 5, 6, 7], 2);
        assert!(worst(&a, &b) < 1e-4, "{}", worst(&a, &b));
        // and the order of the experts does not matter
        let c = run(&split, n, &[7, 3, 0, 5, 1, 6, 2, 4], 2);
        assert!(worst(&a, &c) < 1e-4);
        // leaving experts out does change the result
        let d = run(&split, n, &[0, 1], 2);
        assert!(worst(&a, &d) > 1e-3);
    }

    /// Chunking the feed-forward block and the thread count must not change the result.
    #[test]
    fn chunks_and_threads_do_not_change_the_result() {
        let b = Block::synthetic(DIM, FFN, 4, GgmlType::F32, None, 5);
        let n = FFN_CHUNK + 37; // crosses a chunk boundary
        let one = run(&b, n, &[], 1);
        let four = run(&b, n, &[], 4);
        assert!(worst(&one, &four) < 1e-5);
        // the same for a quantized block that runs some of its experts
        let q = Block::synthetic(DIM, FFN, 4, GgmlType::Q4_0, Some(4), 5);
        let a = run(&q, n, &[1, 2], 1);
        let b = run(&q, n, &[1, 2], 3);
        assert!(worst(&a, &b) < 1e-5);
    }

    /// 4-bit weights stay close to the float block they were made from.
    #[test]
    fn quantized_block_tracks_the_float_block() {
        let f = Block::synthetic(DIM, FFN, 4, GgmlType::F32, None, 9);
        let q8 = Block::synthetic(DIM, FFN, 4, GgmlType::Q8_0, None, 9);
        let q4 = Block::synthetic(DIM, FFN, 4, GgmlType::Q4_0, None, 9);
        let (a, b, c) = (
            run(&f, 16, &[], 2),
            run(&q8, 16, &[], 2),
            run(&q4, 16, &[], 2),
        );
        let rms = |x: &[f32], y: &[f32]| {
            (x.iter().zip(y).map(|(p, q)| (p - q) * (p - q)).sum::<f32>() / x.len() as f32).sqrt()
        };
        let scale = (a.iter().map(|v| v * v).sum::<f32>() / a.len() as f32).sqrt();
        assert!(
            rms(&a, &b) / scale < 0.02,
            "Q8_0 error {}",
            rms(&a, &b) / scale
        );
        assert!(
            rms(&a, &c) / scale < 0.15,
            "Q4_0 error {}",
            rms(&a, &c) / scale
        );
        assert!(rms(&a, &b) < rms(&a, &c));
        assert!(q4.size_bytes() < q8.size_bytes() && q8.size_bytes() < f.size_bytes());
    }
}
