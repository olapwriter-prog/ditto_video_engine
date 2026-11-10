//! Attention over a whole set of tokens at once.
//!
//! An LLM attends from one new token to a cache of earlier ones. A denoiser attends from
//! every token to every token, in every block of every step: `n² · dim` multiply-adds, the
//! part of a step that grows fastest with clip size. `rie-core` has no kernel for this.
//!
//! Layout. Keys and values are transposed once per call to `[head][channel][token]`, so
//! both inner loops run along the token axis, over long contiguous vectors:
//!
//! * scores of one query: `s += q[c] · K[c][..]` for each channel (an `axpy`),
//! * output of one query: `out[c] = s · V[c][..]` (a dot product).
//!
//! Queries are taken in small groups against blocks of keys, so that a block of keys stays
//! in the cache while every query of the group uses it.

use crate::math::{axpy, softmax};
use rie_core::kernels::dot_f32;
use rie_core::threadpool::ThreadPool;

/// Queries handled together by one thread.
const QUERY_GROUP: usize = 16;
/// Bytes of one head's keys (or values) that a block may span: about half a phone core's L2.
const BLOCK_BYTES: usize = 128 * 1024;

/// Keys and values of the tokens being attended to, transposed for [`attend`].
pub struct Memory {
    n: usize,
    dim: usize,
    heads: usize,
    /// `[head][channel][token]`
    kt: Vec<f32>,
    vt: Vec<f32>,
}

impl Memory {
    /// `k` and `v` are `n × dim`, token after token.
    pub fn new(
        pool: &ThreadPool,
        k: &[f32],
        v: &[f32],
        n: usize,
        dim: usize,
        heads: usize,
    ) -> Self {
        assert!(dim % heads == 0 && k.len() >= n * dim && v.len() >= n * dim);
        let mut m = Memory {
            n,
            dim,
            heads,
            kt: vec![0.0; n * dim],
            vt: vec![0.0; n * dim],
        };
        let (kt, vt) = (Shared(m.kt.as_mut_ptr()), Shared(m.vt.as_mut_ptr()));
        // channel c of the model is row c of the transposed arrays
        pool.for_each(dim, 8, |c, _| {
            // SAFETY: each channel writes its own row of `n` values.
            let (kr, vr) = unsafe {
                (
                    std::slice::from_raw_parts_mut(kt.get().add(c * n), n),
                    std::slice::from_raw_parts_mut(vt.get().add(c * n), n),
                )
            };
            for t in 0..n {
                kr[t] = k[t * dim + c];
                vr[t] = v[t * dim + c];
            }
        });
        m
    }

    pub fn tokens(&self) -> usize {
        self.n
    }
}

#[derive(Clone, Copy)]
struct Shared(*mut f32);
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}
impl Shared {
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// `out[i] = Σ_j softmax_j(q[i]·k[j] / sqrt(head_dim)) · v[j]`, per head.
/// `q` and `out` are `n_q × dim`. Self-attention passes the same tokens as queries and
/// memory; cross-attention passes the text as memory.
pub fn attend(pool: &ThreadPool, q: &[f32], n_q: usize, mem: &Memory, out: &mut [f32]) {
    let (n, dim, heads) = (mem.n, mem.dim, mem.heads);
    let hd = dim / heads;
    assert!(q.len() >= n_q * dim && out.len() >= n_q * dim);
    if n == 0 || n_q == 0 {
        out[..n_q * dim].fill(0.0);
        return;
    }
    let scale = 1.0 / (hd as f32).sqrt();
    let block = (BLOCK_BYTES / (hd * 4)).clamp(64, n.max(64));
    let groups = n_q.div_ceil(QUERY_GROUP);
    let outp = Shared(out.as_mut_ptr());

    pool.for_each(heads * groups, 1, |unit, _| {
        let (h, g) = (unit / groups, unit % groups);
        let (i0, i1) = (g * QUERY_GROUP, ((g + 1) * QUERY_GROUP).min(n_q));
        let nq = i1 - i0;
        let kt = &mem.kt[h * hd * n..(h + 1) * hd * n];
        let vt = &mem.vt[h * hd * n..(h + 1) * hd * n];
        let mut s = vec![0.0f32; nq * n];

        let mut j0 = 0;
        while j0 < n {
            let j1 = (j0 + block).min(n);
            for i in 0..nq {
                let qi = &q[(i0 + i) * dim + h * hd..(i0 + i) * dim + (h + 1) * hd];
                let si = &mut s[i * n + j0..i * n + j1];
                for c in 0..hd {
                    axpy(si, qi[c] * scale, &kt[c * n + j0..c * n + j1]);
                }
            }
            j0 = j1;
        }
        for i in 0..nq {
            softmax(&mut s[i * n..(i + 1) * n]);
        }
        // SAFETY: this unit owns head `h` of queries `i0..i1` in `out`.
        let o = |i: usize| unsafe {
            std::slice::from_raw_parts_mut(outp.get().add((i0 + i) * dim + h * hd), hd)
        };
        for i in 0..nq {
            o(i).fill(0.0);
        }
        let mut j0 = 0;
        while j0 < n {
            let j1 = (j0 + block).min(n);
            for i in 0..nq {
                let si = &s[i * n + j0..i * n + j1];
                let oi = o(i);
                for c in 0..hd {
                    oi[c] += dot_f32(si, &vt[c * n + j0..c * n + j1]);
                }
            }
            j0 = j1;
        }
    });
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use rie_core::threadpool::PoolConfig;

    pub fn pool(n: usize) -> ThreadPool {
        ThreadPool::new(PoolConfig {
            n_threads: n,
            cores: vec![],
            spin: std::time::Duration::from_micros(50),
        })
    }

    pub fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    /// The definition, in f64, with no blocking and no transposition.
    fn reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_q: usize,
        n: usize,
        dim: usize,
        heads: usize,
    ) -> Vec<f64> {
        let hd = dim / heads;
        let mut out = vec![0.0f64; n_q * dim];
        for h in 0..heads {
            for i in 0..n_q {
                let s: Vec<f64> = (0..n)
                    .map(|j| {
                        (0..hd)
                            .map(|c| {
                                q[i * dim + h * hd + c] as f64 * k[j * dim + h * hd + c] as f64
                            })
                            .sum::<f64>()
                            / (hd as f64).sqrt()
                    })
                    .collect();
                let max = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = s.iter().map(|x| (x - max).exp()).collect();
                let z: f64 = e.iter().sum();
                for c in 0..hd {
                    out[i * dim + h * hd + c] = (0..n)
                        .map(|j| e[j] / z * v[j * dim + h * hd + c] as f64)
                        .sum();
                }
            }
        }
        out
    }

    fn check(n_q: usize, n: usize, dim: usize, heads: usize, threads: usize) {
        let (q, k, v) = (noise(n_q * dim, 1), noise(n * dim, 2), noise(n * dim, 3));
        // sharpen the scores so the softmax is not flat
        let q: Vec<f32> = q.iter().map(|x| x * 4.0).collect();
        let p = pool(threads);
        let mem = Memory::new(&p, &k, &v, n, dim, heads);
        let mut out = vec![f32::NAN; n_q * dim];
        attend(&p, &q, n_q, &mem, &mut out);
        let want = reference(&q, &k, &v, n_q, n, dim, heads);
        for (i, (g, w)) in out.iter().zip(&want).enumerate() {
            assert!(
                (*g as f64 - w).abs() < 2e-5,
                "n_q {n_q} n {n} at {i}: {g} vs {w}"
            );
        }
    }

    #[test]
    fn matches_the_definition() {
        check(5, 5, 8, 2, 1); // tiny
        check(37, 37, 32, 4, 3); // self-attention, odd count, last group partial
        check(19, 7, 32, 4, 2); // cross-attention: fewer memory tokens than queries
        check(3, 700, 16, 2, 2); // several key blocks
    }

    #[test]
    fn a_token_with_a_dominant_key_copies_its_value() {
        let (n, dim) = (6, 4);
        let mut k = vec![0.0f32; n * dim];
        let v = noise(n * dim, 9);
        k[2 * dim..3 * dim].fill(50.0);
        let q = vec![1.0f32; dim];
        let p = pool(1);
        let mut out = vec![0.0; dim];
        attend(&p, &q, 1, &Memory::new(&p, &k, &v, n, dim, 1), &mut out);
        for c in 0..dim {
            assert!((out[c] - v[2 * dim + c]).abs() < 1e-5);
        }
    }
}
