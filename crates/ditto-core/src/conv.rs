//! 2-D convolution on `rie-core`'s quantized matrix kernels.
//!
//! A convolution is a matrix product once every output position's input patch is laid out
//! as a row. Pictures are kept position after position with the channels of a position
//! together (`[y][x][channel]`), so a patch is `k` short copies and every output position
//! is a "token" for [`rie_core::kernels::matmul`]. The same 4-bit and 8-bit weight formats
//! and the same SIMD kernels as the language models then apply unchanged.
//!
//! Needed twice: by Ditto's image decoder, and by the perception models of the driver
//! warning proposal (`docs/DRIVER_ASSIST.md`).

use rie_core::kernels::matmul::{formats_for, matmul, QuantActs};
use rie_core::quant::{quantize, GgmlType};
use rie_core::tensor::Weight;
use rie_core::threadpool::ThreadPool;

/// Output positions laid out and multiplied together. Bounds the patch buffer.
const CHUNK: usize = 1024;

/// A square convolution with "same" padding (`k / 2`) and no bias.
pub struct Conv2d {
    pub in_ch: usize,
    pub out_ch: usize,
    pub k: usize,
    pub stride: usize,
    /// `out_ch × cols`; a row holds `[ky][kx][channel]`, then zeros up to `cols`.
    w: Weight,
    /// Patch length rounded up to the weight format's block size.
    cols: usize,
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

impl Conv2d {
    /// `weights` is `[out_ch][ky][kx][in_ch]`.
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        ty: GgmlType,
        weights: &[f32],
    ) -> Conv2d {
        assert!(k % 2 == 1 && stride >= 1 && in_ch >= 1 && out_ch >= 1);
        let patch = in_ch * k * k;
        assert_eq!(weights.len(), out_ch * patch);
        let cols = patch.div_ceil(32) * 32;
        let mut padded = vec![0.0f32; out_ch * cols];
        for o in 0..out_ch {
            padded[o * cols..o * cols + patch]
                .copy_from_slice(&weights[o * patch..(o + 1) * patch]);
        }
        let data = match ty {
            GgmlType::Q4_0 => quantize::q4_0(&padded),
            GgmlType::Q8_0 => quantize::q8_0(&padded),
            GgmlType::F16 => quantize::f16(&padded),
            GgmlType::F32 => quantize::f32(&padded),
            other => panic!("convolution weights in {} are not supported", other.name()),
        };
        Conv2d {
            in_ch,
            out_ch,
            k,
            stride,
            w: Weight::owned(ty, out_ch, cols, data),
            cols,
        }
    }

    pub fn out_size(&self, h: usize, w: usize) -> (usize, usize) {
        ((h - 1) / self.stride + 1, (w - 1) / self.stride + 1)
    }

    /// Multiply-adds for an `h × w` input.
    pub fn macs(&self, h: usize, w: usize) -> f64 {
        let (oh, ow) = self.out_size(h, w);
        (oh * ow * self.out_ch * self.in_ch * self.k * self.k) as f64
    }

    pub fn size_bytes(&self) -> usize {
        self.w.size_bytes()
    }

    /// `x` is `h × w × in_ch`; returns `oh × ow × out_ch` and its size.
    pub fn forward(
        &self,
        pool: &ThreadPool,
        x: &[f32],
        h: usize,
        w: usize,
    ) -> (Vec<f32>, usize, usize) {
        let (c, k, s, cols) = (self.in_ch, self.k, self.stride, self.cols);
        assert!(h >= 1 && w >= 1 && x.len() >= h * w * c);
        let pad = k / 2;
        let (oh, ow) = self.out_size(h, w);
        let n = oh * ow;
        let mut out = vec![0.0f32; n * self.out_ch];
        let mut patches = vec![0.0f32; CHUNK.min(n) * cols];
        let mut acts = QuantActs::new();
        let formats = formats_for(&[&self.w]);
        let mut t0 = 0;
        while t0 < n {
            let m = CHUNK.min(n - t0);
            let p = Shared(patches.as_mut_ptr());
            pool.for_each(m, 32, |i, _| {
                // SAFETY: each output position owns one row of `patches`.
                let row = unsafe { std::slice::from_raw_parts_mut(p.get().add(i * cols), cols) };
                let (oy, ox) = ((t0 + i) / ow, (t0 + i) % ow);
                for ky in 0..k {
                    let y = (oy * s + ky) as isize - pad as isize;
                    for kx in 0..k {
                        let xx = (ox * s + kx) as isize - pad as isize;
                        let dst = &mut row[(ky * k + kx) * c..(ky * k + kx + 1) * c];
                        if y < 0 || y >= h as isize || xx < 0 || xx >= w as isize {
                            dst.fill(0.0);
                        } else {
                            let src = (y as usize * w + xx as usize) * c;
                            dst.copy_from_slice(&x[src..src + c]);
                        }
                    }
                }
            });
            acts.prepare(pool, &patches, m, cols, &formats);
            matmul(
                pool,
                &acts,
                &self.w,
                &mut out[t0 * self.out_ch..(t0 + m) * self.out_ch],
            );
            t0 += m;
        }
        (out, oh, ow)
    }
}

/// A plain stack of 3×3 convolutions in the size class of a small detector's backbone:
/// a stem, then four stages that each halve the picture and double the channels.
/// It stands in for a real model's cost; it detects nothing.
pub struct Backbone {
    pub layers: Vec<Conv2d>,
}

impl Backbone {
    /// `mult` scales the channel counts (1.0: 32 in the stem, 64 to 512 in the stages).
    pub fn synthetic(mult: f32, ty: GgmlType, seed: u64) -> Backbone {
        let ch = |c: usize| (((c as f32 * mult) / 32.0).round().max(1.0) as usize) * 32;
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut conv = |i: usize, o: usize, stride: usize| {
            let n = o * i * 9;
            let scale = (2.0 / (i * 9) as f32).sqrt();
            let w: Vec<f32> = (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    ((s >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * scale
                })
                .collect();
            Conv2d::new(i, o, 3, stride, ty, &w)
        };
        let mut layers = vec![conv(3, ch(32), 2)];
        let mut prev = ch(32);
        for (c, depth) in [(64, 1), (128, 2), (256, 2), (512, 1)] {
            layers.push(conv(prev, ch(c), 2));
            for _ in 0..depth {
                layers.push(conv(ch(c), ch(c), 1));
            }
            prev = ch(c);
        }
        Backbone { layers }
    }

    pub fn macs(&self, h: usize, w: usize) -> f64 {
        let (mut h, mut w, mut total) = (h, w, 0.0);
        for l in &self.layers {
            total += l.macs(h, w);
            (h, w) = l.out_size(h, w);
        }
        total
    }

    pub fn size_bytes(&self) -> usize {
        self.layers.iter().map(Conv2d::size_bytes).sum()
    }

    /// `x` is `h × w × 3`. Returns the last feature map and its size.
    pub fn forward(
        &self,
        pool: &ThreadPool,
        x: &[f32],
        h: usize,
        w: usize,
    ) -> (Vec<f32>, usize, usize) {
        let (mut cur, mut h, mut w) = (x.to_vec(), h, w);
        for l in &self.layers {
            let (mut y, oh, ow) = l.forward(pool, &cur, h, w);
            let p = Shared(y.as_mut_ptr());
            let len = y.len();
            pool.for_each(len.div_ceil(4096), 1, |i, _| {
                let (a, b) = (i * 4096, ((i + 1) * 4096).min(len));
                // SAFETY: disjoint ranges.
                crate::math::silu(unsafe { std::slice::from_raw_parts_mut(p.get().add(a), b - a) })
            });
            (cur, h, w) = (y, oh, ow);
        }
        (cur, h, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::tests::{noise, pool};

    /// The definition, in f64, straight from the picture.
    fn reference(
        x: &[f32],
        h: usize,
        w: usize,
        c: usize,
        wt: &[f32],
        o: usize,
        k: usize,
        s: usize,
    ) -> (Vec<f64>, usize, usize) {
        let pad = (k / 2) as isize;
        let (oh, ow) = ((h - 1) / s + 1, (w - 1) / s + 1);
        let mut out = vec![0.0f64; oh * ow * o];
        for oy in 0..oh {
            for ox in 0..ow {
                for oc in 0..o {
                    let mut acc = 0.0f64;
                    for ky in 0..k {
                        for kx in 0..k {
                            let (y, xx) =
                                ((oy * s + ky) as isize - pad, (ox * s + kx) as isize - pad);
                            if y < 0 || y >= h as isize || xx < 0 || xx >= w as isize {
                                continue;
                            }
                            for ch in 0..c {
                                acc += x[(y as usize * w + xx as usize) * c + ch] as f64
                                    * wt[((oc * k + ky) * k + kx) * c + ch] as f64;
                            }
                        }
                    }
                    out[(oy * ow + ox) * o + oc] = acc;
                }
            }
        }
        (out, oh, ow)
    }

    fn check(h: usize, w: usize, c: usize, o: usize, k: usize, s: usize, threads: usize) {
        let (x, wt) = (noise(h * w * c, 1), noise(o * c * k * k, 2));
        let conv = Conv2d::new(c, o, k, s, GgmlType::F32, &wt);
        let p = pool(threads);
        let (got, oh, ow) = conv.forward(&p, &x, h, w);
        let (want, rh, rw) = reference(&x, h, w, c, &wt, o, k, s);
        assert_eq!((oh, ow), (rh, rw));
        assert_eq!(conv.macs(h, w), (oh * ow * o * c * k * k) as f64);
        for (i, (g, v)) in got.iter().zip(&want).enumerate() {
            assert!(
                (*g as f64 - v).abs() < 1e-4,
                "{h}x{w} c{c} o{o} k{k} s{s} at {i}: {g} vs {v}"
            );
        }
    }

    #[test]
    fn matches_the_definition() {
        check(5, 7, 3, 4, 3, 1, 1); // patch of 27 padded to 32 columns
        check(6, 6, 3, 8, 3, 2, 2); // stride 2, even size
        check(7, 9, 5, 6, 3, 2, 2); // stride 2, odd sizes
        check(4, 4, 32, 5, 1, 1, 2); // 1x1
        check(9, 5, 2, 3, 5, 1, 3); // 5x5
        check(40, 37, 4, 3, 3, 1, 3); // more positions than one chunk
        check(1, 1, 3, 2, 3, 1, 1); // a single pixel: only the centre tap sees it
    }

    #[test]
    fn quantized_convolution_tracks_the_float_one() {
        let (h, w, c, o) = (12, 12, 32, 16);
        let (x, wt) = (noise(h * w * c, 3), noise(o * c * 9, 4));
        let p = pool(2);
        let run = |ty| Conv2d::new(c, o, 3, 1, ty, &wt).forward(&p, &x, h, w).0;
        let (f, q8, q4) = (run(GgmlType::F32), run(GgmlType::Q8_0), run(GgmlType::Q4_0));
        let rms = |a: &[f32], b: &[f32]| {
            (a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum::<f32>() / a.len() as f32).sqrt()
        };
        let scale = (f.iter().map(|v| v * v).sum::<f32>() / f.len() as f32).sqrt();
        assert!(rms(&f, &q8) / scale < 0.02, "{}", rms(&f, &q8) / scale);
        assert!(rms(&f, &q4) / scale < 0.15, "{}", rms(&f, &q4) / scale);
    }

    #[test]
    fn backbone_shapes_and_cost() {
        let b = Backbone::synthetic(1.0, GgmlType::Q4_0, 1);
        assert_eq!(b.layers.len(), 11);
        // 416x416: about 3.2 G multiply-adds, the class of a small detector
        let m = b.macs(416, 416);
        assert!((3.0e9..3.4e9).contains(&m), "{m}");
        let p = pool(2);
        let (y, h, w) = b.forward(&p, &noise(64 * 64 * 3, 5), 64, 64);
        assert_eq!((h, w, y.len()), (2, 2, 2 * 2 * 512));
        assert!(y.iter().all(|v| v.is_finite()) && y.iter().any(|v| *v != 0.0));
        // half the channels: about a quarter of the work
        let half = Backbone::synthetic(0.5, GgmlType::Q4_0, 1);
        let r = half.macs(416, 416) / m;
        assert!((0.2..0.3).contains(&r), "{r}");
    }
}
