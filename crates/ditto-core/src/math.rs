//! Element-wise float kernels for long vectors.
//!
//! A denoising step applies `exp` hundreds of millions of times (attention over thousands
//! of tokens) and GELU tens of millions of times, so calls into libm would cost as much as
//! the matrix products. These are written so the compiler vectorises them; on x86-64 a
//! copy compiled for AVX2+FMA is chosen at run time, as `rie-core` does for its kernels.

/// Compiles `$body` twice on x86-64 (baseline and AVX2+FMA) and picks at run time.
macro_rules! multiversion {
    ($(#[$m:meta])* pub fn $name:ident($($arg:ident: $ty:ty),*) $(-> $ret:ty)? $body:block) => {
        $(#[$m])*
        #[inline]
        pub fn $name($($arg: $ty),*) $(-> $ret)? {
            #[inline(always)]
            fn imp($($arg: $ty),*) $(-> $ret)? $body
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2,fma")]
                unsafe fn avx2($($arg: $ty),*) $(-> $ret)? { imp($($arg),*) }
                if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                    // SAFETY: the features were just detected.
                    return unsafe { avx2($($arg),*) };
                }
            }
            imp($($arg),*)
        }
    };
}

/// `e^x` to about 2e-7 relative error. Inputs are clamped to ±87, where f32 runs out.
#[inline(always)]
fn exp1(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    // ln 2 split in two so that `x - n·ln2` keeps its low bits
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;
    let x = x.clamp(-87.0, 87.0);
    let n = (x * LOG2E).round();
    let r = (x - n * LN2_HI) - n * LN2_LO;
    // e^r on [-ln2/2, ln2/2]
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0))))));
    let scale = f32::from_bits((((n as i32) + 127) as u32) << 23);
    p * scale
}

multiversion! {
    /// `y += a · x`.
    pub fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
        for (y, x) in y.iter_mut().zip(x) {
            *y += a * *x;
        }
    }
}

multiversion! {
    /// Softmax in place.
    pub fn softmax(x: &mut [f32]) {
        let mut max = f32::NEG_INFINITY;
        for &v in x.iter() {
            max = max.max(v);
        }
        let mut sum = 0.0f32;
        for v in x.iter_mut() {
            *v = exp1(*v - max);
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

multiversion! {
    /// GELU, tanh form (the one diffusion transformers use), in place.
    pub fn gelu_tanh(x: &mut [f32]) {
        const C: f32 = 0.797_884_6; // sqrt(2/π)
        for v in x.iter_mut() {
            let y = C * (*v + 0.044_715 * *v * *v * *v);
            // 0.5·(1 + tanh y) = 1 / (1 + e^(-2y))
            *v /= 1.0 + exp1(-2.0 * y);
        }
    }
}

multiversion! {
    /// Layer norm without learned scale or shift, one row in place.
    pub fn layer_norm(x: &mut [f32]) {
        let n = x.len() as f32;
        let mut mean = 0.0f32;
        for &v in x.iter() {
            mean += v;
        }
        mean /= n;
        let mut var = 0.0f32;
        for &v in x.iter() {
            var += (v - mean) * (v - mean);
        }
        let inv = 1.0 / (var / n + 1e-6).sqrt();
        for v in x.iter_mut() {
            *v = (*v - mean) * inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_matches_libm() {
        let mut worst = 0.0f64;
        let mut x = -80.0f32;
        while x < 80.0 {
            let want = (x as f64).exp();
            let got = exp1(x) as f64;
            worst = worst.max(((got - want) / want).abs());
            x += 0.0137;
        }
        assert!(worst < 5e-7, "worst relative error {worst}");
        assert_eq!(exp1(0.0), 1.0);
        assert!(exp1(-1000.0) >= 0.0 && exp1(-1000.0) < 1e-37);
    }

    #[test]
    fn softmax_sums_to_one_and_matches_reference() {
        let mut x: Vec<f32> = (0..1000)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.21)
            .collect();
        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let e: Vec<f64> = x.iter().map(|&v| (v as f64 - max).exp()).collect();
        let s: f64 = e.iter().sum();
        softmax(&mut x);
        assert!((x.iter().map(|&v| v as f64).sum::<f64>() - 1.0).abs() < 1e-5);
        for (got, want) in x.iter().zip(&e) {
            assert!((*got as f64 - want / s).abs() < 1e-6);
        }
    }

    #[test]
    fn gelu_matches_reference() {
        let mut x: Vec<f32> = (-600..600).map(|i| i as f32 * 0.01).collect();
        let want: Vec<f64> = x
            .iter()
            .map(|&v| {
                let v = v as f64;
                0.5 * v
                    * (1.0
                        + ((2.0 / std::f64::consts::PI).sqrt() * (v + 0.044715 * v * v * v)).tanh())
            })
            .collect();
        gelu_tanh(&mut x);
        for (g, w) in x.iter().zip(&want) {
            assert!((*g as f64 - w).abs() < 2e-6, "{g} vs {w}");
        }
    }

    #[test]
    fn layer_norm_gives_zero_mean_unit_variance() {
        let mut x: Vec<f32> = (0..1536)
            .map(|i| (i as f32 * 0.37).sin() * 3.0 + 2.0)
            .collect();
        layer_norm(&mut x);
        let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
        let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / x.len() as f32;
        assert!(
            mean.abs() < 1e-4 && (var - 1.0).abs() < 1e-3,
            "{mean} {var}"
        );
    }

    #[test]
    fn axpy_adds() {
        let mut y = vec![1.0f32; 37];
        let x: Vec<f32> = (0..37).map(|i| i as f32).collect();
        axpy(&mut y, 0.5, &x);
        for (i, v) in y.iter().enumerate() {
            assert_eq!(*v, 1.0 + 0.5 * i as f32);
        }
    }
}
