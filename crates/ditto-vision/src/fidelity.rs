//! Did an edit keep the real picture?
//!
//! The most common way a generated picture drifts from reality is not a picture made from
//! nothing, but an edit of a real photo (restore, upscale, remove an object, change the
//! sky) that quietly changes more than it was asked to: a face, a sign, a number plate,
//! the shape of a building. This compares the result with the original, outside the area
//! the edit was allowed to change, and says how much of the real picture survived.
//!
//! It works for edits made by any tool, not only Ditto's. It needs the original.

use crate::stats::luma;
use image::{imageops, RgbaImage};

/// A pixel counts as changed when a channel moved by more than this (of 255). Above the
/// noise of a JPEG re-save at ordinary quality.
pub const CHANGED_ABOVE: u8 = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Outside the allowed area the result is the original, up to compression noise.
    Faithful,
    /// Mostly the original, with visible changes outside the allowed area.
    Drifted,
    /// Not the same picture any more outside the allowed area.
    Different,
}

impl Verdict {
    pub fn name(self) -> &'static str {
        match self {
            Verdict::Faithful => "faithful",
            Verdict::Drifted => "drifted",
            Verdict::Different => "different",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Fidelity {
    /// The result was resized to the original's size before comparing (an upscale).
    pub resized: bool,
    /// Peak signal-to-noise ratio outside the allowed area, in dB. Infinite when identical.
    pub psnr: f64,
    /// Share of pixels outside the allowed area that changed.
    pub changed_outside: f32,
    /// Share of pixels inside the allowed area that changed (`None` without a mask).
    pub changed_inside: Option<f32>,
    /// Share of the picture the edit was allowed to change.
    pub allowed: f32,
    /// Bounding box `(x, y, w, h)` of the changed pixels outside the allowed area.
    pub drift_box: Option<(u32, u32, u32, u32)>,
    pub verdict: Verdict,
}

/// Compare `result` with `original`. `mask`, if given, has the original's size; its light
/// pixels (luma above 127) mark where the edit was allowed to change things.
pub fn compare(
    original: &RgbaImage,
    result: &RgbaImage,
    mask: Option<&RgbaImage>,
) -> Result<Fidelity, String> {
    let (w, h) = original.dimensions();
    if let Some(m) = mask {
        if m.dimensions() != (w, h) {
            return Err(format!(
                "the mask is {}x{}, the original {w}x{h}",
                m.width(),
                m.height()
            ));
        }
    }
    let (rw, rh) = result.dimensions();
    let resized = (rw, rh) != (w, h);
    let scaled;
    let result = if resized {
        // an upscale or a downscale keeps the shape; anything else is a different picture
        let ratio = (rw as f64 / rh as f64) / (w as f64 / h as f64);
        if !(0.99..=1.01).contains(&ratio) {
            return Err(format!("the result is {rw}x{rh}, the original {w}x{h}: a different shape cannot be compared (cropped?)"));
        }
        scaled = imageops::resize(result, w, h, imageops::FilterType::Triangle);
        &scaled
    } else {
        result
    };

    let (mut se, mut n_out, mut ch_out, mut n_in, mut ch_in) =
        (0f64, 0usize, 0usize, 0usize, 0usize);
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
    for y in 0..h {
        for x in 0..w {
            let (a, b) = (original.get_pixel(x, y).0, result.get_pixel(x, y).0);
            let changed = (0..3).any(|k| a[k].abs_diff(b[k]) > CHANGED_ABOVE);
            let allowed = mask.is_some_and(|m| luma(&m.get_pixel(x, y).0) > 127.0);
            if allowed {
                n_in += 1;
                ch_in += changed as usize;
                continue;
            }
            n_out += 1;
            for k in 0..3 {
                se += (a[k] as f64 - b[k] as f64).powi(2);
            }
            if changed {
                ch_out += 1;
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
            }
        }
    }
    let mse = if n_out > 0 {
        se / (3 * n_out) as f64
    } else {
        0.0
    };
    let psnr = if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    };
    let changed_outside = if n_out > 0 {
        ch_out as f32 / n_out as f32
    } else {
        0.0
    };
    let verdict = if changed_outside <= 0.01 && psnr >= 35.0 {
        Verdict::Faithful
    } else if changed_outside <= 0.10 && psnr >= 25.0 {
        Verdict::Drifted
    } else {
        Verdict::Different
    };
    Ok(Fidelity {
        resized,
        psnr,
        changed_outside,
        changed_inside: mask.map(|_| {
            if n_in > 0 {
                ch_in as f32 / n_in as f32
            } else {
                0.0
            }
        }),
        allowed: n_in as f32 / (w * h) as f32,
        drift_box: (ch_out > 0).then(|| (x0, y0, x1 - x0 + 1, y1 - y0 + 1)),
        verdict,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    fn photo() -> RgbaImage {
        RgbaImage::from_fn(200, 120, |x, y| {
            let v = (127.0 + 100.0 * ((x as f32 / 13.0).sin() * (y as f32 / 9.0).cos())) as u8;
            Rgba([v, (x % 256) as u8, (y * 2 % 256) as u8, 255])
        })
    }

    fn mask(x0: u32, y0: u32, x1: u32, y1: u32) -> RgbaImage {
        RgbaImage::from_fn(200, 120, |x, y| {
            if (x0..x1).contains(&x) && (y0..y1).contains(&y) {
                Rgba([255; 4])
            } else {
                Rgba([0, 0, 0, 255])
            }
        })
    }

    fn paint(img: &mut RgbaImage, x0: u32, y0: u32, x1: u32, y1: u32) {
        for y in y0..y1 {
            for x in x0..x1 {
                img.put_pixel(x, y, Rgba([255, 0, 255, 255]));
            }
        }
    }

    #[test]
    fn an_edit_that_stayed_inside_its_area_is_faithful() {
        let mut edited = photo();
        paint(&mut edited, 50, 30, 90, 70);
        let f = compare(&photo(), &edited, Some(&mask(40, 20, 100, 80))).unwrap();
        assert_eq!(f.verdict, Verdict::Faithful);
        assert_eq!(f.psnr, f64::INFINITY);
        assert_eq!(f.drift_box, None);
        assert!((f.allowed - 0.15).abs() < 1e-6); // 60x60 of 200x120
        assert!(f.changed_inside.unwrap() > 0.4);
    }

    #[test]
    fn changes_outside_the_area_are_found_and_located() {
        let mut edited = photo();
        paint(&mut edited, 50, 30, 90, 70); // the asked-for edit
        paint(&mut edited, 150, 100, 160, 110); // something else changed too: 100 px
        let f = compare(&photo(), &edited, Some(&mask(40, 20, 100, 80))).unwrap();
        assert_eq!(f.drift_box, Some((150, 100, 10, 10)));
        assert!(f.changed_outside > 0.0 && f.changed_outside < 0.01);
        assert_eq!(
            f.verdict,
            Verdict::Drifted,
            "few pixels, but large differences: PSNR {}",
            f.psnr
        );
        // without a mask the whole edit counts against the result
        let f = compare(&photo(), &edited, None).unwrap();
        assert_eq!(f.drift_box, Some((50, 30, 110, 80)));
        assert_ne!(f.verdict, Verdict::Faithful);
    }

    #[test]
    fn a_different_picture_is_different() {
        let other = RgbaImage::from_fn(200, 120, |x, y| {
            Rgba([(y * 2) as u8, 255 - (x as u8), 30, 255])
        });
        assert_eq!(
            compare(&photo(), &other, None).unwrap().verdict,
            Verdict::Different
        );
    }

    #[test]
    fn compression_noise_is_not_a_change() {
        let base = photo();
        let noisy = RgbaImage::from_fn(200, 120, |x, y| {
            let p = base.get_pixel(x, y).0;
            let d = ((x * 7 + y * 3) % 9) as i32 - 4; // at most 4 levels
            Rgba([
                (p[0] as i32 + d).clamp(0, 255) as u8,
                p[1],
                (p[2] as i32 - d).clamp(0, 255) as u8,
                255,
            ])
        });
        let f = compare(&photo(), &noisy, None).unwrap();
        assert_eq!(f.verdict, Verdict::Faithful, "PSNR {}", f.psnr);
        assert_eq!(f.changed_outside, 0.0);
    }

    #[test]
    fn an_upscale_is_compared_at_the_original_size_and_crops_are_refused() {
        let up = imageops::resize(&photo(), 400, 240, imageops::FilterType::CatmullRom);
        let f = compare(&photo(), &up, None).unwrap();
        assert!(f.resized);
        assert_ne!(f.verdict, Verdict::Different, "PSNR {}", f.psnr);
        let crop = imageops::crop_imm(&photo(), 0, 0, 100, 100).to_image();
        assert!(compare(&photo(), &crop, None)
            .unwrap_err()
            .contains("different shape"));
        assert!(compare(&photo(), &photo(), Some(&RgbaImage::new(3, 3)))
            .unwrap_err()
            .contains("mask"));
    }
}
