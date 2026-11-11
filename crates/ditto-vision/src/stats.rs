//! Measurements of the pixels themselves: sharpness, exposure, colours, a fingerprint,
//! and whether the picture looks like a photograph or like something drawn by a computer.
//!
//! These are measurements with plain thresholds, not judgements by a model. Each verdict
//! says what it measured.

use image::{imageops, RgbaImage};

/// Long side, in pixels, of the copy the measurements are taken on. Fixing it makes the
/// numbers comparable between a 12-megapixel photo and a thumbnail.
const WORK: u32 = 512;

#[derive(Debug, Clone, PartialEq)]
pub struct Colour {
    pub rgb: [u8; 3],
    /// Share of the picture, 0 to 1.
    pub share: f32,
    pub name: &'static str,
}

impl Colour {
    pub fn hex(&self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.rgb[0], self.rgb[1], self.rgb[2])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// The file carries camera data.
    Photo,
    /// No camera data, and the pixels vary like a photograph's.
    PhotoWithoutCameraData,
    /// Large areas of exactly one colour: a screenshot, a diagram, a logo, a scan of text.
    ScreenOrGraphic,
    /// The file says a generative model made it. Takes precedence over everything else.
    Generated,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Photo => "photo",
            Kind::PhotoWithoutCameraData => "photo, no camera data",
            Kind::ScreenOrGraphic => "screenshot or graphic",
            Kind::Generated => "AI-generated, not a photograph",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stats {
    /// Variance of the Laplacian of the brightness: how much fine detail there is.
    pub sharpness: f32,
    /// Mean brightness, 0 to 255.
    pub brightness: f32,
    /// Share of pixels that are nearly black / nearly white.
    pub dark: f32,
    pub bright: f32,
    /// Share of pixels equal to their right-hand neighbour.
    pub flat: f32,
    /// Share of pixels that are not opaque.
    pub transparent: f32,
    /// Most common colours, largest first, at most five.
    pub colours: Vec<Colour>,
    /// 64-bit difference hash. Near-identical pictures differ in few bits.
    pub fingerprint: u64,
}

/// Below this, a photograph has almost no fine detail at the working size.
pub const BLURRY_BELOW: f32 = 40.0;

pub fn luma(p: &[u8; 4]) -> f32 {
    0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32
}

fn name_of(rgb: [u8; 3]) -> &'static str {
    let (r, g, b) = (
        rgb[0] as f32 / 255.0,
        rgb[1] as f32 / 255.0,
        rgb[2] as f32 / 255.0,
    );
    let (max, min) = (r.max(g).max(b), r.min(g).min(b));
    let (v, c) = (max, max - min);
    let s = if max > 0.0 { c / max } else { 0.0 };
    if v < 0.22 {
        return "black";
    }
    if s < 0.12 {
        return if v > 0.88 { "white" } else { "grey" };
    }
    let h = if c == 0.0 {
        0.0
    } else if max == r {
        60.0 * ((g - b) / c).rem_euclid(6.0)
    } else if max == g {
        60.0 * ((b - r) / c + 2.0)
    } else {
        60.0 * ((r - g) / c + 4.0)
    };
    match h {
        h if !(15.0..345.0).contains(&h) => {
            if v > 0.8 && s < 0.45 {
                "pink"
            } else {
                "red"
            }
        }
        h if h < 45.0 => {
            if v < 0.6 {
                "brown"
            } else {
                "orange"
            }
        }
        h if h < 70.0 => "yellow",
        h if h < 165.0 => "green",
        h if h < 195.0 => "teal",
        h if h < 260.0 => "blue",
        h if h < 300.0 => "purple",
        _ => "pink",
    }
}

pub fn measure(img: &RgbaImage) -> Stats {
    let (w, h) = img.dimensions();
    let small = if w.max(h) > WORK {
        let f = WORK as f32 / w.max(h) as f32;
        imageops::thumbnail(
            img,
            ((w as f32 * f).round() as u32).max(1),
            ((h as f32 * f).round() as u32).max(1),
        )
    } else {
        img.clone()
    };
    let (sw, sh) = (small.width() as usize, small.height() as usize);
    let n = (sw * sh) as f32;
    let y: Vec<f32> = small.pixels().map(|p| luma(&p.0)).collect();

    let (mut sum, mut dark, mut bright) = (0.0f64, 0usize, 0usize);
    for &v in &y {
        sum += v as f64;
        dark += (v < 16.0) as usize;
        bright += (v > 239.0) as usize;
    }

    // Laplacian over the interior
    let (mut ls, mut ls2, mut ln) = (0.0f64, 0.0f64, 0usize);
    for yy in 1..sh.saturating_sub(1) {
        for xx in 1..sw.saturating_sub(1) {
            let i = yy * sw + xx;
            let l = (y[i - 1] + y[i + 1] + y[i - sw] + y[i + sw] - 4.0 * y[i]) as f64;
            ls += l;
            ls2 += l * l;
            ln += 1;
        }
    }
    let sharpness = if ln > 0 {
        (ls2 / ln as f64 - (ls / ln as f64).powi(2)).max(0.0) as f32
    } else {
        0.0
    };

    // flatness and transparency are properties of the original pixels, not of a resized copy
    let (mut same, mut pairs, mut clear) = (0usize, 0usize, 0usize);
    let step = ((w as usize * h as usize) / 2_000_000).max(1); // look at every `step`-th row
    for yy in (0..h).step_by(step) {
        for xx in 0..w {
            let p = img.get_pixel(xx, yy).0;
            clear += (p[3] < 255) as usize;
            if xx + 1 < w {
                same += (p == img.get_pixel(xx + 1, yy).0) as usize;
                pairs += 1;
            }
        }
    }
    let rows = (h as usize).div_ceil(step);

    // colours: 8 levels per channel, the mean colour of each occupied cell
    let mut cells = vec![(0u32, [0u64; 3]); 512];
    let mut opaque = 0u32;
    for p in small.pixels() {
        if p.0[3] < 128 {
            continue;
        }
        opaque += 1;
        let c = &mut cells[((p.0[0] >> 5) as usize) << 6
            | ((p.0[1] >> 5) as usize) << 3
            | (p.0[2] >> 5) as usize];
        c.0 += 1;
        for k in 0..3 {
            c.1[k] += p.0[k] as u64;
        }
    }
    cells.sort_by_key(|c| std::cmp::Reverse(c.0));
    let mut colours: Vec<Colour> = Vec::new();
    for (count, s) in cells.iter().filter(|c| c.0 > 0) {
        let rgb = [
            (s[0] / *count as u64) as u8,
            (s[1] / *count as u64) as u8,
            (s[2] / *count as u64) as u8,
        ];
        let share = *count as f32 / opaque.max(1) as f32;
        let name = name_of(rgb);
        // neighbouring cells of one colour are reported once
        if let Some(c) = colours.iter_mut().find(|c| c.name == name) {
            c.share += share;
        } else if colours.len() < 5 {
            colours.push(Colour { rgb, share, name });
        }
    }
    colours.retain(|c| c.share >= 0.03);
    colours.sort_by(|a, b| b.share.total_cmp(&a.share));

    // difference hash: 9x8 brightness, each bit says whether a cell is brighter than the next
    let tiny = imageops::resize(img, 9, 8, imageops::FilterType::Triangle);
    let mut fingerprint = 0u64;
    for yy in 0..8 {
        for xx in 0..8 {
            let (a, b) = (
                luma(&tiny.get_pixel(xx, yy).0),
                luma(&tiny.get_pixel(xx + 1, yy).0),
            );
            fingerprint = fingerprint << 1 | (a > b) as u64;
        }
    }

    Stats {
        sharpness,
        brightness: (sum / n as f64) as f32,
        dark: dark as f32 / n,
        bright: bright as f32 / n,
        flat: if pairs > 0 {
            same as f32 / pairs as f32
        } else {
            1.0
        },
        transparent: clear as f32 / (rows * w as usize).max(1) as f32,
        colours,
        fingerprint,
    }
}

pub fn kind(stats: &Stats, from_camera: bool) -> Kind {
    if from_camera {
        Kind::Photo
    } else if stats.flat >= 0.5 {
        Kind::ScreenOrGraphic
    } else {
        Kind::PhotoWithoutCameraData
    }
}

/// Plain-language problems with a photograph. Empty when nothing was measured to be wrong,
/// and always empty for screenshots and graphics, where "dark" or "flat" is not a fault.
pub fn problems(stats: &Stats, kind: Kind) -> Vec<&'static str> {
    let mut v = Vec::new();
    if matches!(kind, Kind::ScreenOrGraphic | Kind::Generated) {
        return v;
    }
    if stats.sharpness < BLURRY_BELOW {
        v.push("blurry or without detail");
    }
    if stats.brightness < 45.0 || stats.dark > 0.6 {
        v.push("very dark");
    }
    if stats.brightness > 215.0 && stats.bright > 0.5 {
        v.push("overexposed");
    }
    v
}

/// Bits that differ between two fingerprints: 0 identical, up to about 10 the same picture
/// resized or recompressed, around 32 unrelated.
pub fn distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::noise;
    use image::Rgba;

    fn detailed(w: u32, h: u32, seed: u64) -> RgbaImage {
        let n = noise((w * h * 3) as usize, seed);
        RgbaImage::from_fn(w, h, |x, y| {
            let i = ((y * w + x) * 3) as usize;
            Rgba([n[i], n[i + 1], n[i + 2], 255])
        })
    }

    #[test]
    fn names_colours() {
        for (rgb, want) in [
            ([0, 0, 0], "black"),
            ([255, 255, 255], "white"),
            ([128, 128, 128], "grey"),
            ([220, 20, 30], "red"),
            ([255, 140, 0], "orange"),
            ([120, 70, 20], "brown"),
            ([250, 230, 40], "yellow"),
            ([30, 160, 60], "green"),
            ([0, 150, 150], "teal"),
            ([30, 80, 220], "blue"),
            ([130, 40, 200], "purple"),
            ([255, 170, 200], "pink"),
        ] {
            assert_eq!(name_of(rgb), want, "{rgb:?}");
        }
    }

    #[test]
    fn a_flat_picture_and_a_detailed_one() {
        let mut flat = RgbaImage::from_pixel(200, 100, Rgba([250, 250, 250, 255]));
        for x in 20..120 {
            for y in 30..60 {
                flat.put_pixel(x, y, Rgba([20, 90, 220, 255]));
            }
        }
        let s = measure(&flat);
        assert!(s.flat > 0.95, "{}", s.flat);
        assert_eq!(kind(&s, false), Kind::ScreenOrGraphic);
        assert_eq!(s.colours[0].name, "white");
        assert_eq!(s.colours[1].name, "blue");
        assert!(
            (s.colours[1].share - 0.15).abs() < 0.01,
            "{}",
            s.colours[1].share
        );
        assert_eq!(s.colours[1].hex(), "#145adc");
        assert!(problems(&s, kind(&s, false)).is_empty());

        let s = measure(&detailed(200, 100, 3));
        assert!(s.flat < 0.05);
        assert_eq!(kind(&s, false), Kind::PhotoWithoutCameraData);
        assert_eq!(kind(&s, true), Kind::Photo);
        assert!(s.sharpness > BLURRY_BELOW);
    }

    #[test]
    fn blur_and_darkness_are_measured() {
        let sharp = detailed(300, 200, 5);
        let blurred = imageops::blur(&sharp, 6.0);
        let (a, b) = (measure(&sharp), measure(&blurred));
        assert!(
            a.sharpness > 20.0 * b.sharpness,
            "{} vs {}",
            a.sharpness,
            b.sharpness
        );
        assert_eq!(problems(&b, Kind::Photo), vec!["blurry or without detail"]);
        assert!(problems(&a, Kind::Photo).is_empty());

        let dark = RgbaImage::from_fn(100, 100, |x, y| {
            Rgba([(x % 7) as u8, (y % 5) as u8, 3, 255])
        });
        assert!(problems(&measure(&dark), Kind::Photo).contains(&"very dark"));
        let white = RgbaImage::from_fn(100, 100, |x, _| Rgba([250 + (x % 5) as u8, 252, 251, 255]));
        assert!(problems(&measure(&white), Kind::Photo).contains(&"overexposed"));
        // the same white page as a screenshot is not a fault
        assert!(problems(&measure(&white), Kind::ScreenOrGraphic).is_empty());
    }

    #[test]
    fn fingerprint_survives_resizing_and_tells_pictures_apart() {
        // smooth structure, as real pictures have
        let a = RgbaImage::from_fn(320, 240, |x, y| {
            let v = (127.0 + 120.0 * ((x as f32 / 40.0).sin() * (y as f32 / 25.0).cos())) as u8;
            Rgba([v, 255 - v, (x / 2) as u8, 255])
        });
        let b = RgbaImage::from_fn(320, 240, |x, y| {
            let v =
                (127.0 + 120.0 * ((y as f32 / 17.0).sin() + (x as f32 / 90.0).cos()) / 2.0) as u8;
            Rgba([v, v, 255 - v, 255])
        });
        let fa = measure(&a).fingerprint;
        let resized = imageops::resize(&a, 160, 120, imageops::FilterType::Triangle);
        assert!(distance(fa, measure(&resized).fingerprint) <= 6);
        assert!(distance(fa, measure(&b).fingerprint) >= 16);
        assert_eq!(distance(fa, fa), 0);
    }

    #[test]
    fn transparency_is_counted_and_ignored_in_colours() {
        let img = RgbaImage::from_fn(40, 40, |x, _| {
            if x < 10 {
                Rgba([255, 0, 0, 255])
            } else {
                Rgba([0, 0, 0, 0])
            }
        });
        let s = measure(&img);
        assert!((s.transparent - 0.75).abs() < 1e-6);
        assert_eq!(s.colours.len(), 1);
        assert_eq!((s.colours[0].name, s.colours[0].share), ("red", 1.0));
    }
}
