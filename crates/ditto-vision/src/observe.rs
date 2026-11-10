//! Everything known about one picture, in the forms other programs take: a flat JSON
//! object (the inference engine renders it as a card), one sentence (for its memory and
//! for speech), and the typed record.

use crate::load::{load, Meta};
use crate::stats::{kind, measure, problems, Kind, Stats};
use serde_json::{json, Map, Value};

#[derive(Debug, Clone)]
pub struct Observation {
    /// File name as given, for the reader's benefit only.
    pub file: String,
    pub format: &'static str,
    pub bytes: usize,
    /// As a viewer shows it.
    pub width: u32,
    pub height: u32,
    pub kind: Kind,
    pub meta: Meta,
    pub stats: Stats,
    pub problems: Vec<&'static str>,
    pub qr_codes: Vec<String>,
}

pub fn observe(file: &str, bytes: &[u8]) -> Result<Observation, String> {
    let l = load(bytes)?;
    let stats = measure(&l.image);
    let k = kind(&stats, l.meta.from_camera());
    let (width, height) = l.display_size();
    Ok(Observation {
        file: file.to_string(),
        format: l.format,
        bytes: l.bytes,
        width,
        height,
        kind: k,
        problems: problems(&stats, k),
        qr_codes: crate::codes::qr_codes(&l.image),
        meta: l.meta,
        stats,
    })
}

fn round(x: f64, places: i32) -> f64 {
    let p = 10f64.powi(places);
    (x * p).round() / p
}

impl Observation {
    pub fn megapixels(&self) -> f64 {
        round(self.width as f64 * self.height as f64 / 1e6, 1)
    }

    fn camera(&self) -> Option<String> {
        match (&self.meta.make, &self.meta.model) {
            // many cameras repeat the maker in the model name
            (Some(a), Some(b)) if b.to_lowercase().starts_with(&a.to_lowercase()) => {
                Some(b.clone())
            }
            (Some(a), Some(b)) => Some(format!("{a} {b}")),
            (a, b) => a.clone().or_else(|| b.clone()),
        }
    }

    fn colours(&self) -> String {
        self.stats
            .colours
            .iter()
            .map(|c| format!("{} {:.0}%", c.name, c.share * 100.0))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// A flat object of plain values, absent facts left out, in a fixed key order.
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        let mut put = |k: &str, v: Value| {
            m.insert(k.to_string(), v);
        };
        put("file", json!(self.file));
        put("kind", json!(self.kind.name()));
        put("format", json!(self.format));
        put("width", json!(self.width));
        put("height", json!(self.height));
        put("megapixels", json!(self.megapixels()));
        put("bytes", json!(self.bytes));
        if let Some(t) = &self.meta.taken_at {
            put("taken_at", json!(t));
        }
        if let Some(c) = self.camera() {
            put("camera", json!(c));
        }
        if let (Some(a), Some(b)) = (self.meta.latitude, self.meta.longitude) {
            put("latitude", json!(round(a, 6)));
            put("longitude", json!(round(b, 6)));
        }
        if let Some(a) = self.meta.altitude_m {
            put("altitude_m", json!(round(a, 1)));
        }
        if let Some(v) = self.meta.iso {
            put("iso", json!(v));
        }
        if let Some(v) = self.meta.exposure_s {
            put(
                "exposure",
                json!(if v < 1.0 {
                    format!("1/{:.0} s", 1.0 / v)
                } else {
                    format!("{v:.1} s")
                }),
            );
        }
        if let Some(v) = self.meta.f_number {
            put("aperture", json!(format!("f/{v:.1}")));
        }
        if let Some(v) = self.meta.focal_mm {
            put("focal_length_mm", json!(round(v, 1)));
        }
        put("colours", json!(self.colours()));
        put("sharpness", json!(round(self.stats.sharpness as f64, 1)));
        put("brightness", json!(round(self.stats.brightness as f64, 0)));
        if self.stats.transparent > 0.0 {
            put(
                "transparent_percent",
                json!(round(self.stats.transparent as f64 * 100.0, 1)),
            );
        }
        put(
            "problems",
            json!(if self.problems.is_empty() {
                "none measured".to_string()
            } else {
                self.problems.join(", ")
            }),
        );
        if !self.qr_codes.is_empty() {
            put("qr_codes", json!(self.qr_codes.join(" | ")));
        }
        put(
            "fingerprint",
            json!(format!("{:016x}", self.stats.fingerprint)),
        );
        put("summary", json!(self.summary()));
        Value::Object(m)
    }

    /// One or two sentences. Only what was read or measured.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "{}: {}, {}x{} ({} MP)",
            self.file,
            self.kind.name(),
            self.width,
            self.height,
            self.megapixels()
        );
        if let Some(t) = &self.meta.taken_at {
            s += &format!(", taken {t}");
        }
        if let Some(c) = self.camera() {
            s += &format!(" with {c}");
        }
        if let (Some(a), Some(b)) = (self.meta.latitude, self.meta.longitude) {
            s += &format!(" at {a:.5}, {b:.5}");
        }
        s.push('.');
        if !self.stats.colours.is_empty() {
            s += &format!(" Colours: {}.", self.colours());
        }
        for q in &self.qr_codes {
            s += &format!(" QR code: {q}.");
        }
        if !self.problems.is_empty() {
            s += &format!(" Problems: {}.", self.problems.join(", "));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{jpeg_with_exif, noise, png, Exif};
    use image::{Rgba, RgbaImage};

    #[test]
    fn a_camera_photo() {
        let n = noise(64 * 48 * 3, 2);
        let img = RgbaImage::from_fn(64, 48, |x, y| {
            let i = ((y * 64 + x) * 3) as usize;
            Rgba([n[i] / 4, 128 + n[i + 1] / 2, n[i + 2] / 4, 255])
        });
        let e = Exif {
            make: "Google",
            model: "Pixel 7",
            when: "2026:03:01 18:22:05",
            orientation: 1,
            lat: (17, 23, 6.0, 'N'),
            lon: (78, 29, 12.12, 'E'),
        };
        let o = observe("IMG_1.jpg", &jpeg_with_exif(&img, &e)).unwrap();
        assert_eq!(o.kind, Kind::Photo);
        let j = o.to_json();
        assert_eq!(j["kind"], "photo");
        assert_eq!(j["taken_at"], "2026-03-01 18:22:05");
        assert_eq!(j["camera"], "Google Pixel 7");
        assert_eq!(j["latitude"], 17.385);
        assert_eq!(
            (j["width"].as_u64(), j["height"].as_u64()),
            (Some(64), Some(48))
        );
        assert!(j["colours"].as_str().unwrap().starts_with("green"));
        let s = o.summary();
        assert!(s.starts_with("IMG_1.jpg: photo, 64x48 (0 MP), taken 2026-03-01 18:22:05 with Google Pixel 7 at 17.38500, 78.48670."), "{s}");
        // every value is a plain string or number: the inference engine shows it as fields
        assert!(j
            .as_object()
            .unwrap()
            .values()
            .all(|v| v.is_string() || v.is_number()));
        // fixed key order, the same every time
        let keys: Vec<&String> = j.as_object().unwrap().keys().collect();
        assert_eq!(&keys[..3], &["file", "kind", "format"]);
        assert_eq!(*keys.last().unwrap(), "summary");
    }

    #[test]
    fn a_screenshot_has_no_invented_facts() {
        let mut img = RgbaImage::from_pixel(120, 80, Rgba([255, 255, 255, 255]));
        for x in 10..60 {
            for y in 10..20 {
                img.put_pixel(x, y, Rgba([30, 30, 30, 255]));
            }
        }
        let o = observe("shot.png", &png(&img)).unwrap();
        assert_eq!(o.kind, Kind::ScreenOrGraphic);
        let j = o.to_json();
        for absent in [
            "taken_at",
            "camera",
            "latitude",
            "longitude",
            "iso",
            "qr_codes",
            "transparent_percent",
        ] {
            assert!(j.get(absent).is_none(), "{absent} should be absent");
        }
        assert_eq!(j["problems"], "none measured");
        assert_eq!(
            o.summary(),
            "shot.png: screenshot or graphic, 120x80 (0 MP). Colours: white 95%, black 5%."
        );
    }

    #[test]
    fn the_camera_name_is_not_doubled() {
        let img = RgbaImage::from_pixel(8, 8, Rgba([9, 9, 9, 255]));
        let e = Exif {
            make: "samsung",
            model: "Samsung Galaxy A52",
            when: "2026:01:02 03:04:05",
            orientation: 1,
            lat: (0, 0, 0.0, 'N'),
            lon: (0, 0, 0.0, 'E'),
        };
        let o = observe("a.jpg", &jpeg_with_exif(&img, &e)).unwrap();
        assert_eq!(o.to_json()["camera"], "Samsung Galaxy A52");
    }

    #[test]
    fn errors_name_the_problem() {
        assert!(observe("x.txt", b"plain text")
            .unwrap_err()
            .contains("not a picture"));
    }
}
