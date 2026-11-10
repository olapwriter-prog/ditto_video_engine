//! Decoding a picture and reading what the camera wrote into it.

use image::RgbaImage;
use std::io::Cursor;

/// What the file says about itself (EXIF). Every field is absent unless the file has it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Meta {
    /// `YYYY-MM-DD HH:MM:SS`, in the camera's local time (EXIF records no time zone here).
    pub taken_at: Option<String>,
    pub make: Option<String>,
    pub model: Option<String>,
    /// EXIF orientation, 1 to 8.
    pub orientation: Option<u16>,
    /// Degrees; south and west are negative.
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub altitude_m: Option<f64>,
    pub iso: Option<u32>,
    pub exposure_s: Option<f64>,
    pub f_number: Option<f64>,
    pub focal_mm: Option<f64>,
}

impl Meta {
    /// Written by a camera, as opposed to a screenshot or an exported graphic.
    pub fn from_camera(&self) -> bool {
        self.make.is_some()
            || self.model.is_some()
            || self.taken_at.is_some()
            || self.exposure_s.is_some()
    }
}

#[derive(Debug)]
pub struct Loaded {
    /// `png`, `jpeg`, `webp`, `gif` or `bmp`.
    pub format: &'static str,
    pub bytes: usize,
    /// Pixels as stored. [`Loaded::display_size`] accounts for the orientation.
    pub image: RgbaImage,
    pub meta: Meta,
}

/// Largest picture decoded, in pixels: 100 megapixels. A file can claim any size.
pub const MAX_PIXELS: u64 = 100_000_000;

impl Loaded {
    /// Width and height as a viewer shows the picture.
    pub fn display_size(&self) -> (u32, u32) {
        let (w, h) = self.image.dimensions();
        match self.meta.orientation {
            Some(5..=8) => (h, w),
            _ => (w, h),
        }
    }
}

pub fn load(bytes: &[u8]) -> Result<Loaded, String> {
    let fmt = image::guess_format(bytes)
        .map_err(|_| "not a picture this engine reads (png, jpeg, webp, gif, bmp)".to_string())?;
    let format = match fmt {
        image::ImageFormat::Png => "png",
        image::ImageFormat::Jpeg => "jpeg",
        image::ImageFormat::WebP => "webp",
        image::ImageFormat::Gif => "gif",
        image::ImageFormat::Bmp => "bmp",
        _ => return Err("not a picture this engine reads (png, jpeg, webp, gif, bmp)".into()),
    };
    let mut reader = image::ImageReader::with_format(Cursor::new(bytes), fmt);
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(MAX_PIXELS * 4 + (64 << 20));
    reader.limits(limits);
    let (w, h) = image::ImageReader::with_format(Cursor::new(bytes), fmt)
        .into_dimensions()
        .map_err(|e| format!("damaged {format} file: {e}"))?;
    if w as u64 * h as u64 > MAX_PIXELS {
        return Err(format!(
            "{w}x{h} is larger than the {} megapixels this engine decodes",
            MAX_PIXELS / 1_000_000
        ));
    }
    if w == 0 || h == 0 {
        return Err("the picture has no pixels".into());
    }
    let image = reader
        .decode()
        .map_err(|e| format!("damaged {format} file: {e}"))?
        .to_rgba8();
    Ok(Loaded {
        format,
        bytes: bytes.len(),
        image,
        meta: read_meta(bytes),
    })
}

fn read_meta(bytes: &[u8]) -> Meta {
    use exif::{In, Tag, Value};
    let Ok(ex) = exif::Reader::new().read_from_container(&mut Cursor::new(bytes)) else {
        return Meta::default();
    };
    let text = |tag: Tag| -> Option<String> {
        match &ex.get_field(tag, In::PRIMARY)?.value {
            Value::Ascii(v) => {
                let s = String::from_utf8_lossy(v.first()?)
                    .trim_matches(|c: char| c.is_whitespace() || c == '\0')
                    .to_string();
                (!s.is_empty()).then_some(s)
            }
            _ => None,
        }
    };
    let number = |tag: Tag| -> Option<f64> {
        match &ex.get_field(tag, In::PRIMARY)?.value {
            Value::Rational(v) => v.first().filter(|r| r.denom != 0).map(|r| r.to_f64()),
            Value::SRational(v) => v.first().filter(|r| r.denom != 0).map(|r| r.to_f64()),
            Value::Short(v) => v.first().map(|x| *x as f64),
            Value::Long(v) => v.first().map(|x| *x as f64),
            _ => None,
        }
        .filter(|x| x.is_finite())
    };
    // degrees, minutes, seconds, with the hemisphere in a separate tag
    let angle = |tag: Tag, reference: Tag, negative: &str, limit: f64| -> Option<f64> {
        let Value::Rational(v) = &ex.get_field(tag, In::PRIMARY)?.value else {
            return None;
        };
        if v.len() < 3 || v.iter().take(3).any(|r| r.denom == 0) {
            return None;
        }
        let deg = v[0].to_f64() + v[1].to_f64() / 60.0 + v[2].to_f64() / 3600.0;
        let sign = if text(reference).is_some_and(|r| r.eq_ignore_ascii_case(negative)) {
            -1.0
        } else {
            1.0
        };
        (deg.is_finite() && deg <= limit).then_some(sign * deg)
    };
    let taken_at = text(Tag::DateTimeOriginal)
        .or_else(|| text(Tag::DateTime))
        .and_then(|s| {
            // "2026:03:01 18:22:05" -> "2026-03-01 18:22:05"; anything else is not trusted
            let b = s.as_bytes();
            let ok = b.len() == 19
                && b[4] == b':'
                && b[7] == b':'
                && b[10] == b' '
                && b.iter()
                    .enumerate()
                    .all(|(i, c)| matches!(i, 4 | 7 | 10 | 13 | 16) || c.is_ascii_digit());
            (ok && !s.starts_with("0000"))
                .then(|| format!("{}-{}-{}{}", &s[0..4], &s[5..7], &s[8..10], &s[10..]))
        });
    let (latitude, longitude) = match (
        angle(Tag::GPSLatitude, Tag::GPSLatitudeRef, "S", 90.0),
        angle(Tag::GPSLongitude, Tag::GPSLongitudeRef, "W", 180.0),
    ) {
        // a position needs both, and 0,0 is what cameras write when they have no fix
        (Some(a), Some(b)) if a != 0.0 || b != 0.0 => (Some(a), Some(b)),
        _ => (None, None),
    };
    let below = matches!(ex.get_field(Tag::GPSAltitudeRef, In::PRIMARY).map(|f| &f.value), Some(Value::Byte(v)) if v.first() == Some(&1));
    Meta {
        taken_at,
        make: text(Tag::Make),
        model: text(Tag::Model),
        orientation: number(Tag::Orientation)
            .map(|o| o as u16)
            .filter(|o| (1..=8).contains(o)),
        latitude,
        longitude,
        altitude_m: number(Tag::GPSAltitude)
            .filter(|_| latitude.is_some())
            .map(|a| if below { -a } else { a }),
        iso: number(Tag::PhotographicSensitivity)
            .map(|x| x as u32)
            .filter(|x| *x > 0),
        exposure_s: number(Tag::ExposureTime).filter(|x| *x > 0.0),
        f_number: number(Tag::FNumber).filter(|x| *x > 0.0),
        focal_mm: number(Tag::FocalLength).filter(|x| *x > 0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{jpeg, jpeg_with_exif, png, Exif};
    use image::{Rgba, RgbaImage};

    #[test]
    fn reads_formats_and_sizes() {
        let img = RgbaImage::from_pixel(7, 5, Rgba([10, 200, 30, 255]));
        let l = load(&png(&img)).unwrap();
        assert_eq!(
            (l.format, l.image.dimensions(), l.display_size()),
            ("png", (7, 5), (7, 5))
        );
        assert_eq!(l.meta, Meta::default());
        assert!(!l.meta.from_camera());
        let l = load(&jpeg(&img)).unwrap();
        assert_eq!((l.format, l.image.dimensions()), ("jpeg", (7, 5)));
    }

    #[test]
    fn reads_what_the_camera_wrote() {
        let img = RgbaImage::from_pixel(8, 4, Rgba([90, 90, 90, 255]));
        let e = Exif {
            make: "Google",
            model: "Pixel 7",
            when: "2026:03:01 18:22:05",
            orientation: 6,
            lat: (17, 23, 6.0, 'N'),
            lon: (78, 29, 12.12, 'E'),
        };
        let l = load(&jpeg_with_exif(&img, &e)).unwrap();
        let m = &l.meta;
        assert_eq!(m.make.as_deref(), Some("Google"));
        assert_eq!(m.model.as_deref(), Some("Pixel 7"));
        assert_eq!(m.taken_at.as_deref(), Some("2026-03-01 18:22:05"));
        assert_eq!(m.orientation, Some(6));
        assert!(
            (m.latitude.unwrap() - 17.385).abs() < 1e-6,
            "{:?}",
            m.latitude
        );
        assert!(
            (m.longitude.unwrap() - 78.4867).abs() < 1e-4,
            "{:?}",
            m.longitude
        );
        assert!(m.from_camera());
        // orientation 6 is a quarter turn: shown as 4 wide, 8 high
        assert_eq!(l.display_size(), (4, 8));
    }

    #[test]
    fn south_and_west_are_negative_and_no_fix_is_no_position() {
        let img = RgbaImage::from_pixel(4, 4, Rgba([1, 2, 3, 255]));
        let e = Exif {
            make: "X",
            model: "Y",
            when: "2025:12:31 23:59:59",
            orientation: 1,
            lat: (33, 52, 0.0, 'S'),
            lon: (9, 8, 24.0, 'W'),
        };
        let m = load(&jpeg_with_exif(&img, &e)).unwrap().meta;
        assert!(
            (m.latitude.unwrap() + 33.8667).abs() < 1e-3
                && (m.longitude.unwrap() + 9.14).abs() < 1e-6
        );
        let e = Exif {
            lat: (0, 0, 0.0, 'N'),
            lon: (0, 0, 0.0, 'E'),
            when: "0000:00:00 00:00:00",
            ..e
        };
        let m = load(&jpeg_with_exif(&img, &e)).unwrap().meta;
        assert_eq!((m.latitude, m.longitude, m.taken_at), (None, None, None));
    }

    #[test]
    fn refuses_what_is_not_a_picture() {
        assert!(load(b"hello, this is text")
            .unwrap_err()
            .contains("not a picture"));
        assert!(load(&[]).is_err());
        // a real header with the rest cut off
        let bytes = png(&RgbaImage::from_pixel(64, 64, Rgba([5, 5, 5, 255])));
        assert!(load(&bytes[..40]).unwrap_err().contains("damaged"));
        // MP4 shares nothing with these formats
        assert!(load(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom").is_err());
    }
}
