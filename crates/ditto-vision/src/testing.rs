//! Pictures made by the tests, so no test needs a file or a download.

use image::RgbaImage;
use std::io::Cursor;

pub fn png(img: &RgbaImage) -> Vec<u8> {
    let mut out = Vec::new();
    img.write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
        .unwrap();
    out
}

pub fn jpeg(img: &RgbaImage) -> Vec<u8> {
    let mut out = Vec::new();
    let rgb = image::DynamicImage::ImageRgba8(img.clone()).to_rgb8();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 92)
        .encode_image(&rgb)
        .unwrap();
    out
}

pub struct Exif {
    pub make: &'static str,
    pub model: &'static str,
    /// `YYYY:MM:DD HH:MM:SS`
    pub when: &'static str,
    pub orientation: u16,
    /// degrees, minutes, seconds, hemisphere
    pub lat: (u32, u32, f64, char),
    pub lon: (u32, u32, f64, char),
}

/// A JPEG with a hand-built EXIF block, as a phone camera writes it.
pub fn jpeg_with_exif(img: &RgbaImage, e: &Exif) -> Vec<u8> {
    // TIFF, little-endian. Three directories, then the values that do not fit in 4 bytes.
    let mut t: Vec<u8> = b"II\x2A\x00\x08\x00\x00\x00".to_vec();
    let ifd0 = 8usize;
    let exif_ifd = ifd0 + 2 + 5 * 12 + 4;
    let gps_ifd = exif_ifd + 2 + 12 + 4;
    let mut data_at = gps_ifd + 2 + 4 * 12 + 4;
    let mut data: Vec<u8> = Vec::new();
    let mut put = |bytes: &[u8]| -> u32 {
        let at = data_at as u32;
        data.extend_from_slice(bytes);
        data_at += bytes.len();
        at
    };
    let ascii = |s: &str| -> Vec<u8> { [s.as_bytes(), &[0]].concat() };
    let rationals = |(d, m, s, _): (u32, u32, f64, char)| -> Vec<u8> {
        [d, 1, m, 1, (s * 100.0).round() as u32, 100]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    };
    let entry = |t: &mut Vec<u8>, tag: u16, ty: u16, count: u32, value: [u8; 4]| {
        t.extend_from_slice(&tag.to_le_bytes());
        t.extend_from_slice(&ty.to_le_bytes());
        t.extend_from_slice(&count.to_le_bytes());
        t.extend_from_slice(&value);
    };
    let (make, model, when) = (ascii(e.make), ascii(e.model), ascii(e.when));
    let (make_at, model_at, when_at) = (put(&make), put(&model), put(&when));
    let (lat_at, lon_at) = (put(&rationals(e.lat)), put(&rationals(e.lon)));

    t.extend_from_slice(&5u16.to_le_bytes());
    entry(&mut t, 0x010F, 2, make.len() as u32, make_at.to_le_bytes());
    entry(
        &mut t,
        0x0110,
        2,
        model.len() as u32,
        model_at.to_le_bytes(),
    );
    entry(&mut t, 0x0112, 3, 1, [e.orientation as u8, 0, 0, 0]);
    entry(&mut t, 0x8769, 4, 1, (exif_ifd as u32).to_le_bytes());
    entry(&mut t, 0x8825, 4, 1, (gps_ifd as u32).to_le_bytes());
    t.extend_from_slice(&0u32.to_le_bytes());
    assert_eq!(t.len(), exif_ifd);
    t.extend_from_slice(&1u16.to_le_bytes());
    entry(&mut t, 0x9003, 2, when.len() as u32, when_at.to_le_bytes());
    t.extend_from_slice(&0u32.to_le_bytes());
    assert_eq!(t.len(), gps_ifd);
    t.extend_from_slice(&4u16.to_le_bytes());
    entry(&mut t, 0x0001, 2, 2, [e.lat.3 as u8, 0, 0, 0]);
    entry(&mut t, 0x0002, 5, 3, lat_at.to_le_bytes());
    entry(&mut t, 0x0003, 2, 2, [e.lon.3 as u8, 0, 0, 0]);
    entry(&mut t, 0x0004, 5, 3, lon_at.to_le_bytes());
    t.extend_from_slice(&0u32.to_le_bytes());
    t.extend_from_slice(&data);

    let base = jpeg(img);
    let mut out = base[..2].to_vec(); // start-of-image marker
    out.extend_from_slice(&[0xFF, 0xE1]);
    out.extend_from_slice(&((t.len() + 8) as u16).to_be_bytes());
    out.extend_from_slice(b"Exif\0\0");
    out.extend_from_slice(&t);
    out.extend_from_slice(&base[2..]);
    out
}

/// Deterministic noise in 0..255.
pub fn noise(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 32) as u8
        })
        .collect()
}
