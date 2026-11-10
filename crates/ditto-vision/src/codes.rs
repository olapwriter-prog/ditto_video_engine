//! Machine-readable codes in a picture. QR codes for now.

use crate::stats::luma;
use image::{imageops, RgbaImage};

/// QR codes found, as text, in reading order of detection, without duplicates.
pub fn qr_codes(img: &RgbaImage) -> Vec<String> {
    // The detector's cost grows with the pixel count; phone photos are far larger than a
    // code needs.
    let (w, h) = img.dimensions();
    let scaled;
    let img = if w.max(h) > 1600 {
        let f = 1600.0 / w.max(h) as f32;
        scaled = imageops::thumbnail(img, (w as f32 * f) as u32, (h as f32 * f) as u32);
        &scaled
    } else {
        img
    };
    let (w, h) = (img.width() as usize, img.height() as usize);
    // transparent pixels count as white paper
    let grey: Vec<u8> = img
        .pixels()
        .map(|p| {
            let a = p.0[3] as f32 / 255.0;
            (luma(&p.0) * a + 255.0 * (1.0 - a)) as u8
        })
        .collect();
    let mut prepared = rqrr::PreparedImage::prepare_from_greyscale(w, h, |x, y| grey[y * w + x]);
    let mut out: Vec<String> = Vec::new();
    for grid in prepared.detect_grids() {
        if let Ok((_, text)) = grid.decode() {
            if !text.is_empty() && !out.contains(&text) {
                out.push(text);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    pub fn qr_image(text: &str, scale: u32) -> RgbaImage {
        let code = qrcode::QrCode::new(text.as_bytes()).unwrap();
        let n = code.width() as u32;
        let colors = code.to_colors();
        let quiet = 4;
        RgbaImage::from_fn((n + 2 * quiet) * scale, (n + 2 * quiet) * scale, |x, y| {
            let (cx, cy) = (x / scale, y / scale);
            let dark = cx >= quiet
                && cy >= quiet
                && cx < n + quiet
                && cy < n + quiet
                && colors[((cy - quiet) * n + (cx - quiet)) as usize] == qrcode::Color::Dark;
            if dark {
                Rgba([0, 0, 0, 255])
            } else {
                Rgba([255, 255, 255, 255])
            }
        })
    }

    #[test]
    fn reads_a_code() {
        let url = "upi://pay?pa=shop@bank&pn=Kirana%20Store&am=120.00";
        assert_eq!(qr_codes(&qr_image(url, 6)), vec![url.to_string()]);
    }

    #[test]
    fn reads_a_code_inside_a_larger_picture() {
        let code = qr_image("https://example.org/menu", 5);
        let mut page = RgbaImage::from_pixel(900, 700, Rgba([235, 232, 225, 255]));
        imageops::overlay(&mut page, &code, 430, 260);
        assert_eq!(
            qr_codes(&page),
            vec!["https://example.org/menu".to_string()]
        );
    }

    #[test]
    fn finds_nothing_where_there_is_nothing() {
        assert!(qr_codes(&RgbaImage::from_pixel(300, 200, Rgba([200, 200, 200, 255]))).is_empty());
        let stripes = RgbaImage::from_fn(300, 200, |x, _| {
            if x / 8 % 2 == 0 {
                Rgba([0, 0, 0, 255])
            } else {
                Rgba([255, 255, 255, 255])
            }
        });
        assert!(qr_codes(&stripes).is_empty());
    }
}
