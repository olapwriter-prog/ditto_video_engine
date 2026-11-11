//! Labels that say a picture was generated, and the checks that read them.
//!
//! The rule this module serves: **a generated picture must never pass for a photograph.**
//! Every picture Ditto makes carries two labels inside the PNG file:
//!
//! * an XMP packet with the IPTC digital source type `trainedAlgorithmicMedia`, the
//!   industry's standard marker for AI-made media, which other apps and services read;
//! * a Ditto record: that it is AI-generated, the prompt, seed and model, a SHA-256 of the
//!   pixels, and an Ed25519 signature over all of it by the device's key.
//!
//! Reading a picture tells apart: generated and untouched; generated, then edited (the
//! pixels no longer match the record); a record whose text was changed (the signature
//! fails); a marker written by another generator; and no marker at all.
//!
//! **No marker does not mean real.** Labels inside a file are lost when a picture is
//! screenshotted, converted to JPEG by a messenger, or stripped on purpose. A visible badge
//! ([`badge`]) survives more of that, and the reader is always told what was *not* checked.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use image::{Rgba, RgbaImage};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

/// PNG text keyword of Ditto's record.
pub const KEYWORD: &str = "ditto-provenance";
const XMP_KEYWORD: &str = "XML:com.adobe.xmp";
/// IPTC digital source types meaning "made by a generative model".
const IPTC_AI: &str = "trainedAlgorithmicMedia";
const IPTC_AI_COMPOSITE: &str = "compositeWithTrainedAlgorithmicMedia";
const XMP: &str = "<?xpacket begin=\"\u{feff}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\
<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\
<rdf:Description rdf:about=\"\" xmlns:Iptc4xmpExt=\"http://iptc.org/std/Iptc4xmpExt/2008-02-29/\" \
Iptc4xmpExt:DigitalSourceType=\"http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia\"/>\
</rdf:RDF></x:xmpmeta><?xpacket end=\"r\"?>";

/// What a picture was made from and how.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    pub prompt: String,
    pub seed: u64,
    /// Model file name or hash.
    pub model: String,
    /// `text`, or what a real picture was changed into, such as `edit of <sha256>`.
    pub source: String,
    /// `YYYY-MM-DD HH:MM:SS UTC`.
    pub created: String,
}

/// SHA-256 over the size and the RGBA pixels: the same for any lossless re-save.
pub fn pixel_hash(img: &RgbaImage) -> String {
    let mut h = Sha256::new();
    h.update(img.width().to_le_bytes());
    h.update(img.height().to_le_bytes());
    h.update(img.as_raw());
    hex(&h.finalize())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok())
        .collect()
}

/// A device key from 32 random bytes.
pub fn key_from_bytes(b: &[u8; 32]) -> SigningKey {
    SigningKey::from_bytes(b)
}

// ---- PNG chunks ------------------------------------------------------------------------

const PNG_SIG: &[u8; 8] = b"\x89PNG\r\n\x1a\n";

fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
    }
    !c
}

/// (type, data) of every chunk, or `None` if the file is not a well-formed PNG.
fn chunks(png: &[u8]) -> Option<Vec<([u8; 4], &[u8])>> {
    if !png.starts_with(PNG_SIG) {
        return None;
    }
    let mut out = Vec::new();
    let mut i = 8;
    while i + 12 <= png.len() {
        let len = u32::from_be_bytes(png[i..i + 4].try_into().ok()?) as usize;
        let ty: [u8; 4] = png[i + 4..i + 8].try_into().ok()?;
        let end = i.checked_add(12)?.checked_add(len)?;
        if end > png.len() {
            return None;
        }
        out.push((ty, &png[i + 8..i + 8 + len]));
        i = end;
        if &ty == b"IEND" {
            return Some(out);
        }
    }
    None
}

fn write_chunk(out: &mut Vec<u8>, ty: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(ty);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// An uncompressed international text chunk.
fn itxt(keyword: &str, text: &str) -> Vec<u8> {
    let mut d = keyword.as_bytes().to_vec();
    d.extend_from_slice(&[0, 0, 0, 0, 0]); // end of keyword, not compressed, method, no language, no translation
    d.extend_from_slice(text.as_bytes());
    d
}

/// (keyword, text) of an uncompressed iTXt or tEXt chunk.
fn text_of<'a>(ty: &[u8; 4], data: &'a [u8]) -> Option<(&'a str, &'a str)> {
    let k = data.iter().position(|&b| b == 0)?;
    let keyword = std::str::from_utf8(&data[..k]).ok()?;
    match ty {
        b"tEXt" => Some((keyword, std::str::from_utf8(&data[k + 1..]).ok()?)),
        b"iTXt" => {
            let rest = data.get(k + 1..)?;
            if rest.first() != Some(&0) {
                return None; // compressed: Ditto never writes these
            }
            let rest = rest.get(2..)?;
            let lang_end = rest.iter().position(|&b| b == 0)?;
            let rest = &rest[lang_end + 1..];
            let tr_end = rest.iter().position(|&b| b == 0)?;
            Some((keyword, std::str::from_utf8(&rest[tr_end + 1..]).ok()?))
        }
        _ => None,
    }
}

// ---- labelling -------------------------------------------------------------------------

fn decode_png(png: &[u8]) -> Result<RgbaImage, String> {
    image::load_from_memory_with_format(png, image::ImageFormat::Png)
        .map(|i| i.to_rgba8())
        .map_err(|e| format!("not a readable PNG: {e}"))
}

/// The record as signed: fixed key order, no signature.
fn payload(rec: &Record, w: u32, h: u32, pixels: &str, key: &VerifyingKey) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("ai_generated".into(), json!(true));
    m.insert(
        "generator".into(),
        json!(format!("ditto {}", env!("CARGO_PKG_VERSION"))),
    );
    m.insert("prompt".into(), json!(rec.prompt));
    m.insert("seed".into(), json!(rec.seed));
    m.insert("model".into(), json!(rec.model));
    m.insert("source".into(), json!(rec.source));
    m.insert("created".into(), json!(rec.created));
    m.insert("width".into(), json!(w));
    m.insert("height".into(), json!(h));
    m.insert("pixels_sha256".into(), json!(pixels));
    m.insert("public_key".into(), json!(hex(key.as_bytes())));
    m
}

/// Add both labels to a PNG. Labels already in it (from an earlier run) are replaced; the
/// pixels are not touched.
pub fn label_png(png: &[u8], rec: &Record, key: &SigningKey) -> Result<Vec<u8>, String> {
    let img = decode_png(png)?;
    let list = chunks(png).ok_or("not a well-formed PNG")?;
    let mut m = payload(
        rec,
        img.width(),
        img.height(),
        &pixel_hash(&img),
        &key.verifying_key(),
    );
    let signed = Value::Object(m.clone()).to_string();
    m.insert(
        "signature".into(),
        json!(hex(&key.sign(signed.as_bytes()).to_bytes())),
    );
    let record = Value::Object(m).to_string();

    let mut out = PNG_SIG.to_vec();
    for (ty, data) in list {
        let ours = matches!(text_of(&ty, data), Some((k, _)) if k == KEYWORD || k == XMP_KEYWORD);
        if ours {
            continue;
        }
        if &ty == b"IEND" {
            write_chunk(&mut out, b"iTXt", &itxt(XMP_KEYWORD, XMP));
            write_chunk(&mut out, b"iTXt", &itxt(KEYWORD, &record));
        }
        write_chunk(&mut out, &ty, data);
    }
    Ok(out)
}

/// Draw a small "AI" mark in the bottom-right corner, white on dark, scaled to the picture.
/// It survives screenshots and messengers, which file labels do not.
pub fn badge(img: &mut RgbaImage) {
    // 5x7 glyphs, one row per line, most significant bit on the left
    const A: [u8; 7] = [
        0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
    ];
    const I: [u8; 7] = [
        0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111,
    ];
    let (w, h) = img.dimensions();
    let s = (w.min(h) / 200).max(2); // one glyph pixel, in picture pixels: about 5% of the short side
    let (bw, bh) = (15 * s, 11 * s); // 2 glyphs + gaps and margins
    if w < bw + 2 * s || h < bh + 2 * s {
        return; // too small to carry a legible mark
    }
    let (x0, y0) = (w - bw - 2 * s, h - bh - 2 * s);
    for y in 0..bh {
        for x in 0..bw {
            img.put_pixel(x0 + x, y0 + y, Rgba([20, 20, 20, 255]));
        }
    }
    for (gi, glyph) in [A, I].iter().enumerate() {
        let gx = x0 + 2 * s + gi as u32 * 6 * s;
        for (row, bits) in glyph.iter().enumerate() {
            for col in 0..5 {
                if bits >> (4 - col) & 1 == 1 {
                    for dy in 0..s {
                        for dx in 0..s {
                            img.put_pixel(
                                gx + col * s + dx,
                                y0 + 2 * s + row as u32 * s + dy,
                                Rgba([255, 255, 255, 255]),
                            );
                        }
                    }
                }
            }
        }
    }
}

// ---- reading ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelStatus {
    /// Signature valid and the pixels are the ones that were labelled.
    Intact,
    /// Signature valid, but the pixels changed after labelling: edited after generation.
    PixelsChanged,
    /// The record's text was altered, or it was not signed by the key it names.
    Forged,
    /// A Ditto record that cannot be parsed.
    Damaged,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DittoLabel {
    pub status: LabelStatus,
    /// The record's fields, as written (empty when damaged).
    pub record: Map<String, Value>,
}

/// Everything a file says about how it was made.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Provenance {
    pub ditto: Option<DittoLabel>,
    /// An IPTC marker for AI-made media, written by any generator (Ditto included).
    pub iptc_ai: bool,
    /// A C2PA content-credentials manifest is present. Its signature is **not** checked here.
    pub c2pa: bool,
}

impl Provenance {
    /// The file says it was made, wholly or partly, by a generative model.
    pub fn says_generated(&self) -> bool {
        self.iptc_ai
            || self
                .ditto
                .as_ref()
                .is_some_and(|d| d.status != LabelStatus::Damaged)
    }

    /// One plain sentence, including what this check cannot tell.
    pub fn describe(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(d) = &self.ditto {
            let prompt = d.record.get("prompt").and_then(Value::as_str).unwrap_or("");
            parts.push(match d.status {
                LabelStatus::Intact => format!("Generated by Ditto from \"{prompt}\"; label signed and the pixels unchanged since"),
                LabelStatus::PixelsChanged => format!("Generated by Ditto from \"{prompt}\", then edited: the pixels differ from the labelled ones"),
                LabelStatus::Forged => "Carries a Ditto label whose text was altered or wrongly signed; treat it as generated, details untrusted".into(),
                LabelStatus::Damaged => "Carries a damaged Ditto label".into(),
            });
        }
        if self.iptc_ai && self.ditto.is_none() {
            parts.push("Marked by its maker as AI-generated (IPTC)".into());
        }
        if self.c2pa {
            parts.push("Has a C2PA content-credentials manifest, not verified here".into());
        }
        if parts.is_empty() {
            return "No generation marker found. This does not show the picture is real: markers are lost in screenshots, conversions and on purpose".into();
        }
        parts.join(". ")
    }
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

/// Read the labels of a file of any supported format. `img` is its decoded picture.
pub fn inspect(bytes: &[u8], img: &RgbaImage) -> Provenance {
    let mut p = Provenance {
        iptc_ai: contains(bytes, IPTC_AI.as_bytes())
            || contains(bytes, IPTC_AI_COMPOSITE.as_bytes()),
        // a JUMBF box labelled "c2pa", as content credentials are stored
        c2pa: contains(bytes, b"jumb") && contains(bytes, b"c2pa"),
        ditto: None,
    };
    let Some(list) = chunks(bytes) else { return p };
    for (ty, data) in list {
        let Some((k, text)) = text_of(&ty, data) else {
            continue;
        };
        if k != KEYWORD {
            continue;
        }
        p.ditto = Some(check_record(text, img));
        break;
    }
    p
}

fn check_record(text: &str, img: &RgbaImage) -> DittoLabel {
    let damaged = DittoLabel {
        status: LabelStatus::Damaged,
        record: Map::new(),
    };
    let Ok(Value::Object(mut m)) = serde_json::from_str::<Value>(text) else {
        return damaged;
    };
    let Some(sig) = m
        .remove("signature")
        .and_then(|v| v.as_str().and_then(unhex))
    else {
        return DittoLabel {
            status: LabelStatus::Forged,
            record: m,
        };
    };
    let key = m
        .get("public_key")
        .and_then(Value::as_str)
        .and_then(unhex)
        .and_then(|b| <[u8; 32]>::try_from(b).ok())
        .and_then(|b| VerifyingKey::from_bytes(&b).ok());
    let sig = <[u8; 64]>::try_from(sig)
        .ok()
        .map(|b| Signature::from_bytes(&b));
    let signed = Value::Object(m.clone()).to_string();
    let valid = matches!((key, sig), (Some(k), Some(s)) if k.verify(signed.as_bytes(), &s).is_ok());
    if !valid {
        return DittoLabel {
            status: LabelStatus::Forged,
            record: m,
        };
    }
    let same = m.get("pixels_sha256").and_then(Value::as_str) == Some(pixel_hash(img).as_str());
    DittoLabel {
        status: if same {
            LabelStatus::Intact
        } else {
            LabelStatus::PixelsChanged
        },
        record: m,
    }
}

/// The device's signing key, created on first use: 32 bytes from the system's random
/// source, readable by the owner only.
pub fn device_key(path: &std::path::Path) -> Result<SigningKey, String> {
    use std::io::Read;
    if let Ok(b) = std::fs::read(path) {
        let b: [u8; 32] = b
            .try_into()
            .map_err(|_| format!("{}: not a Ditto key (32 bytes)", path.display()))?;
        return Ok(key_from_bytes(&b));
    }
    let mut b = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut b))
        .map_err(|e| format!("no random source to make a key: {e}"))?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut opts, 0o600);
    std::io::Write::write_all(
        &mut opts
            .open(path)
            .map_err(|e| format!("{}: {e}", path.display()))?,
        &b,
    )
    .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(key_from_bytes(&b))
}

/// Now, as `YYYY-MM-DD HH:MM:SS UTC`.
pub fn now_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs()) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // civil date from days since 1970-01-01 (Howard Hinnant's algorithm)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02} UTC",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{jpeg, png};

    fn key(n: u8) -> SigningKey {
        key_from_bytes(&[n; 32])
    }

    fn rec() -> Record {
        Record {
            prompt: "a paper boat on a river".into(),
            seed: 42,
            model: "test-model".into(),
            source: "text".into(),
            created: "2026-11-11 02:00:00 UTC".into(),
        }
    }

    fn picture() -> RgbaImage {
        RgbaImage::from_fn(64, 48, |x, y| Rgba([(x * 4) as u8, (y * 5) as u8, 90, 255]))
    }

    fn read(bytes: &[u8]) -> Provenance {
        let img = image::load_from_memory(bytes).unwrap().to_rgba8();
        inspect(bytes, &img)
    }

    #[test]
    fn a_labelled_picture_reads_back_intact() {
        let labelled = label_png(&png(&picture()), &rec(), &key(1)).unwrap();
        let p = read(&labelled);
        let d = p.ditto.as_ref().unwrap();
        assert_eq!(d.status, LabelStatus::Intact);
        assert_eq!(d.record["prompt"], "a paper boat on a river");
        assert_eq!(d.record["ai_generated"], true);
        assert!(p.iptc_ai, "the standard marker is written too");
        assert!(p.says_generated());
        assert!(p
            .describe()
            .starts_with("Generated by Ditto from \"a paper boat on a river\"; label signed"));
        // the pixels are untouched
        assert_eq!(
            image::load_from_memory(&labelled).unwrap().to_rgba8(),
            picture()
        );
        // and the file is still a valid PNG for the decoder, chunk CRCs included
        assert!(chunks(&labelled).is_some());
    }

    #[test]
    fn edits_after_generation_are_seen() {
        let labelled = label_png(&png(&picture()), &rec(), &key(1)).unwrap();
        // change one pixel, keep the chunks: what an editor that preserves metadata does
        let mut img = image::load_from_memory(&labelled).unwrap().to_rgba8();
        img.put_pixel(3, 3, Rgba([0, 0, 0, 255]));
        let edited_pixels = png(&img);
        let list = chunks(&labelled).unwrap();
        let mut edited = PNG_SIG.to_vec();
        for (ty, data) in chunks(&edited_pixels).unwrap() {
            if &ty == b"IEND" {
                for (t2, d2) in &list {
                    if t2 == b"iTXt" {
                        write_chunk(&mut edited, t2, d2);
                    }
                }
            }
            write_chunk(&mut edited, &ty, data);
        }
        let p = read(&edited);
        assert_eq!(p.ditto.unwrap().status, LabelStatus::PixelsChanged);
    }

    #[test]
    fn a_changed_record_or_wrong_key_is_forged() {
        let labelled = label_png(&png(&picture()), &rec(), &key(1)).unwrap();
        // change the prompt inside the file without re-signing, keeping the length
        let s = labelled.clone();
        let at = s.windows(5).position(|w| w == b"paper").unwrap();
        let mut forged = s.clone();
        forged[at..at + 5].copy_from_slice(b"stone");
        // fix the chunk CRC so only the signature can catch it
        let fixed = {
            let list: Vec<([u8; 4], Vec<u8>)> = {
                // CRC is wrong now; parse by lengths only
                let mut v = Vec::new();
                let mut i = 8;
                while i < forged.len() {
                    let len = u32::from_be_bytes(forged[i..i + 4].try_into().unwrap()) as usize;
                    let ty: [u8; 4] = forged[i + 4..i + 8].try_into().unwrap();
                    v.push((ty, forged[i + 8..i + 8 + len].to_vec()));
                    i += 12 + len;
                }
                v
            };
            let mut out = PNG_SIG.to_vec();
            for (ty, d) in list {
                write_chunk(&mut out, &ty, &d);
            }
            out
        };
        let p = read(&fixed);
        assert_eq!(p.ditto.as_ref().unwrap().status, LabelStatus::Forged);
        assert!(
            p.says_generated(),
            "a tampered label still counts as generated"
        );
        // labelling again with another key replaces the label; it verifies under that key
        let relabelled = label_png(&labelled, &rec(), &key(2)).unwrap();
        let p = read(&relabelled);
        assert_eq!(p.ditto.as_ref().unwrap().status, LabelStatus::Intact);
        assert_eq!(
            p.ditto.unwrap().record["public_key"],
            hex(key(2).verifying_key().as_bytes())
        );
        // only one record and one XMP packet after relabelling
        let n = chunks(&relabelled)
            .unwrap()
            .iter()
            .filter(|(t, _)| t == b"iTXt")
            .count();
        assert_eq!(n, 2);
    }

    #[test]
    fn unlabelled_pictures_say_what_is_unknown() {
        let p = read(&png(&picture()));
        assert_eq!(p, Provenance::default());
        assert!(!p.says_generated());
        assert!(p.describe().contains("does not show the picture is real"));
        // a label is lost when the picture is converted to JPEG: the reader must not be told otherwise
        let labelled = label_png(&png(&picture()), &rec(), &key(1)).unwrap();
        let as_jpeg = jpeg(&image::load_from_memory(&labelled).unwrap().to_rgba8());
        assert!(!read(&as_jpeg).says_generated());
    }

    #[test]
    fn markers_from_other_generators() {
        // an IPTC marker in a JPEG's XMP, as other generators write it
        let mut j = jpeg(&picture());
        let xmp = format!("http://ns.adobe.com/xap/1.0/\0{XMP}");
        let mut seg = vec![0xFF, 0xE1];
        seg.extend_from_slice(&((xmp.len() + 2) as u16).to_be_bytes());
        seg.extend_from_slice(xmp.as_bytes());
        j.splice(2..2, seg);
        let p = read(&j);
        assert!(p.iptc_ai && p.ditto.is_none());
        assert_eq!(p.describe(), "Marked by its maker as AI-generated (IPTC)");
        // a damaged Ditto record
        let mut out = PNG_SIG.to_vec();
        for (ty, d) in chunks(&png(&picture())).unwrap() {
            if &ty == b"IEND" {
                write_chunk(&mut out, b"iTXt", &itxt(KEYWORD, "{not json"));
            }
            write_chunk(&mut out, &ty, d);
        }
        assert_eq!(read(&out).ditto.unwrap().status, LabelStatus::Damaged);
    }

    #[test]
    fn the_badge_is_drawn_where_it_fits() {
        let mut big = RgbaImage::from_pixel(640, 480, Rgba([200, 200, 200, 255]));
        badge(&mut big);
        // scale 2: a dark box in the corner with white strokes inside
        assert_eq!(
            *big.get_pixel(640 - 6 - 1, 480 - 6 - 1),
            Rgba([20, 20, 20, 255])
        );
        let white = big.pixels().filter(|p| p.0 == [255, 255, 255, 255]).count();
        assert_eq!(
            white,
            (18 + 15) * 4,
            "the strokes of A (18 cells) and I (15), 2x2 pixels each"
        );
        assert_eq!(*big.get_pixel(0, 0), Rgba([200, 200, 200, 255]));
        let mut tiny = RgbaImage::from_pixel(10, 10, Rgba([1, 2, 3, 255]));
        badge(&mut tiny);
        assert!(tiny.pixels().all(|p| p.0 == [1, 2, 3, 255]));
    }

    #[test]
    fn crc_matches_the_png_reference_value() {
        assert_eq!(crc32(b"IEND"), 0xAE42_6082);
    }
}
