//! `ditto look` and `ditto trace`: pictures in, facts and SVG out, and the hand-over of
//! those facts to the inference engine.

use ditto_vision::{observe, trace, Observation};
use rie_core::memory::{Kind, MemoryStore};
use serde_json::json;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The name under which observations are filed in the inference engine's memory.
pub const MEMORY_TOOL: &str = "ditto.look";
/// Observations kept by a memory directory this command opens itself (the server's default).
const MEMORY_CAPACITY: usize = 4096;
/// Largest file read: more than any phone photo, far less than a video.
const MAX_FILE: u64 = 64 << 20;

struct Opts {
    json: bool,
    rie: Option<String>,
    memory: Option<PathBuf>,
    colours: usize,
    out: Option<PathBuf>,
    force: bool,
    files: Vec<String>,
}

fn parse(argv: &[String]) -> Result<Opts, String> {
    let mut o = Opts {
        json: false,
        rie: None,
        memory: None,
        colours: 16,
        out: None,
        force: false,
        files: Vec::new(),
    };
    let mut it = argv.iter();
    let mut only_files = false;
    while let Some(a) = it.next() {
        if only_files || !a.starts_with('-') || a == "-" {
            o.files.push(a.clone());
            continue;
        }
        let mut val = || it.next().cloned().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--" => only_files = true,
            "--json" => o.json = true,
            "--force" => o.force = true,
            "--rie" => o.rie = Some(val()?),
            "--memory" => o.memory = Some(PathBuf::from(val()?)),
            "-o" | "--out" => o.out = Some(PathBuf::from(val()?)),
            "--colours" | "--colors" => {
                let v = val()?;
                o.colours = v
                    .parse::<usize>()
                    .ok()
                    .filter(|n| (2..=256).contains(n))
                    .ok_or(format!("--colours: '{v}' is not a number from 2 to 256"))?;
            }
            other => return Err(format!("unknown option '{other}'")),
        }
    }
    if o.files.is_empty() {
        return Err("no picture given".into());
    }
    Ok(o)
}

fn read(path: &str) -> Result<Vec<u8>, String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("{path}: {e}"))?;
    if !meta.is_file() {
        return Err(format!("{path}: not a file"));
    }
    if meta.len() > MAX_FILE {
        return Err(format!(
            "{path}: {} MB is larger than the {} MB this command reads",
            meta.len() >> 20,
            MAX_FILE >> 20
        ));
    }
    std::fs::read(path).map_err(|e| format!("{path}: {e}"))
}

fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map_or_else(|| path.to_string(), |n| n.to_string_lossy().into_owned())
}

/// What goes into the inference engine's memory for one picture: the sentence, and where
/// the picture is, so a later question can lead back to the file.
pub fn memory_text(o: &Observation, path: &str) -> String {
    let full =
        std::fs::canonicalize(path).map_or_else(|_| path.to_string(), |p| p.display().to_string());
    if o.generated() {
        // never stored as evidence: the language model must not treat it as a photo
        format!(
            "Generated picture, not a photograph, not evidence of anything real. {} File: {full}",
            o.summary()
        )
    } else {
        format!("Picture {} File: {full}", o.summary())
    }
}

/// `POST /v1/memory` on a running `rie-server`. Plain HTTP to a local address only: the
/// text holds where the user was and what they photographed.
pub fn post_memory(url: &str, text: &str) -> Result<String, String> {
    let rest = url
        .strip_prefix("http://")
        .ok_or("--rie takes an http:// address, such as http://127.0.0.1:8080")?;
    let host = rest.split('/').next().unwrap_or("");
    let name = host.rsplit_once(':').map_or(host, |(h, _)| h);
    if !matches!(name, "127.0.0.1" | "localhost" | "[::1]") {
        return Err(format!("--rie: {name} is not this device; observations are only sent to a local inference engine"));
    }
    let addr = if host.contains(':') {
        host.to_string()
    } else {
        format!("{host}:80")
    };
    let body =
        json!({"kind": "note", "tool": MEMORY_TOOL, "ok": true, "importance": 0.6, "text": text})
            .to_string();
    let mut s =
        TcpStream::connect(&addr).map_err(|e| format!("inference engine at {addr}: {e}"))?;
    s.set_read_timeout(Some(Duration::from_secs(10))).ok();
    s.set_write_timeout(Some(Duration::from_secs(10))).ok();
    write!(s, "POST /v1/memory HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
        .map_err(|e| format!("inference engine at {addr}: {e}"))?;
    let mut reply = String::new();
    s.take(64 * 1024)
        .read_to_string(&mut reply)
        .map_err(|e| format!("inference engine at {addr}: {e}"))?;
    let status = reply.lines().next().unwrap_or("").to_string();
    let answer = reply
        .split_once("\r\n\r\n")
        .map_or("", |x| x.1)
        .trim()
        .to_string();
    if !status.contains(" 200") {
        return Err(format!(
            "inference engine at {addr} answered '{status}' {answer}"
        ));
    }
    Ok(answer)
}

pub fn cmd_look(argv: &[String]) -> Result<(), String> {
    let o = parse(argv)?;
    if o.out.is_some() || o.force {
        return Err("-o and --force belong to `ditto trace`".into());
    }
    let mut store = match &o.memory {
        Some(dir) => Some(
            MemoryStore::open(dir, MEMORY_CAPACITY)
                .map_err(|e| format!("{}: {e}", dir.display()))?,
        ),
        None => None,
    };
    let (mut results, mut failed) = (Vec::new(), 0);
    for path in &o.files {
        let seen = read(path)
            .and_then(|b| observe(&file_name(path), &b).map_err(|e| format!("{path}: {e}")));
        match seen {
            Ok(obs) => {
                let text = memory_text(&obs, path);
                let mut filed = None;
                if let Some(st) = store.as_mut() {
                    // `None`: the same observation is already there
                    filed = Some(
                        st.record(Kind::Note, MEMORY_TOOL, Some(true), 0.6, &text)
                            .is_some(),
                    );
                }
                if let Some(url) = &o.rie {
                    post_memory(url, &text)?;
                    filed = Some(true);
                }
                if o.json {
                    let mut j = obs.to_json();
                    if let Some(f) = filed {
                        j["remembered"] = json!(if f { "yes" } else { "already known" });
                    }
                    results.push(j);
                } else {
                    println!("{}", obs.summary());
                    match filed {
                        Some(true) => println!("  remembered by the inference engine"),
                        Some(false) => println!("  already in the inference engine's memory"),
                        None => {}
                    }
                }
            }
            Err(e) => {
                failed += 1;
                if o.json {
                    results.push(json!({"file": file_name(path), "error": e}));
                } else {
                    eprintln!("ditto: {e}");
                }
            }
        }
    }
    if o.json {
        // one picture: one object (a card of fields); several: a table
        let out = if results.len() == 1 {
            results.pop().unwrap()
        } else {
            json!(results)
        };
        println!("{out}");
    }
    if failed == o.files.len() {
        return Err(if failed == 1 {
            "the picture could not be read".into()
        } else {
            "none of the pictures could be read".into()
        });
    }
    Ok(())
}

pub fn cmd_trace(argv: &[String]) -> Result<(), String> {
    let o = parse(argv)?;
    if o.rie.is_some() || o.memory.is_some() {
        return Err("--rie and --memory belong to `ditto look`".into());
    }
    if o.files.len() != 1 {
        return Err("`ditto trace` takes one picture".into());
    }
    let path = &o.files[0];
    let loaded = ditto_vision::load::load(&read(path)?).map_err(|e| format!("{path}: {e}"))?;
    // The outlines walk every pixel edge; this bounds time and memory for a huge photo.
    let (w, h) = loaded.image.dimensions();
    if w as u64 * h as u64 > 16_000_000 {
        return Err(format!("{path}: {w}x{h} is too large to trace (limit 16 megapixels); it is a photograph, not a screenshot"));
    }
    let t = trace::trace(&loaded.image, o.colours);
    let svg = t.to_svg();
    let out = o
        .out
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("{path}.svg")));
    if out.exists() && !o.force {
        return Err(format!(
            "{} exists; pass --force to replace it",
            out.display()
        ));
    }
    std::fs::write(&out, &svg).map_err(|e| format!("{}: {e}", out.display()))?;
    let exact = t.original_colours <= t.palette.len();
    let per_pixel = t.bytes_per_pixel(svg.len());
    let advice = if per_pixel > 0.5 {
        "this picture is not flat (a photo, a gradient or small text): the SVG is larger than a PNG would be"
    } else if exact {
        "exact: the SVG shows the same pixels as the picture"
    } else {
        "the SVG shows the picture reduced to the palette; raise --colours for a closer match"
    };
    if o.json {
        println!(
            "{}",
            json!({
                "svg": out.display().to_string(), "width": t.width, "height": t.height, "colours": t.layers.len(),
                "colours_in_picture": t.original_colours, "exact": if exact { "yes" } else { "no" },
                "svg_bytes": svg.len(), "picture_bytes": loaded.bytes, "note": advice
            })
        );
    } else {
        println!(
            "{}: {}x{}, {} of {} colours, {} corner points, {} bytes",
            out.display(),
            t.width,
            t.height,
            t.layers.len(),
            t.original_colours,
            t.points(),
            svg.len()
        );
        println!("  {advice}");
    }
    Ok(())
}

/// `ditto scripts TEXT...`: the scripts in a piece of text, as a sign or a receipt mixes them.
pub fn cmd_scripts(argv: &[String]) -> Result<(), String> {
    use ditto_vision::scripts::{ascii_digits, runs};
    let json_out = argv.first().is_some_and(|a| a == "--json");
    let text = argv[json_out as usize..].join(" ");
    if text.trim().is_empty() {
        return Err("no text given".into());
    }
    let found = runs(&text);
    if json_out {
        let rows: Vec<_> = found
            .iter()
            .map(|r| {
                json!({
                    "script": r.script.name(), "text": r.text, "digits_as_ascii": ascii_digits(&r.text),
                    "languages": r.languages().iter().map(|l| l.name).collect::<Vec<_>>().join(", "),
                })
            })
            .collect();
        println!("{}", json!(rows));
        return Ok(());
    }
    if found.is_empty() {
        println!(
            "no letters of a covered script (numbers only: {})",
            ascii_digits(text.trim())
        );
    }
    for r in &found {
        let langs = r
            .languages()
            .iter()
            .map(|l| l.name)
            .collect::<Vec<_>>()
            .join(", ");
        println!("{:<17} {}", r.script.name(), ascii_digits(&r.text));
        println!(
            "{:<17} could be: {}",
            "",
            if langs.is_empty() {
                "a language outside India's scheduled ones"
            } else {
                &langs
            }
        );
    }
    Ok(())
}

fn key_path(given: Option<PathBuf>) -> Result<PathBuf, String> {
    if let Some(p) = given {
        return Ok(p);
    }
    if let Some(d) = std::env::var_os("DITTO_HOME") {
        return Ok(PathBuf::from(d).join("device.key"));
    }
    std::env::var_os("HOME")
        .map(|h| PathBuf::from(h).join(".ditto/device.key"))
        .ok_or_else(|| "no key: pass --key FILE or set DITTO_HOME".into())
}

/// `ditto label`: mark a generated PNG as generated, signed by this device.
pub fn cmd_label(argv: &[String]) -> Result<(), String> {
    use ditto_vision::provenance::{badge, device_key, label_png, now_utc, Record};
    let (mut input, mut out, mut key, mut force, mut with_badge) = (None, None, None, false, false);
    let mut rec = Record {
        prompt: String::new(),
        seed: 0,
        model: String::new(),
        source: "text".into(),
        created: now_utc(),
    };
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "-o" | "--out" => out = Some(PathBuf::from(val()?)),
            "--prompt" => rec.prompt = val()?,
            "--seed" => rec.seed = val()?.parse().map_err(|_| "--seed: not a whole number")?,
            "--model" => rec.model = val()?,
            "--source" => rec.source = val()?,
            "--key" => key = Some(PathBuf::from(val()?)),
            "--badge" => with_badge = true,
            "--force" => force = true,
            s if s.starts_with('-') => return Err(format!("unknown option '{s}'")),
            _ if input.is_none() => input = Some(a.clone()),
            _ => return Err("`ditto label` takes one picture".into()),
        }
    }
    let input = input.ok_or("no picture given")?;
    if rec.prompt.trim().is_empty() || rec.model.trim().is_empty() {
        return Err(
            "--prompt and --model are required: a label must say what made the picture".into(),
        );
    }
    let out = out.ok_or("-o OUT.png is required; the original is never changed in place")?;
    if out.exists() && !force {
        return Err(format!(
            "{} exists; pass --force to replace it",
            out.display()
        ));
    }
    let bytes = read(&input)?;
    let loaded = ditto_vision::load::load(&bytes).map_err(|e| format!("{input}: {e}"))?;
    // labels live in PNG text chunks: anything else is re-encoded as PNG, losslessly
    let mut img = loaded.image;
    if with_badge {
        badge(&mut img);
    }
    let mut png = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    let key = device_key(&key_path(key)?)?;
    let labelled = label_png(&png, &rec, &key)?;
    std::fs::write(&out, &labelled).map_err(|e| format!("{}: {e}", out.display()))?;
    println!(
        "{}: labelled as AI-generated{}, signed by this device",
        out.display(),
        if with_badge {
            ", with a visible AI badge"
        } else {
            ""
        }
    );
    Ok(())
}

/// `ditto verify`: what a file says about how it was made.
pub fn cmd_verify(argv: &[String]) -> Result<(), String> {
    let o = parse(argv)?;
    let mut any_broken = false;
    for path in &o.files {
        let bytes = read(path)?;
        let loaded = ditto_vision::load::load(&bytes).map_err(|e| format!("{path}: {e}"))?;
        let p = ditto_vision::provenance::inspect(&bytes, &loaded.image);
        let status = p
            .ditto
            .as_ref()
            .map(|d| format!("{:?}", d.status).to_lowercase());
        any_broken |= matches!(status.as_deref(), Some("pixelschanged" | "forged"));
        if o.json {
            let mut j = json!({"file": file_name(path), "generated": if p.says_generated() { "yes" } else { "no marker found" }, "provenance": p.describe()});
            if let Some(s) = status {
                j["ditto_label"] = json!(s);
            }
            if let Some(d) = &p.ditto {
                for k in ["prompt", "seed", "model", "source", "created"] {
                    if let Some(v) = d.record.get(k) {
                        j[k] = v.clone();
                    }
                }
            }
            println!("{j}");
        } else {
            println!("{}: {}", file_name(path), p.describe());
        }
    }
    if any_broken {
        return Err("a label shows the picture was changed after it was made, or the label was tampered with".into());
    }
    Ok(())
}

/// `ditto compare ORIGINAL RESULT [--mask MASK]`: did an edit keep the real picture?
pub fn cmd_compare(argv: &[String]) -> Result<(), String> {
    use ditto_vision::fidelity::{compare, Verdict};
    let (mut files, mut mask, mut json_out) = (Vec::new(), None, false);
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--mask" => mask = Some(it.next().cloned().ok_or("--mask needs a value")?),
            "--json" => json_out = true,
            "--" => files.extend(it.by_ref().cloned()),
            s if s.starts_with('-') => return Err(format!("unknown option '{s}'")),
            _ => files.push(a.clone()),
        }
    }
    let [original, result] = <[String; 2]>::try_from(files)
        .map_err(|_| "give the original picture and the edited one")?;
    let pic = |p: &str| -> Result<image::RgbaImage, String> {
        ditto_vision::load::load(&read(p)?)
            .map(|l| l.image)
            .map_err(|e| format!("{p}: {e}"))
    };
    let m = mask.as_deref().map(pic).transpose()?;
    let f = compare(&pic(&original)?, &pic(&result)?, m.as_ref())?;
    let what = if m.is_some() {
        "outside the area the edit was allowed to change"
    } else {
        "anywhere (no mask given, so every change counts)"
    };
    let meaning = match f.verdict {
        Verdict::Faithful => format!("the real picture is kept {what}"),
        Verdict::Drifted => {
            format!("the edit changed things {what}: check them before using or sharing it")
        }
        Verdict::Different => format!(
            "this is no longer the same picture {what}: do not present it as the original scene"
        ),
    };
    let psnr = if f.psnr.is_finite() {
        format!("{:.1} dB", f.psnr)
    } else {
        "identical".into()
    };
    if json_out {
        let mut j = json!({"verdict": f.verdict.name(), "meaning": meaning, "psnr": psnr, "changed_outside_percent": (f.changed_outside as f64 * 1000.0).round() / 10.0, "resized": if f.resized { "yes" } else { "no" }});
        if let Some(i) = f.changed_inside {
            j["changed_inside_percent"] = json!((i as f64 * 1000.0).round() / 10.0);
        }
        if let Some((x, y, w, h)) = f.drift_box {
            j["changes_outside_at"] = json!(format!("x {x}, y {y}, {w}x{h}"));
        }
        println!("{j}");
    } else {
        println!("{}: {meaning}", f.verdict.name());
        println!(
            "  {:.1}% of those pixels changed, PSNR {psnr}{}",
            f.changed_outside * 100.0,
            if f.resized {
                ", result resized to the original's size"
            } else {
                ""
            }
        );
        if let Some((x, y, w, h)) = f.drift_box {
            println!("  changes found within x {x}, y {y}, {w}x{h}");
        }
    }
    Ok(())
}
