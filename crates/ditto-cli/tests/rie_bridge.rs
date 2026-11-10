//! The hand-over to the inference engine, tested with the inference engine's own code:
//! its manifest loader and tool runner (`rie-server`) and its memory store (`rie-core`).

use image::{Rgba, RgbaImage};
use rie_core::memory::{Kind, MemoryStore};
use rie_server::headless::{load_manifest, Status};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const DITTO: &str = env!("CARGO_BIN_EXE_ditto");

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ditto-bridge-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A small flat picture, like a screenshot: white, a dark bar, a blue button.
fn screenshot(path: &Path) {
    let mut img = RgbaImage::from_pixel(160, 100, Rgba([255, 255, 255, 255]));
    for x in 0..160 {
        for y in 0..12 {
            img.put_pixel(x, y, Rgba([32, 32, 40, 255]));
        }
    }
    for x in 20..90 {
        for y in 40..64 {
            img.put_pixel(x, y, Rgba([20, 90, 220, 255]));
        }
    }
    img.save(path).unwrap();
}

/// The shipped manifest, with `ditto` pointing at the binary under test.
fn manifest(d: &Path) -> PathBuf {
    let shipped =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../integrations/rie/ditto-tools.json");
    let mut m: Value = serde_json::from_str(&std::fs::read_to_string(shipped).unwrap()).unwrap();
    for t in m["tools"].as_array_mut().unwrap() {
        assert_eq!(t["command"][0], "ditto");
        t["command"][0] = json!(DITTO);
    }
    let p = d.join("tools.json");
    std::fs::write(&p, m.to_string()).unwrap();
    p
}

#[test]
fn the_inference_engine_runs_the_tools_from_the_shipped_manifest() {
    let d = dir("tools");
    let pic = d.join("shot.png");
    screenshot(&pic);
    let tools = load_manifest(&manifest(&d)).expect("the manifest loads in rie-server");
    assert_eq!(
        tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        ["photo.inspect", "image.to_svg"]
    );

    let look = &tools[0];
    let out = look.run(&json!({"path": pic.to_str().unwrap()}));
    assert_eq!(out.status, Status::Ok, "{:?}", out.error);
    let v = out
        .value
        .as_object()
        .expect("a JSON object, which the engine shows as fields");
    assert_eq!(v["file"], "shot.png");
    assert_eq!(v["kind"], "screenshot or graphic");
    assert_eq!(
        (v["width"].as_u64(), v["height"].as_u64()),
        (Some(160), Some(100))
    );
    assert!(v["colours"].as_str().unwrap().starts_with("white"));
    assert!(v.values().all(|x| x.is_string() || x.is_number()));

    // a file that is not a picture: an error card, not a crash
    let txt = d.join("notes.txt");
    std::fs::write(&txt, "not a picture").unwrap();
    let out = look.run(&json!({"path": txt.to_str().unwrap()}));
    assert_eq!(out.status, Status::Error);
    // a path that looks like an option is still a path
    let out = look.run(&json!({"path": "--memory"}));
    assert_eq!(out.status, Status::Error);
    assert!(!Path::new("--memory").exists());

    let svg_tool = &tools[1];
    let out = svg_tool.run(&json!({"path": pic.to_str().unwrap()}));
    assert_eq!(out.status, Status::Ok, "{:?}", out.error);
    assert_eq!(out.value["exact"], "yes");
    assert_eq!(out.value["colours"], 3);
    let svg = std::fs::read_to_string(d.join("shot.png.svg")).unwrap();
    assert!(
        svg.starts_with("<svg ")
            && svg.contains("fill=\"#145adc\"")
            && svg.contains("M20 40H90V64H20Z"),
        "{svg}"
    );
    // it never replaces a file on its own
    let out = svg_tool.run(&json!({"path": pic.to_str().unwrap()}));
    assert_eq!(out.status, Status::Error);
    assert!(out.error.unwrap_or_default().contains("exists"));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn observations_land_in_the_inference_engines_memory_and_are_recalled() {
    let d = dir("memory");
    let pic = d.join("receipt.png");
    screenshot(&pic);
    let mem = d.join("rie-memory");
    let run = || {
        Command::new(DITTO)
            .args([
                "look",
                "--memory",
                mem.to_str().unwrap(),
                "--json",
                "--",
                pic.to_str().unwrap(),
            ])
            .output()
            .unwrap()
    };
    let out = run();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let j: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(j["remembered"], "yes");

    // read back with the inference engine's own store
    {
        let mut store = MemoryStore::open(&mem, 4096).unwrap();
        assert_eq!(store.len(), 1);
        let hits = store.recall("receipt.png screenshot picture", 3, Some(Kind::Note), 0.0);
        assert_eq!(hits.len(), 1);
        let text = &hits[0].obs.text;
        assert!(
            text.starts_with("Picture receipt.png: screenshot or graphic, 160x100"),
            "{text}"
        );
        assert!(text.contains(&format!(
            "File: {}",
            std::fs::canonicalize(&pic).unwrap().display()
        )));
        assert_eq!(hits[0].obs.tool, "ditto.look");
    }
    // the same picture again is not stored twice
    let j: Value = serde_json::from_slice(&run().stdout).unwrap();
    assert_eq!(j["remembered"], "already known");
    assert_eq!(MemoryStore::open(&mem, 4096).unwrap().len(), 1);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn observations_are_posted_to_a_running_server_and_only_a_local_one() {
    let d = dir("post");
    let pic = d.join("a.png");
    screenshot(&pic);
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
        // read the headers, then exactly the announced body
        let (head, mut body) = loop {
            let n = s.read(&mut chunk).unwrap();
            buf.extend_from_slice(&chunk[..n]);
            if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break (
                    String::from_utf8(buf[..i].to_vec()).unwrap(),
                    buf[i + 4..].to_vec(),
                );
            }
        };
        let len: usize = head
            .lines()
            .find_map(|l| l.strip_prefix("Content-Length: "))
            .unwrap()
            .parse()
            .unwrap();
        while body.len() < len {
            let n = s.read(&mut chunk).unwrap();
            body.extend_from_slice(&chunk[..n]);
        }
        s.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{\"ok\":true}",
        )
        .unwrap();
        (head, String::from_utf8(body).unwrap())
    });
    let out = Command::new(DITTO)
        .args([
            "look",
            "--rie",
            &format!("http://127.0.0.1:{port}"),
            pic.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("remembered by the inference engine"));
    let (head, body) = server.join().unwrap();
    assert!(head.starts_with("POST /v1/memory HTTP/1.1"), "{head}");
    // the fields rie-server's /v1/memory reads
    let b: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        (b["kind"].as_str(), b["tool"].as_str(), b["ok"].as_bool()),
        (Some("note"), Some("ditto.look"), Some(true))
    );
    assert!(b["text"]
        .as_str()
        .unwrap()
        .starts_with("Picture a.png: screenshot or graphic"));

    // where the user was and what they photographed does not leave the phone
    for url in [
        "http://example.com:8080",
        "https://127.0.0.1:8080",
        "http://10.0.0.5",
    ] {
        let out = Command::new(DITTO)
            .args(["look", "--rie", url, pic.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(!out.status.success(), "{url} should be refused");
    }
    // nothing listening: an error, not a hang
    let out = Command::new(DITTO)
        .args(["look", "--rie", "http://127.0.0.1:9", pic.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn several_pictures_and_bad_ones() {
    let d = dir("many");
    let (a, b, bad) = (d.join("a.png"), d.join("b.png"), d.join("c.png"));
    screenshot(&a);
    screenshot(&b);
    std::fs::write(&bad, b"\x89PNG\r\n\x1a\nbroken").unwrap();
    let out = Command::new(DITTO)
        .args(["look", "--json"])
        .args([&a, &b, &bad])
        .output()
        .unwrap();
    assert!(out.status.success(), "some pictures were read");
    let j: Value = serde_json::from_slice(&out.stdout).unwrap();
    let rows = j.as_array().unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["fingerprint"], rows[1]["fingerprint"]); // identical pictures
    assert!(rows[2]["error"].as_str().unwrap().contains("damaged"));
    let out = Command::new(DITTO)
        .args(["look"])
        .arg(&bad)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    let out = Command::new(DITTO)
        .args(["trace", "--colours", "1"])
        .arg(&a)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    let _ = std::fs::remove_dir_all(&d);
}
