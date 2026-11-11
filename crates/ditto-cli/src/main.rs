//! `ditto`: for now, the Phase 0 measurements of `docs/PLAN.md`.

mod pictures;

use ditto_core::bench::{self, BenchConfig, BenchResult};
use ditto_core::shapes::{self, DitShape};
use rie_core::quant::GgmlType;
use std::process::ExitCode;

const USAGE: &str = "\
ditto look PICTURE... [--json] [--rie http://127.0.0.1:8080] [--memory DIR]
                           what a picture contains, read and measured without a model; optionally
                           filed in the inference engine's memory (running server, or its directory)
ditto trace PICTURE [-o OUT.svg] [--colours 16] [--json] [--force]
                           a screenshot, diagram or logo as SVG, exact to the pixel
ditto verify PICTURE... [--json]
                           was it made by AI? reads Ditto's signed label, the IPTC marker other
                           generators write, and C2PA presence; says what it cannot know
ditto label PICTURE -o OUT.png --prompt TEXT --model NAME [--seed N] [--badge] [--key FILE]
                           mark a generated picture as AI-made, signed by this device
ditto compare ORIGINAL EDITED [--mask MASK] [--json]
                           did an AI edit keep the real picture outside the area it was asked to change?
ditto scripts [--json] TEXT  the scripts in mixed text (a sign in Telugu, Hindi and English), the
                           languages each could be, and every script's digits as 0-9
ditto bench    [options]   time one denoiser block with random weights, scale to a clip
ditto estimate [options]   the same table from a throughput figure, without running anything
ditto bench-conv [--size 416] [--mult 1.0] [--wtype T] [--threads N] [--reps N]
                           time a convolution stack the size of a small detector's backbone

  --model NAME     wan-1.3b (default) or pixart-0.6b
  --width W --height H --frames F    output size (default 480x272, 33 frames; images: 1 frame)
  --tokens N       set the token count directly instead
  --text N         prompt length in text tokens (default 64)
  --steps N        denoising steps (default 6)
  --passes N       passes through the model per step: 1 distilled, 2 with guidance (default 1)
  --experts E --used K    also time the feed-forward block split in E experts, K running
bench only:
  --wtype T        q4_0 (default), q8_0, f16, f32
  --threads N      default: the performance cores this process may use
  --reps N         runs of the block, fastest counts (default 2)
estimate only:
  --gmacs X        multiply-adds per second, in billions (default 15, the plan's assumption)
";

struct Args {
    shape: DitShape,
    width: usize,
    height: usize,
    frames: usize,
    tokens: Option<usize>,
    text: usize,
    steps: usize,
    passes: usize,
    experts: Option<usize>,
    used: Option<usize>,
    wtype: GgmlType,
    threads: Option<usize>,
    reps: usize,
    gmacs: f64,
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let mut a = Args {
        shape: shapes::WAN_1_3B,
        width: 480,
        height: 272,
        frames: 33,
        tokens: None,
        text: 64,
        steps: 6,
        passes: 1,
        experts: None,
        used: None,
        wtype: GgmlType::Q4_0,
        threads: None,
        reps: 2,
        gmacs: 15.0,
    };
    let mut frames_given = false;
    let mut it = argv.iter();
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or(format!("{flag} needs a value"));
        let num = |s: &String| {
            s.parse::<usize>()
                .map_err(|_| format!("{flag}: '{s}' is not a whole number"))
        };
        match flag.as_str() {
            "--model" => {
                let v = val()?;
                a.shape = shapes::by_name(v).ok_or(format!(
                    "unknown model '{v}' (known: wan-1.3b, pixart-0.6b)"
                ))?;
            }
            "--width" => a.width = num(val()?)?,
            "--height" => a.height = num(val()?)?,
            "--frames" => {
                a.frames = num(val()?)?;
                frames_given = true;
            }
            "--tokens" => a.tokens = Some(num(val()?)?),
            "--text" => a.text = num(val()?)?,
            "--steps" => a.steps = num(val()?)?,
            "--passes" => a.passes = num(val()?)?,
            "--experts" => a.experts = Some(num(val()?)?),
            "--used" => a.used = Some(num(val()?)?),
            "--threads" => a.threads = Some(num(val()?)?),
            "--reps" => a.reps = num(val()?)?,
            "--wtype" => {
                let v = val()?;
                a.wtype = match v.to_ascii_lowercase().as_str() {
                    "q4_0" => GgmlType::Q4_0,
                    "q8_0" => GgmlType::Q8_0,
                    "f16" => GgmlType::F16,
                    "f32" => GgmlType::F32,
                    _ => return Err(format!("unknown weight type '{v}' (q4_0, q8_0, f16, f32)")),
                };
            }
            "--gmacs" => {
                let v = val()?;
                a.gmacs = v
                    .parse::<f64>()
                    .ok()
                    .filter(|g| *g > 0.0)
                    .ok_or(format!("--gmacs: '{v}' is not a positive number"))?;
            }
            other => return Err(format!("unknown option '{other}'")),
        }
    }
    if a.shape.vae_time <= 1 && !frames_given {
        a.frames = 1;
    }
    match (a.experts, a.used) {
        (None, None) => {}
        (Some(e), Some(k)) if e >= 2 && k >= 1 && k <= e && a.shape.ffn % e == 0 => {}
        (Some(e), Some(_)) if e >= 2 && a.shape.ffn % e != 0 => {
            return Err(format!(
                "{}'s feed-forward width {} is not divisible by {e} experts",
                a.shape.name, a.shape.ffn
            ))
        }
        (Some(_), Some(_)) => {
            return Err("need --experts of at least 2 and 1 <= --used <= --experts".into())
        }
        _ => return Err("--experts and --used go together".into()),
    }
    if a.steps == 0 || a.passes == 0 || a.reps == 0 || a.threads == Some(0) {
        return Err("--steps, --passes, --reps and --threads must be at least 1".into());
    }
    Ok(a)
}

fn tokens(a: &Args) -> Result<usize, String> {
    let n = a
        .tokens
        .unwrap_or_else(|| a.shape.tokens(a.width, a.height, a.frames));
    if n == 0 {
        return Err("that size gives no tokens".into());
    }
    Ok(n)
}

fn duration(secs: f64) -> String {
    if secs < 90.0 {
        format!("{secs:.1} s")
    } else if secs < 5400.0 {
        format!("{:.1} min", secs / 60.0)
    } else {
        format!("{:.1} h", secs / 3600.0)
    }
}

fn describe(a: &Args, n: usize) -> String {
    let what = match a.tokens {
        Some(_) => format!("{n} tokens"),
        None if a.shape.vae_time <= 1 => format!("{}x{} image, {n} tokens", a.width, a.height),
        None => format!("{}x{}, {} frames, {n} tokens", a.width, a.height, a.frames),
    };
    format!(
        "{}: {what}, {} text tokens, {} steps x {} pass(es)",
        a.shape.name, a.text, a.steps, a.passes
    )
}

fn print_run(label: &str, r: &BenchResult, a: &Args) {
    let b = &r.block;
    let pct = |d: std::time::Duration| 100.0 * d.as_secs_f64() / b.total().as_secs_f64();
    println!("{label}");
    println!(
        "  one block      {:>9}   weights {:.0}%, self-attention {:.0}%, cross-attention {:.0}%, other {:.0}%",
        duration(b.total().as_secs_f64()),
        pct(b.linear),
        pct(b.self_attention),
        pct(b.cross_attention),
        pct(b.other)
    );
    println!(
        "  throughput     weights {:.1} G multiply-adds/s, attention {:.1} G/s, whole block {:.1} G/s",
        r.linear_rate() / 1e9,
        r.attention_rate() / 1e9,
        r.overall_rate() / 1e9
    );
    println!(
        "  one step       {:>9}   ({} blocks)",
        duration(r.step_secs),
        a.shape.layers
    );
    println!(
        "  whole job      {:>9}   denoising only; text encoding and decoding come on top",
        duration(r.step_secs * (a.steps * a.passes) as f64)
    );
}

fn cmd_bench(a: &Args) -> Result<(), String> {
    let n = tokens(a)?;
    println!("{}", describe(a, n));
    let cfg = BenchConfig {
        shape: a.shape,
        tokens: n,
        text_tokens: a.text,
        wtype: a.wtype,
        threads: a.threads,
        experts: None,
        reps: a.reps,
    };
    let dense = bench::run(&cfg);
    println!(
        "{} weights, {} thread(s), kernel {}, {:.0} MB per block",
        a.wtype.name(),
        dense.threads,
        rie_core::CpuFeatures::get().kernel_name(),
        dense.block_bytes as f64 / 1e6
    );
    print_run("dense", &dense, a);
    if let (Some(e), Some(k)) = (a.experts, a.used) {
        let moe = bench::run(&BenchConfig {
            experts: Some((e, k)),
            ..cfg
        });
        print_run(&format!("{k} of {e} experts"), &moe, a);
        println!(
            "  against dense  {:.0}% of the time",
            100.0 * moe.step_secs / dense.step_secs
        );
    }
    if let Some(rss) = bench::peak_rss() {
        println!(
            "peak memory of this benchmark: {:.0} MB (one block's weights and buffers)",
            rss as f64 / 1e6
        );
    }
    Ok(())
}

fn cmd_estimate(a: &Args) -> Result<(), String> {
    let n = tokens(a)?;
    println!("{}", describe(a, n));
    println!(
        "assuming {} G multiply-adds per second (not measured: run `ditto bench` on the phone)",
        a.gmacs
    );
    let row = |label: &str, share: f64| {
        let m = a.shape.macs(n, a.text, share);
        let step = m.total() / (a.gmacs * 1e9);
        println!(
            "{label:<18} step {:>9}   job {:>9}   weights {:.0}%, attention {:.0}%",
            duration(step),
            duration(step * (a.steps * a.passes) as f64),
            100.0 * m.linear / m.total(),
            100.0 * m.attention / m.total()
        );
    };
    row("dense", 1.0);
    if let (Some(e), Some(k)) = (a.experts, a.used) {
        row(&format!("{k} of {e} experts"), k as f64 / e as f64);
    }
    Ok(())
}

/// Frames per second of a perception-sized convolution stack (the driver warning proposal).
fn cmd_bench_conv(argv: &[String]) -> Result<(), String> {
    use ditto_core::conv::Backbone;
    use rie_core::threadpool::{PoolConfig, ThreadPool};
    let (mut size, mut mult, mut wtype, mut threads, mut reps) =
        (416usize, 1.0f32, GgmlType::Q4_0, None, 3usize);
    let mut it = argv.iter();
    while let Some(flag) = it.next() {
        let v = it.next().ok_or(format!("{flag} needs a value"))?;
        let num = || {
            v.parse::<usize>()
                .ok()
                .filter(|n| *n >= 1)
                .ok_or(format!("{flag}: '{v}' is not a whole number of at least 1"))
        };
        match flag.as_str() {
            "--size" => size = num()?,
            "--threads" => threads = Some(num()?),
            "--reps" => reps = num()?,
            "--mult" => {
                mult = v
                    .parse::<f32>()
                    .ok()
                    .filter(|m| *m > 0.0 && *m <= 4.0)
                    .ok_or(format!("--mult: '{v}' is not between 0 and 4"))?
            }
            "--wtype" => {
                wtype = match v.to_ascii_lowercase().as_str() {
                    "q4_0" => GgmlType::Q4_0,
                    "q8_0" => GgmlType::Q8_0,
                    "f16" => GgmlType::F16,
                    "f32" => GgmlType::F32,
                    _ => return Err(format!("unknown weight type '{v}' (q4_0, q8_0, f16, f32)")),
                }
            }
            other => return Err(format!("unknown option '{other}'")),
        }
    }
    if size < 32 {
        return Err("--size must be at least 32".into());
    }
    let threads = threads.unwrap_or_else(bench::default_threads);
    let pool = ThreadPool::new(PoolConfig {
        n_threads: threads,
        cores: vec![],
        spin: std::time::Duration::from_micros(50),
    });
    let net = Backbone::synthetic(mult, wtype, 1);
    let x: Vec<f32> = (0..size * size * 3)
        .map(|i| ((i * 7919) % 255) as f32 / 127.5 - 1.0)
        .collect();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let c = std::time::Instant::now();
        let (y, ..) = net.forward(&pool, &x, size, size);
        std::hint::black_box(y);
        best = best.min(c.elapsed().as_secs_f64());
    }
    let macs = net.macs(size, size);
    println!(
        "convolution stack, {} layers, channels x{mult}, {size}x{size} input: {:.2} G multiply-adds per frame, {:.1} MB of weights",
        net.layers.len(),
        macs / 1e9,
        net.size_bytes() as f64 / 1e6
    );
    println!(
        "{} weights, {threads} thread(s), kernel {}",
        wtype.name(),
        rie_core::CpuFeatures::get().kernel_name()
    );
    println!(
        "  one frame      {:.0} ms   {:.1} frames per second",
        best * 1e3,
        1.0 / best
    );
    println!(
        "  throughput     {:.1} G multiply-adds/s",
        macs / best / 1e9
    );
    println!("random weights: this measures cost only, it detects nothing");
    Ok(())
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let run = |f: fn(&Args) -> Result<(), String>| parse(&argv[1..]).and_then(|a| f(&a));
    let res = match argv.first().map(String::as_str) {
        Some("bench") => run(cmd_bench),
        Some("estimate") => run(cmd_estimate),
        Some("look") => pictures::cmd_look(&argv[1..]),
        Some("verify") => pictures::cmd_verify(&argv[1..]),
        Some("label") => pictures::cmd_label(&argv[1..]),
        Some("compare") => pictures::cmd_compare(&argv[1..]),
        Some("scripts") => pictures::cmd_scripts(&argv[1..]),
        Some("trace") => pictures::cmd_trace(&argv[1..]),
        Some("bench-conv") => cmd_bench_conv(&argv[1..]),
        Some("help" | "--help" | "-h") | None => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Some(other) => Err(format!("unknown command '{other}'")),
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ditto: {e}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}
