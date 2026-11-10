//! Phase 0: time one synthetic block and scale it to a step and a clip.

use crate::attention::Memory;
use crate::block::{Block, Timing};
use crate::shapes::DitShape;
use rie_core::cpu::CpuTopology;
use rie_core::quant::GgmlType;
use rie_core::threadpool::{PoolConfig, ThreadPool};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub shape: DitShape,
    pub tokens: usize,
    /// Prompt length in text tokens.
    pub text_tokens: usize,
    pub wtype: GgmlType,
    /// `None`: the performance cores this process may use.
    pub threads: Option<usize>,
    /// `Some((experts, used))`: split the feed-forward block and run `used` experts.
    pub experts: Option<(usize, usize)>,
    /// The block is run this many times; the fastest run counts.
    pub reps: usize,
}

#[derive(Debug, Clone)]
pub struct BenchResult {
    pub threads: usize,
    /// Fastest run of one block.
    pub block: Timing,
    /// Multiply-adds of that block.
    pub linear_macs: f64,
    pub attention_macs: f64,
    /// One pass through the whole model: `block × layers`.
    pub step_secs: f64,
    /// Bytes of one block's weights.
    pub block_bytes: usize,
}

impl BenchResult {
    /// Multiply-adds per second in the weight matrices.
    pub fn linear_rate(&self) -> f64 {
        self.linear_macs / self.block.linear.as_secs_f64()
    }
    /// Multiply-adds per second in attention.
    pub fn attention_rate(&self) -> f64 {
        self.attention_macs / (self.block.self_attention + self.block.cross_attention).as_secs_f64()
    }
    /// Multiply-adds per second over the whole block, everything included.
    pub fn overall_rate(&self) -> f64 {
        (self.linear_macs + self.attention_macs) / self.block.total().as_secs_f64()
    }
}

/// Threads to use by default: the performance cores, but never more than the cores this
/// process is allowed to run on. On agenticOS the model's cgroup decides that.
pub fn default_threads() -> usize {
    let allowed = std::thread::available_parallelism().map_or(1, |n| n.get());
    CpuTopology::detect()
        .recommended_threads()
        .clamp(1, allowed)
}

pub fn run(cfg: &BenchConfig) -> BenchResult {
    let s = cfg.shape;
    let threads = cfg.threads.unwrap_or_else(default_threads).max(1);
    let pool = ThreadPool::new(PoolConfig {
        n_threads: threads,
        cores: vec![],
        spin: Duration::from_micros(50),
    });
    let block = Block::synthetic(
        s.dim,
        s.ffn,
        s.heads,
        cfg.wtype,
        cfg.experts.map(|e| e.0),
        1,
    );
    let used: Vec<usize> = (0..cfg.experts.map_or(0, |e| e.1)).collect();
    let n = cfg.tokens;

    let mut seed = 0x0123_4567_89AB_CDEFu64;
    let mut noise = |len: usize| -> Vec<f32> {
        (0..len)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    };
    let text = Memory::new(
        &pool,
        &noise(cfg.text_tokens * s.dim),
        &noise(cfg.text_tokens * s.dim),
        cfg.text_tokens,
        s.dim,
        s.heads,
    );
    let x0 = noise(n * s.dim);

    let mut best: Option<Timing> = None;
    for _ in 0..cfg.reps.max(1) {
        let mut x = x0.clone();
        let t = block.forward(&pool, &mut x, n, &text, &used);
        if best.is_none_or(|b| t.total() < b.total()) {
            best = Some(t);
        }
    }
    let best = best.unwrap();
    let share = cfg.experts.map_or(1.0, |(e, k)| k as f64 / e as f64);
    BenchResult {
        threads,
        block: best,
        linear_macs: n as f64 * s.linear_macs_per_token(share),
        attention_macs: n as f64 * s.attention_macs_per_token(n, cfg.text_tokens),
        step_secs: best.total().as_secs_f64() * s.layers as f64,
        block_bytes: block.size_bytes(),
    }
}

/// Peak resident memory of this process in bytes (Linux), if it can be read.
pub fn peak_rss() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with("VmHWM:"))?;
    line.split_whitespace()
        .nth(1)?
        .parse::<u64>()
        .ok()
        .map(|kb| kb * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes::DitShape;

    const TINY: DitShape = DitShape {
        name: "tiny",
        dim: 64,
        ffn: 256,
        layers: 3,
        heads: 4,
        vae_space: 8,
        vae_time: 4,
        patch: 2,
    };

    #[test]
    fn runs_and_accounts_for_the_work() {
        let cfg = BenchConfig {
            shape: TINY,
            tokens: 40,
            text_tokens: 6,
            wtype: GgmlType::Q4_0,
            threads: Some(2),
            experts: None,
            reps: 2,
        };
        let dense = run(&cfg);
        assert_eq!(dense.threads, 2);
        assert_eq!(
            dense.linear_macs,
            40.0 * (6.0 * 64.0 * 64.0 + 2.0 * 64.0 * 256.0)
        );
        assert_eq!(dense.attention_macs, 40.0 * 2.0 * 46.0 * 64.0);
        assert!((dense.step_secs - dense.block.total().as_secs_f64() * 3.0).abs() < 1e-9);
        assert!(
            dense.linear_rate() > 0.0 && dense.attention_rate() > 0.0 && dense.overall_rate() > 0.0
        );

        let moe = run(&BenchConfig {
            experts: Some((4, 1)),
            ..cfg
        });
        assert_eq!(moe.attention_macs, dense.attention_macs);
        assert_eq!(
            dense.linear_macs - moe.linear_macs,
            40.0 * 2.0 * 64.0 * 256.0 * 0.75
        );
    }

    #[test]
    fn default_threads_fit_the_machine() {
        let n = default_threads();
        assert!(n >= 1 && n <= std::thread::available_parallelism().unwrap().get());
    }
}
