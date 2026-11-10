//! Shapes of the candidate denoisers and the cost model used in `docs/PLAN.md`.
//!
//! The shapes are the published hyper-parameters as recalled when the plan was written.
//! They are checked against the real model files when the converter exists (Phase 1).

/// A diffusion transformer, as far as its cost is concerned.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DitShape {
    pub name: &'static str,
    /// Width of a token.
    pub dim: usize,
    /// Hidden width of the feed-forward block.
    pub ffn: usize,
    pub layers: usize,
    pub heads: usize,
    /// Pixels per latent cell, per side (VAE).
    pub vae_space: usize,
    /// Frames per latent frame after the first (VAE). 1 for image models.
    pub vae_time: usize,
    /// Latent cells per token, per side.
    pub patch: usize,
}

pub const WAN_1_3B: DitShape = DitShape {
    name: "wan-1.3b",
    dim: 1536,
    ffn: 8960,
    layers: 30,
    heads: 12,
    vae_space: 8,
    vae_time: 4,
    patch: 2,
};
pub const PIXART_0_6B: DitShape = DitShape {
    name: "pixart-0.6b",
    dim: 1152,
    ffn: 4608,
    layers: 28,
    heads: 16,
    vae_space: 8,
    vae_time: 1,
    patch: 2,
};

pub const ALL: &[DitShape] = &[WAN_1_3B, PIXART_0_6B];

pub fn by_name(name: &str) -> Option<DitShape> {
    ALL.iter().copied().find(|s| s.name == name)
}

/// Multiply-adds of one pass through the whole denoiser.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Macs {
    /// Weight matrices: attention projections and the feed-forward block.
    pub linear: f64,
    /// Scores and weighted sums of self- and cross-attention.
    pub attention: f64,
}

impl Macs {
    pub fn total(&self) -> f64 {
        self.linear + self.attention
    }
}

impl DitShape {
    /// Tokens for an image (`frames == 1`) or a clip. Sizes are rounded down to whole patches.
    pub fn tokens(&self, width: usize, height: usize, frames: usize) -> usize {
        let cell = self.vae_space * self.patch;
        let t = if self.vae_time <= 1 {
            1
        } else {
            1 + frames.saturating_sub(1) / self.vae_time
        };
        t * (width / cell) * (height / cell)
    }

    /// Multiply-adds per token in one block's weight matrices. `ffn_share` is the part of
    /// the feed-forward block that runs (`used / experts`, 1.0 for the dense model).
    ///
    /// Self-attention has four `dim × dim` matrices. Cross-attention has two per image
    /// token; its key and value projections act on the text and are computed once per job.
    pub fn linear_macs_per_token(&self, ffn_share: f64) -> f64 {
        let d = self.dim as f64;
        6.0 * d * d + 2.0 * d * self.ffn as f64 * ffn_share
    }

    /// Multiply-adds per token in one block's attention: scores and the weighted sum, over
    /// `n` image tokens and `text` text tokens.
    pub fn attention_macs_per_token(&self, n: usize, text: usize) -> f64 {
        2.0 * (n + text) as f64 * self.dim as f64
    }

    /// One pass through all blocks for `n` tokens.
    pub fn macs(&self, n: usize, text: usize, ffn_share: f64) -> Macs {
        let per = (n * self.layers) as f64;
        Macs {
            linear: per * self.linear_macs_per_token(ffn_share),
            attention: per * self.attention_macs_per_token(n, text),
        }
    }

    /// Parameters in the blocks' weight matrices (biases, norms and embeddings left out).
    pub fn block_params(&self) -> f64 {
        let d = self.dim as f64;
        self.layers as f64 * (8.0 * d * d + 2.0 * d * self.ffn as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The token counts in the plan's table.
    #[test]
    fn plan_token_counts() {
        assert_eq!(WAN_1_3B.tokens(320, 192, 17), 1_200);
        assert_eq!(WAN_1_3B.tokens(480, 272, 33), 4_590);
        assert_eq!(WAN_1_3B.tokens(480, 272, 81), 10_710);
        assert_eq!(WAN_1_3B.tokens(832, 480, 81), 32_760);
        assert_eq!(PIXART_0_6B.tokens(512, 512, 1), 1_024);
        // an image model ignores the frame count
        assert_eq!(PIXART_0_6B.tokens(512, 512, 81), 1_024);
    }

    /// The per-token figure the plan quotes (41.7 M) and the model's size class.
    #[test]
    fn plan_cost_model() {
        let m = WAN_1_3B.linear_macs_per_token(1.0);
        assert!((m / 1e6 - 41.7).abs() < 0.1, "{m}");
        let p = WAN_1_3B.block_params();
        assert!((1.3e9..1.5e9).contains(&p), "{p}");
        let p = PIXART_0_6B.block_params();
        assert!((0.55e9..0.65e9).contains(&p), "{p}");
    }

    #[test]
    fn experts_only_shrink_the_feed_forward_part() {
        let s = WAN_1_3B;
        let dense = s.macs(4590, 0, 1.0);
        let moe = s.macs(4590, 0, 0.25);
        assert_eq!(dense.attention, moe.attention);
        let saved = 1.0 - moe.total() / dense.total();
        // about a third of a step at 4,590 tokens, as the plan says
        assert!((0.30..0.40).contains(&saved), "{saved}");
    }
}
