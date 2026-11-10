//! # ditto-core
//!
//! Image and video generation for phones at night, built on `rie-core`.
//!
//! Phase 0 of `docs/PLAN.md` lives here: the shapes of the candidate models, the cost
//! model the plan's estimates come from, and one synthetic denoiser block that can be
//! timed on a phone before anything else is built.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

pub mod attention;
pub mod bench;
pub mod block;
pub mod math;
pub mod shapes;
