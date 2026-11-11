//! # ditto-vision
//!
//! What a picture contains, as far as code can tell without a model, and flat pictures
//! (screenshots, diagrams, logos) as exact SVG.
//!
//! Everything here is deterministic and runs in milliseconds, so it can run the moment a
//! picture arrives, in agenticOS's Active stage. What needs a model (reading text,
//! naming objects, describing a scene) is planned in `docs/USE_CASES.md` and is **not**
//! here: an [`Observation`] never contains a guess presented as a fact.

#![allow(clippy::needless_range_loop)]

pub mod codes;
pub mod fidelity;
pub mod load;
pub mod observe;
pub mod provenance;
pub mod scripts;
pub mod stats;
pub mod trace;

pub use load::{Loaded, Meta};
pub use observe::{observe, Observation};

#[cfg(test)]
pub(crate) mod testing;
