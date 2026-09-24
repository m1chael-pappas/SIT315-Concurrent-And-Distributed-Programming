//! citysim: a city traffic simulator whose backends produce bit-identical state.
//!
//! The library holds the model, the backends and the GPU code. The `citysim`
//! binary and the integration tests are its only users.

#![deny(unsafe_code)]

pub mod car;
pub mod checksum;
pub mod city;
pub mod compare;
#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
pub mod gpu;
pub mod grid;
pub mod invariants;
pub mod junction;
pub mod nasch;
pub mod params;
pub mod philox;
pub mod sched;
pub mod sensors;
pub mod signal;
pub mod stats;
