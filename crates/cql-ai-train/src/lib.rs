#![allow(clippy::needless_range_loop)] // index loops over the label axis are intentional
//! Dataset loading, labelling from measurements, training and evaluation for cql-ai.

pub mod baseline;
pub mod dataset;
pub mod labels;
pub mod train;
