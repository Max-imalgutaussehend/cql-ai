//! Bottleneck labels derived from *measurements* (`cql-hpc-collect` output). The thresholds are explicit and
//! documented; everything is relative to a baseline measured on the same table, so labels do not depend on the
//! hardware the data was collected on.

use crate::dataset::Raw;
use cql_ai_model::N_LABELS;

/// Thresholds of the labelling rules.
#[derive(Clone, Copy, Debug)]
pub struct Thresholds {
    /// A read is "slow" when it takes at least this many times a point read.
    pub slow: f64,
    /// Rows/bytes returned above which a slow read counts as a large result.
    pub rows: f64,
    pub bytes: f64,
    /// `p50 / (same statement with LIMIT 1)` above which the result size explains the cost.
    pub limit1: f64,
    /// `p50 / (same statement on a clean twin table)` above which tombstones explain the cost.
    pub twin: f64,
    /// A write is expensive when it takes at least this many times a plain insert.
    pub write: f64,
    /// An LWT is flagged when it takes at least this many times a plain insert.
    pub lwt: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            slow: 3.0,
            rows: 100.0,
            bytes: 32_768.0,
            limit1: 2.5,
            twin: 2.0,
            write: 1.8,
            lwt: 1.5,
        }
    }
}

/// `p50 / baseline`: relative to a point read for reads, to a plain insert for writes.
pub fn ratio(r: &Raw) -> f64 {
    let base = if r.write {
        r.base_write_us
    } else {
        r.base_read_us
    };
    if base.is_finite() && base > 0.0 {
        r.p50_us / base
    } else {
        f64::NAN
    }
}

/// Label vector in `cql_ai_model::LABELS` order.
pub fn label(r: &Raw, t: &Thresholds) -> [bool; N_LABELS] {
    let ratio = ratio(r);
    let mut l = [false; N_LABELS];
    if ratio.is_nan() {
        return l;
    }
    if !r.write {
        let slow = ratio >= t.slow;
        let big = r.rows_returned >= t.rows || r.bytes_est >= t.bytes;
        // result size explains the cost if cutting the result to one row removes most of it
        let size_explains = r
            .limit1_p50_us
            .map_or(r.rows_returned >= t.rows, |l1| r.p50_us / l1 >= t.limit1);
        l[0] = slow && big && size_explains;
        l[1] = slow && !big;
        l[2] = slow && r.partitions_touched > 1;
        l[3] = r.twin_p50_us.is_some_and(|tw| r.p50_us / tw >= t.twin) && ratio >= 2.0;
    } else {
        l[4] = !r.lwt && ratio >= t.write;
        l[5] = r.lwt && ratio >= t.lwt;
    }
    l
}

/// Regression target: `log10(ratio)`.
pub fn cost_target(r: &Raw) -> f32 {
    ratio(r).max(0.05).log10() as f32
}
