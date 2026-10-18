//! Tiny Transformer encoder over `cqlc-core` lexer tokens that predicts CQL performance bottlenecks.
//!
//! * [`vocab`] / [`tokenizer`]: the zero-overhead tokenizer (lexer tokens -> fixed integer ids)
//! * [`features`]: data-shape context and statement facts
//! * [`model`]: the Transformer encoder with a bottleneck head and a cost head
//! * [`io`]: safetensors + config persistence

pub mod features;
pub mod io;
pub mod model;
pub mod tokenizer;
pub mod vocab;

/// Performance bottleneck classes, in output-head order. They are *measured* outcomes (see `cql-hpc-collect`),
/// not rules: a statement carries a label when its measured behaviour on real data showed it.
pub const LABELS: [&str; 6] = [
    "LargeResult",
    "ServerScan",
    "FanOut",
    "TombstoneScan",
    "WriteCost",
    "LightweightTxn",
];

/// Number of bottleneck labels.
pub const N_LABELS: usize = LABELS.len();

/// One-line explanation per label, in the same order as [`LABELS`].
pub const LABEL_HELP: [&str; 6] = [
    "cost is dominated by the number of rows/bytes returned (materialisation, serialisation, network): add a LIMIT, page, or project fewer columns",
    "the server scans much more than it returns (count, filtering, wide partition slices): model a table around this access pattern",
    "the statement touches many partitions or nodes (IN lists, token/full scans, secondary indexes): scatter-gather latency grows with the cluster",
    "reads walk over tombstones of deleted rows: avoid delete-heavy partitions, use TTL/time windows, or compact",
    "a write is several times more expensive than a plain insert (indexes, TTL, batches across partitions): reduce write amplification",
    "a lightweight transaction pays a Paxos round trip: avoid it on hot paths or partition the contention away",
];
