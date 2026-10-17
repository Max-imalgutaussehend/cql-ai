//! Tiny Transformer encoder over `cqlc-core` lexer tokens that flags CQL anti-patterns.
//!
//! * [`vocab`] / [`tokenizer`]: the zero-overhead tokenizer (lexer tokens -> fixed integer ids)
//! * [`model`]: the Transformer encoder and its multi-label head
//! * [`io`]: safetensors + config persistence

pub mod io;
pub mod model;
pub mod tokenizer;
pub mod vocab;

/// Anti-pattern classes, in output-head order.
pub const LABELS: [&str; 6] = [
    "MissingPartitionKey",
    "InefficientAllowFiltering",
    "NonKeyRestriction",
    "ClusteringGap",
    "OrderByMismatch",
    "LargeCollectionUpdate",
];

/// One-line explanation per label, in the same order as [`LABELS`].
pub const LABEL_HELP: [&str; 6] = [
    "partition key not fully restricted: the query can fan out across partitions (hot or scatter-gather reads)",
    "ALLOW FILTERING scans data and discards rows; add a table or index shaped for the query",
    "restriction on a non-key column needs an index or ALLOW FILTERING",
    "clustering restriction skips an earlier clustering column",
    "ORDER BY does not follow the table's clustering order",
    "assigning a whole collection rewrites it (tombstones, large mutations); append/remove elements instead",
];
