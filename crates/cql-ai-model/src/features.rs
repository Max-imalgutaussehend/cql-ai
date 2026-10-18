//! Numeric context of a statement: data shape (how big is what it touches) and literals the token ids cannot carry.

use cqlc_core::lexer::{TokenKind, lex};
use cqlc_core::span::FileId;
use serde::{Deserialize, Serialize};

/// Length of the context vector fed to the model.
pub const N_CTX: usize = 12;

/// What is known about the data a statement touches. Everything is optional: CI often knows nothing,
/// and the model was trained with the whole block masked out part of the time.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DataStats {
    /// Number of partitions in the table.
    pub partitions: Option<f64>,
    /// Typical live rows per partition.
    pub rows_per_partition: Option<f64>,
    /// Typical stored bytes per row.
    pub row_bytes: Option<f64>,
    /// Total rows (defaults to `partitions * rows_per_partition`).
    pub table_rows: Option<f64>,
    /// The table is delete-heavy (many tombstones before compaction).
    #[serde(default)]
    pub tombstones: bool,
    /// The table has a secondary index.
    #[serde(default)]
    pub indexed: bool,
}

impl DataStats {
    pub fn is_known(&self) -> bool {
        self.partitions.is_some()
            || self.rows_per_partition.is_some()
            || self.row_bytes.is_some()
            || self.table_rows.is_some()
    }
}

/// Facts read off the statement text.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Facts {
    /// The `LIMIT` value, if it is a literal.
    pub limit: Option<u64>,
    /// Number of values in an `IN (...)` list (`IN ?` counts as 8).
    pub in_len: usize,
    /// Rows written together in a batch (only known from an annotation).
    pub batch_n: usize,
}

/// Reads `LIMIT n` and `IN (...)` from a statement.
pub fn facts(src: &str) -> Facts {
    let (lexed, _) = lex(FileId(0), src);
    let toks: Vec<&TokenKind> = lexed.tokens.iter().map(|t| &t.kind).collect();
    let mut f = Facts::default();
    for (i, t) in toks.iter().enumerate() {
        let TokenKind::Ident(w) = t else { continue };
        match w.to_ascii_uppercase().as_str() {
            "LIMIT" => {
                if let Some(TokenKind::Number(n)) = toks.get(i + 1) {
                    f.limit = n.parse().ok();
                }
            }
            "IN" => match toks.get(i + 1) {
                Some(TokenKind::LParen) => {
                    let (mut depth, mut commas, mut any) = (0usize, 0usize, false);
                    for t in &toks[i + 1..] {
                        match t {
                            TokenKind::LParen => depth += 1,
                            TokenKind::RParen => {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
                            }
                            TokenKind::Comma if depth == 1 => commas += 1,
                            TokenKind::Eof => break,
                            _ => any = true,
                        }
                    }
                    f.in_len = if any { commas + 1 } else { 0 };
                }
                Some(TokenKind::Bind) => f.in_len = 8,
                _ => {}
            },
            _ => {}
        }
    }
    f
}

fn lg(x: f64) -> f32 {
    x.max(1.0).log10() as f32
}

/// The model's context vector. With `stats == None` (or unknown stats) the data-shape block is zero and `has_stats` is 0.
pub fn context(stats: Option<&DataStats>, facts: &Facts) -> [f32; N_CTX] {
    let mut v = [0f32; N_CTX];
    if let Some(s) = stats.filter(|s| s.is_known()) {
        let parts = s.partitions;
        let rpp = s.rows_per_partition;
        let rows = s.table_rows.or_else(|| parts.zip(rpp).map(|(p, r)| p * r));
        v[0] = 1.0;
        v[1] = parts.map_or(0.0, |p| lg(p) / 7.0);
        v[2] = rpp.map_or(0.0, |r| lg(r) / 5.0);
        v[3] = s.row_bytes.map_or(0.0, |b| lg(b) / 4.0);
        v[4] = rows.map_or(0.0, |r| lg(r) / 7.0);
        v[5] = f32::from(s.tombstones);
        v[6] = f32::from(s.indexed);
    }
    v[7] = f32::from(facts.limit.is_some());
    v[8] = facts.limit.map_or(0.0, |l| lg(l as f64 + 1.0) / 4.0);
    v[9] = lg(facts.in_len as f64 + 1.0) / 2.0;
    v[10] = lg(facts.batch_n as f64 + 1.0) / 2.0;
    v[11] = 1.0;
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_limit_and_in_lists() {
        assert_eq!(
            facts("SELECT * FROM t WHERE k = ? LIMIT 100").limit,
            Some(100)
        );
        assert_eq!(
            facts("SELECT * FROM t WHERE k IN (?, ?, ?) LIMIT 5").in_len,
            3
        );
        assert_eq!(facts("SELECT * FROM t WHERE k IN ?").in_len, 8);
        assert_eq!(facts("SELECT * FROM t WHERE k = ?"), Facts::default());
    }

    #[test]
    fn unknown_stats_zero_the_data_block() {
        let c = context(
            None,
            &Facts {
                limit: Some(10),
                ..Facts::default()
            },
        );
        assert_eq!(&c[..7], &[0.0; 7]);
        assert!(c[7] == 1.0 && c[8] > 0.0);
        let s = DataStats {
            partitions: Some(1000.0),
            rows_per_partition: Some(100.0),
            ..Default::default()
        };
        let c = context(Some(&s), &Facts::default());
        assert_eq!(c[0], 1.0);
        assert!(
            (c[4] - 5.0 / 7.0).abs() < 1e-6,
            "table rows derived from partitions * rows"
        );
    }
}
