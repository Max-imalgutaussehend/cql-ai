//! Fixed vocabulary on top of the `cqlc-core` lexer.
//!
//! No text tokenizer is involved: every [`cqlc_core::lexer::TokenKind`] maps to a fixed id, CQL
//! keywords have their own ids, and identifiers become either a *role token* (when the schema
//! says what the column is) or one of [`HASH_BUCKETS`] hash buckets.

use cqlc_core::lexer::TokenKind;

pub const PAD: u32 = 0;
pub const CLS: u32 = 1;
pub const UNK: u32 = 2;

const PUNCT_BASE: u32 = 4;
const KEYWORD_BASE: u32 = 32;

/// CQL keywords that matter for query shape. Lower-case; matching is case-insensitive.
pub const KEYWORDS: &[&str] = &[
    "select",
    "from",
    "where",
    "and",
    "or",
    "insert",
    "into",
    "values",
    "update",
    "set",
    "delete",
    "if",
    "exists",
    "not",
    "in",
    "order",
    "by",
    "asc",
    "desc",
    "limit",
    "allow",
    "filtering",
    "using",
    "ttl",
    "timestamp",
    "token",
    "contains",
    "like",
    "is",
    "null",
    "true",
    "false",
    "batch",
    "begin",
    "apply",
    "distinct",
    "group",
    "per",
    "partition",
    "json",
    "writetime",
    "as",
    "cast",
    "count",
    "sum",
    "min",
    "max",
    "avg",
    "now",
    "uuid",
    "totimestamp",
    "todate",
    "mintimeuuid",
    "maxtimeuuid",
    "unixtimestampof",
    "dateof",
    "ascii",
    "bigint",
    "blob",
    "boolean",
    "counter",
    "date",
    "decimal",
    "double",
    "float",
    "inet",
    "int",
    "smallint",
    "text",
    "time",
    "timeuuid",
    "tinyint",
    "varchar",
    "varint",
    "frozen",
    "list",
    "map",
    "key",
    "entries",
    "keys",
    "unlogged",
    "logged",
    "add",
    "remove",
];

const ROLE_BASE: u32 = 160;
pub const TABLE: u32 = ROLE_BASE;
pub const COL_PK: u32 = ROLE_BASE + 1;
/// `COL_CK0 ..= COL_CK3`: clustering column by position (capped at 3).
pub const COL_CK0: u32 = ROLE_BASE + 2;
pub const COL_REG: u32 = ROLE_BASE + 6;
pub const COL_STATIC: u32 = ROLE_BASE + 7;
pub const COL_LIST: u32 = ROLE_BASE + 8;
pub const COL_SET: u32 = ROLE_BASE + 9;
pub const COL_MAP: u32 = ROLE_BASE + 10;
pub const COL_COUNTER: u32 = ROLE_BASE + 11;
pub const COL_UNKNOWN: u32 = ROLE_BASE + 12;
/// `TABLE` is used for single-column partition keys; these mark tables with 2 and 3+ partition columns.
pub const TABLE_PK2: u32 = ROLE_BASE + 13;
pub const TABLE_PK3: u32 = ROLE_BASE + 14;

pub const HASH_BUCKETS: u32 = 64;
const HASH_BASE: u32 = 176;

/// Total ids in use; the embedding table is padded to [`VOCAB_SIZE`].
pub const USED_IDS: u32 = HASH_BASE + HASH_BUCKETS;
pub const VOCAB_SIZE: usize = 256;

/// Id of a keyword, if `word` is one.
pub fn keyword_id(word: &str) -> Option<u32> {
    let w = word.to_ascii_lowercase();
    KEYWORDS
        .iter()
        .position(|k| *k == w)
        .map(|i| KEYWORD_BASE + i as u32)
}

/// Id of a hash bucket for an arbitrary identifier (FNV-1a, case-insensitive).
pub fn bucket_id(ident: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in ident.bytes() {
        h ^= u32::from(b.to_ascii_lowercase());
        h = h.wrapping_mul(0x0100_0193);
    }
    HASH_BASE + h % HASH_BUCKETS
}

/// Id for every token kind that is not an identifier.
pub fn kind_id(kind: &TokenKind) -> u32 {
    use TokenKind::*;
    PUNCT_BASE
        + match kind {
            LParen => 0,
            RParen => 1,
            LBracket => 2,
            RBracket => 3,
            LBrace => 4,
            RBrace => 5,
            Comma => 6,
            Semi => 7,
            Dot => 8,
            Star => 9,
            Plus => 10,
            Minus => 11,
            Colon => 12,
            Eq => 13,
            Ne => 14,
            Lt => 15,
            Le => 16,
            Gt => 17,
            Ge => 18,
            Bind => 19,
            Number(_) => 20,
            String(_) => 21,
            Uuid(_) => 22,
            Blob(_) => 23,
            // Identifiers are resolved by the caller; keep the arm total.
            Ident(_) | QuotedIdent(_) => return UNK,
            Eof => return UNK,
        }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_disjoint_and_in_range() {
        const { assert!(PUNCT_BASE + 24 <= KEYWORD_BASE) };
        const { assert!(USED_IDS as usize <= VOCAB_SIZE) };
        assert!(KEYWORD_BASE + KEYWORDS.len() as u32 <= ROLE_BASE);
        assert!(bucket_id("anything") >= HASH_BASE && bucket_id("anything") < USED_IDS);
    }

    #[test]
    fn keywords_are_case_insensitive_and_unique() {
        assert_eq!(keyword_id("SELECT"), keyword_id("select"));
        let mut s = KEYWORDS.to_vec();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), KEYWORDS.len());
    }
}
