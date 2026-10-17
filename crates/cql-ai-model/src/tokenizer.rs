//! Statement tokenizer: `cqlc-core` lexer + schema roles -> integer ids.

use crate::vocab::*;
use cqlc_core::lexer::{TokenKind, lex};
use cqlc_core::schema::{ColumnRole, Schema};
use cqlc_core::span::{FileId, SourceFile};
use cqlc_core::types::CqlType;
use std::collections::HashMap;

/// Longest encoded sequence including the leading `CLS`.
pub const MAX_LEN: usize = 96;

/// Role of a column, as seen by the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Partition,
    Clustering(u8),
    Regular,
    Static,
    List,
    Set,
    Map,
    Counter,
}

impl Role {
    fn token(self) -> u32 {
        match self {
            Role::Partition => COL_PK,
            Role::Clustering(i) => COL_CK0 + u32::from(i.min(3)),
            Role::Regular => COL_REG,
            Role::Static => COL_STATIC,
            Role::List => COL_LIST,
            Role::Set => COL_SET,
            Role::Map => COL_MAP,
            Role::Counter => COL_COUNTER,
        }
    }
}

/// Column roles of every table, used to turn identifiers into role tokens.
#[derive(Clone, Debug, Default)]
pub struct SchemaInfo {
    tables: HashMap<String, HashMap<String, Role>>,
}

impl SchemaInfo {
    /// Builds the role map from a resolved `cqlc-core` schema.
    pub fn from_schema(schema: &Schema) -> Self {
        let mut tables = HashMap::new();
        for t in &schema.tables {
            let mut cols = HashMap::new();
            for c in &t.columns {
                let role = match c.role {
                    ColumnRole::PartitionKey => Role::Partition,
                    ColumnRole::ClusteringKey => {
                        let i = t
                            .clustering
                            .iter()
                            .position(|k| k.name == c.name)
                            .unwrap_or(0);
                        Role::Clustering(i as u8)
                    }
                    _ if c.is_static => Role::Static,
                    _ => match c.cql_type.unfrozen() {
                        CqlType::List(_) => Role::List,
                        CqlType::Set(_) => Role::Set,
                        CqlType::Map(..) => Role::Map,
                        CqlType::Counter => Role::Counter,
                        _ => Role::Regular,
                    },
                };
                cols.insert(c.name.to_ascii_lowercase(), role);
            }
            tables.insert(t.name.to_ascii_lowercase(), cols);
        }
        SchemaInfo { tables }
    }

    /// Parses DDL text (one or more `CREATE TABLE`s) with `cqlc-core`.
    pub fn from_ddl(ddl: &str) -> Self {
        let c = cqlc_core::compile(&[cqlc_core::Input::new("schema.cql", ddl)], &[]);
        Self::from_schema(&c.schema)
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }
}

/// One encoded statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Encoded {
    /// `CLS` followed by token ids, at most [`MAX_LEN`] long.
    pub ids: Vec<u32>,
    /// 1-based line of the first token.
    pub line: usize,
    /// 1-based column of the first token.
    pub column: usize,
}

/// Location of one statement inside a source text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatementSpan {
    /// Byte range of the statement (without the terminating `;`).
    pub range: std::ops::Range<usize>,
    /// 1-based line of the first token.
    pub line: usize,
    /// 1-based column of the first token.
    pub column: usize,
}

/// Splits `src` at `;` into non-empty statements (comments are ignored by the lexer).
pub fn statements(src: &str) -> Vec<StatementSpan> {
    let (lexed, _diags) = lex(FileId(0), src);
    let file = SourceFile::new("input", src);
    let mut out = Vec::new();
    let mut start = 0;
    let toks = &lexed.tokens;
    for (i, t) in toks.iter().enumerate() {
        if !matches!(t.kind, TokenKind::Semi | TokenKind::Eof) {
            continue;
        }
        let stmt = &toks[start..i];
        if let (Some(first), Some(last)) = (stmt.first(), stmt.last()) {
            let (line, column) = file.line_col(first.span.start as usize);
            out.push(StatementSpan {
                range: first.span.start as usize..last.span.end as usize,
                line,
                column,
            });
        }
        start = i + 1;
    }
    out
}

/// Splits `src` at `;` and encodes each non-empty statement.
pub fn encode_file(src: &str, schema: &SchemaInfo) -> Vec<Encoded> {
    statements(src)
        .into_iter()
        .map(|s| Encoded {
            ids: encode(&src[s.range], schema),
            line: s.line,
            column: s.column,
        })
        .collect()
}

/// Encodes a single statement given as text (a trailing `;` is fine).
pub fn encode(src: &str, schema: &SchemaInfo) -> Vec<u32> {
    let (lexed, _) = lex(FileId(0), src);
    let kinds: Vec<&TokenKind> = lexed
        .tokens
        .iter()
        .map(|t| &t.kind)
        .filter(|k| !matches!(k, TokenKind::Eof | TokenKind::Semi))
        .collect();
    encode_kinds(&kinds, schema)
}

fn ident_text(k: &TokenKind) -> Option<&str> {
    match k {
        TokenKind::Ident(s) | TokenKind::QuotedIdent(s) => Some(s),
        _ => None,
    }
}

/// Finds the statement's table: the identifier after `FROM`, `UPDATE` or `INTO` (`ks.table` allowed).
fn find_table<'a>(kinds: &[&'a TokenKind]) -> Option<(usize, &'a str)> {
    for (i, k) in kinds.iter().enumerate() {
        let Some(w) = ident_text(k) else { continue };
        if !matches!(w.to_ascii_lowercase().as_str(), "from" | "update" | "into") {
            continue;
        }
        let mut j = i + 1;
        let mut name = ident_text(kinds.get(j)?)?;
        while matches!(kinds.get(j + 1), Some(TokenKind::Dot)) {
            name = ident_text(kinds.get(j + 2)?)?;
            j += 2;
        }
        return Some((j, name));
    }
    None
}

fn encode_kinds(kinds: &[&TokenKind], schema: &SchemaInfo) -> Vec<u32> {
    let table = find_table(kinds);
    let cols = table.and_then(|(_, t)| schema.tables.get(&t.to_ascii_lowercase()));
    let mut ids = Vec::with_capacity(kinds.len() + 1);
    ids.push(CLS);
    for (i, k) in kinds.iter().enumerate() {
        let id = match ident_text(k) {
            Some(word) => {
                let lower = word.to_ascii_lowercase();
                if table.is_some_and(|(at, _)| at == i) {
                    TABLE
                } else if let Some(role) = cols.and_then(|c| c.get(&lower)) {
                    role.token()
                } else if let Some(kw) = matches!(k, TokenKind::Ident(_))
                    .then(|| keyword_id(word))
                    .flatten()
                {
                    kw
                } else if cols.is_some() {
                    COL_UNKNOWN
                } else {
                    bucket_id(word)
                }
            }
            None => kind_id(k),
        };
        ids.push(id);
        if ids.len() == MAX_LEN {
            break;
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    const DDL: &str = "CREATE TABLE app.events (tenant text, day int, ts bigint, body text, tags set<text>, \
                       PRIMARY KEY ((tenant, day), ts));";

    #[test]
    fn roles_replace_identifiers() {
        let schema = SchemaInfo::from_ddl(DDL);
        let ids = encode(
            "SELECT body FROM events WHERE tenant = ? AND ts > ? ALLOW FILTERING",
            &schema,
        );
        assert_eq!(ids[0], CLS);
        assert!(ids.contains(&COL_PK));
        assert!(ids.contains(&COL_CK0));
        assert!(ids.contains(&COL_REG));
        assert!(ids.contains(&TABLE));
        assert!(ids.contains(&keyword_id("filtering").unwrap()));
    }

    #[test]
    fn collections_and_unknown_columns() {
        let schema = SchemaInfo::from_ddl(DDL);
        let ids = encode("UPDATE events SET tags = {'a'} WHERE nope = 1", &schema);
        assert!(ids.contains(&COL_SET));
        assert!(ids.contains(&COL_UNKNOWN));
    }

    #[test]
    fn without_schema_identifiers_are_hashed() {
        let a = encode("SELECT user_id FROM users", &SchemaInfo::default());
        let b = encode("select USER_ID from USERS", &SchemaInfo::default());
        assert_eq!(a, b);
    }

    #[test]
    fn file_is_split_with_line_numbers() {
        let src = "SELECT a FROM t WHERE k = ?;\n\n-- c\nSELECT b FROM t WHERE k = ?;\n";
        let e = encode_file(src, &SchemaInfo::default());
        assert_eq!(e.len(), 2);
        assert_eq!((e[0].line, e[1].line), (1, 4));
    }

    #[test]
    fn long_statements_are_truncated() {
        let sql = format!("SELECT {} FROM t", vec!["a"; 500].join(", "));
        assert_eq!(encode(&sql, &SchemaInfo::default()).len(), MAX_LEN);
    }
}
