//! Randomised AST -> CQL text generator. Labels come from running `cqlc-core`'s analyzer.

use cql_ai_model::LABELS;
use cql_ai_model::tokenizer::{SchemaInfo, encode};
use cqlc_core::diagnostic::DiagnosticCode as C;
use cqlc_core::{Input, compile};
use rand::prelude::*;
use serde::Serialize;
use std::collections::BTreeMap;

pub const N_LABELS: usize = LABELS.len();

const COLUMN_NAMES: &[&str] = &[
    "user_id",
    "order_id",
    "tenant",
    "region",
    "device",
    "sensor",
    "day",
    "bucket",
    "ts",
    "created",
    "seq",
    "name",
    "email",
    "status_code",
    "city",
    "country",
    "price",
    "qty",
    "score",
    "balance",
    "session_id",
    "page",
    "campaign",
    "channel",
    "shard",
    "version",
    "owner",
    "title",
    "body",
    "payload",
    "ip_addr",
    "lat",
    "lon",
    "product_id",
    "category",
    "level",
    "source",
    "event_type",
    "account",
    "group_id",
    "item_id",
    "reading",
    "unit",
    "label",
    "note",
    "color",
    "size_cm",
];
const TABLE_NAMES: &[&str] = &[
    "events",
    "users",
    "orders",
    "sensors",
    "sessions",
    "messages",
    "readings",
    "accounts",
    "products",
    "metrics",
    "audit_log",
    "carts",
    "invoices",
    "devices",
    "feeds",
    "ratings",
];

#[derive(Clone, Debug, PartialEq)]
enum Ty {
    Text,
    Int,
    BigInt,
    Uuid,
    Boolean,
    Double,
    Timestamp,
    List(Box<Ty>),
    Set(Box<Ty>),
    Map(Box<Ty>, Box<Ty>),
}

impl Ty {
    fn cql(&self) -> String {
        match self {
            Ty::Text => "text".into(),
            Ty::Int => "int".into(),
            Ty::BigInt => "bigint".into(),
            Ty::Uuid => "uuid".into(),
            Ty::Boolean => "boolean".into(),
            Ty::Double => "double".into(),
            Ty::Timestamp => "timestamp".into(),
            Ty::List(t) => format!("list<{}>", t.cql()),
            Ty::Set(t) => format!("set<{}>", t.cql()),
            Ty::Map(k, v) => format!("map<{}, {}>", k.cql(), v.cql()),
        }
    }
    fn is_collection(&self) -> bool {
        matches!(self, Ty::List(_) | Ty::Set(_) | Ty::Map(..))
    }
}

#[derive(Clone, Debug)]
struct Col {
    name: String,
    ty: Ty,
}

#[derive(Clone, Debug)]
struct Table {
    name: String,
    cols: Vec<Col>,
    pk: Vec<usize>,
    ck: Vec<(usize, bool)>, // (column, descending)
}

impl Table {
    fn is_key(&self, i: usize) -> bool {
        self.pk.contains(&i) || self.ck.iter().any(|(c, _)| *c == i)
    }
    fn ddl(&self) -> String {
        let cols: Vec<String> = self
            .cols
            .iter()
            .map(|c| format!("{} {}", c.name, c.ty.cql()))
            .collect();
        let pk: Vec<_> = self.pk.iter().map(|&i| self.cols[i].name.clone()).collect();
        let mut key = format!("({})", pk.join(", "));
        for (i, _) in &self.ck {
            key.push_str(&format!(", {}", self.cols[*i].name));
        }
        let mut s = format!(
            "CREATE TABLE {} ({}, PRIMARY KEY ({key}))",
            self.name,
            cols.join(", ")
        );
        if !self.ck.is_empty() {
            let o: Vec<_> = self
                .ck
                .iter()
                .map(|(i, d)| format!("{} {}", self.cols[*i].name, if *d { "DESC" } else { "ASC" }))
                .collect();
            s.push_str(&format!(" WITH CLUSTERING ORDER BY ({})", o.join(", ")));
        }
        s.push(';');
        s
    }
}

/// A random keyspace schema plus its role map.
pub struct World {
    tables: Vec<Table>,
    pub ddl: String,
    pub info: SchemaInfo,
}

fn key_ty(rng: &mut impl Rng) -> Ty {
    [Ty::Text, Ty::Int, Ty::BigInt, Ty::Uuid, Ty::Timestamp]
        .choose(rng)
        .unwrap()
        .clone()
}

fn scalar_ty(rng: &mut impl Rng) -> Ty {
    [
        Ty::Text,
        Ty::Int,
        Ty::BigInt,
        Ty::Uuid,
        Ty::Boolean,
        Ty::Double,
        Ty::Timestamp,
    ]
    .choose(rng)
    .unwrap()
    .clone()
}

fn coll_ty(rng: &mut impl Rng) -> Ty {
    let elem = |rng: &mut dyn RngCore| [Ty::Text, Ty::Int, Ty::BigInt].choose(rng).unwrap().clone();
    match rng.random_range(0..3) {
        0 => Ty::List(Box::new(elem(rng))),
        1 => Ty::Set(Box::new(elem(rng))),
        _ => Ty::Map(Box::new(Ty::Text), Box::new(elem(rng))),
    }
}

impl World {
    pub fn random(rng: &mut impl Rng) -> World {
        let mut tnames = TABLE_NAMES.to_vec();
        tnames.shuffle(rng);
        let n_tables = rng.random_range(2..=5);
        let mut tables = Vec::new();
        for name in tnames.into_iter().take(n_tables) {
            let mut names = COLUMN_NAMES.to_vec();
            names.shuffle(rng);
            let n_pk = *[1, 1, 1, 2, 2, 3].choose(rng).unwrap();
            let n_ck = *[0, 1, 1, 2, 2, 3].choose(rng).unwrap();
            let n_reg = rng.random_range(2..=6);
            let mut cols = Vec::new();
            for (i, n) in names.iter().take(n_pk + n_ck + n_reg).enumerate() {
                let ty = if i < n_pk + n_ck {
                    key_ty(rng)
                } else if rng.random_bool(0.3) {
                    coll_ty(rng)
                } else {
                    scalar_ty(rng)
                };
                cols.push(Col {
                    name: (*n).to_string(),
                    ty,
                });
            }
            let pk = (0..n_pk).collect();
            let ck = (n_pk..n_pk + n_ck)
                .map(|i| (i, rng.random_bool(0.4)))
                .collect();
            tables.push(Table {
                name: name.to_string(),
                cols,
                pk,
                ck,
            });
        }
        let ddl = tables.iter().map(Table::ddl).collect::<Vec<_>>().join("\n");
        let info = SchemaInfo::from_ddl(&ddl);
        World { tables, ddl, info }
    }
}

fn hex(rng: &mut impl Rng, n: usize) -> String {
    (0..n)
        .map(|_| char::from_digit(rng.random_range(0..16), 16).unwrap())
        .collect()
}

fn literal(rng: &mut impl Rng, ty: &Ty) -> String {
    if rng.random_bool(0.6) {
        return "?".into();
    }
    match ty {
        Ty::Text => format!(
            "'{}'",
            ["a", "acme", "eu", "x1", "hello"].choose(rng).unwrap()
        ),
        Ty::Int | Ty::BigInt => rng.random_range(0..1000).to_string(),
        Ty::Boolean => rng.random_bool(0.5).to_string(),
        Ty::Double => format!("{:.1}", rng.random_range(0.0..100.0)),
        Ty::Uuid => {
            let parts: Vec<String> = [8, 4, 4, 4, 12].iter().map(|&n| hex(rng, n)).collect();
            parts.join("-")
        }
        _ => "?".into(),
    }
}

struct Where {
    text: String,
    non_key: bool,
    gap_or_missing: bool,
}

fn gen_where(rng: &mut impl Rng, t: &Table) -> Where {
    let mut conds: Vec<String> = Vec::new();
    let mut non_key = false;
    let mut gap_or_missing = false;
    let drop_pk = rng
        .random_bool(0.15)
        .then(|| rng.random_range(0..t.pk.len()));
    for (n, &i) in t.pk.iter().enumerate() {
        if drop_pk == Some(n) {
            gap_or_missing = true;
            continue;
        }
        conds.push(format!(
            "{} = {}",
            t.cols[i].name,
            literal(rng, &t.cols[i].ty)
        ));
    }
    if !t.ck.is_empty() && rng.random_bool(0.6) {
        let k = rng.random_range(1..=t.ck.len());
        let skip = (k >= 2 && rng.random_bool(0.4)).then(|| rng.random_range(0..k - 1));
        for (n, (i, _)) in t.ck.iter().take(k).enumerate() {
            if skip == Some(n) {
                gap_or_missing = true;
                continue;
            }
            let col = &t.cols[*i];
            let last = n == k - 1;
            let op = if last && rng.random_bool(0.4) {
                *[">", ">=", "<", "<="].choose(rng).unwrap()
            } else {
                "="
            };
            conds.push(format!("{} {} {}", col.name, op, literal(rng, &col.ty)));
        }
    }
    if rng.random_bool(0.15) {
        let regs: Vec<usize> = (0..t.cols.len())
            .filter(|&i| !t.is_key(i) && !t.cols[i].ty.is_collection())
            .collect();
        if let Some(&i) = regs.choose(rng) {
            conds.push(format!(
                "{} = {}",
                t.cols[i].name,
                literal(rng, &t.cols[i].ty)
            ));
            non_key = true;
        }
    }
    if conds.is_empty() {
        // `WHERE` needs at least one restriction; fall back to the first partition column.
        conds.push(format!("{} = ?", t.cols[t.pk[0]].name));
        gap_or_missing = t.pk.len() > 1;
    }
    Where {
        text: conds.join(" AND "),
        non_key,
        gap_or_missing,
    }
}

/// A generated statement and the label that comes from the generator itself rather than from cqlc.
pub struct Generated {
    pub text: String,
    pub large_collection_update: bool,
}

pub fn gen_statement(rng: &mut impl Rng, w: &World) -> Generated {
    let t = w.tables.choose(rng).unwrap();
    let mut large = false;
    let text = match rng.random_range(0..100) {
        0..=54 => {
            let cols = if rng.random_bool(0.2) {
                "*".to_string()
            } else {
                let mut idx: Vec<usize> = (0..t.cols.len()).collect();
                idx.shuffle(rng);
                idx.truncate(rng.random_range(1..=4.min(t.cols.len())));
                idx.iter()
                    .map(|&i| t.cols[i].name.clone())
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            let wh = gen_where(rng, t);
            let mut s = format!("SELECT {cols} FROM {} WHERE {}", t.name, wh.text);
            if !t.ck.is_empty() && rng.random_bool(0.2) {
                let (i, _) = t.ck[if rng.random_bool(0.7) {
                    0
                } else {
                    rng.random_range(0..t.ck.len())
                }];
                s.push_str(&format!(
                    " ORDER BY {} {}",
                    t.cols[i].name,
                    if rng.random_bool(0.5) { "ASC" } else { "DESC" }
                ));
            }
            if rng.random_bool(0.4) {
                s.push_str(&format!(" LIMIT {}", [10, 100, 1000].choose(rng).unwrap()));
            }
            let p = if wh.non_key || wh.gap_or_missing {
                0.5
            } else {
                0.04
            };
            if rng.random_bool(p) {
                s.push_str(" ALLOW FILTERING");
            }
            format!("-- name: Q :many\n{s};")
        }
        55..=74 => {
            let regs: Vec<usize> = (0..t.cols.len()).filter(|&i| !t.is_key(i)).collect();
            let mut sets = Vec::new();
            let n = rng.random_range(1..=3.min(regs.len()));
            for &i in regs.choose_multiple(rng, n) {
                let c = &t.cols[i];
                let name = &c.name;
                sets.push(match &c.ty {
                    Ty::List(e) | Ty::Set(e) => {
                        let one = literal_elem(rng, e);
                        let (open, close) = if matches!(c.ty, Ty::List(_)) {
                            ("[", "]")
                        } else {
                            ("{", "}")
                        };
                        match rng.random_range(0..4) {
                            0 => {
                                large = true;
                                format!("{name} = {open}{one}, {}{close}", literal_elem(rng, e))
                            }
                            1 => {
                                large = true;
                                format!("{name} = ?")
                            }
                            2 => format!("{name} = {name} + {open}{one}{close}"),
                            _ => format!("{name} = {name} - ?"),
                        }
                    }
                    Ty::Map(_, v) => match rng.random_range(0..3) {
                        0 => {
                            large = true;
                            format!("{name} = {{'k': {}}}", literal_elem(rng, v))
                        }
                        1 => format!("{name} = {name} + ?"),
                        _ => format!("{name}['k'] = {}", literal_elem(rng, v)),
                    },
                    ty => format!("{name} = {}", literal(rng, ty)),
                });
            }
            let mut wh = gen_where(rng, t);
            // UPDATE can only restrict key columns; keep the label semantics of cqlc intact.
            if wh.non_key {
                wh.text = wh
                    .text
                    .rsplit_once(" AND ")
                    .map(|(a, _)| a.to_string())
                    .unwrap_or(wh.text);
            }
            let ttl = if rng.random_bool(0.1) {
                " USING TTL 3600"
            } else {
                ""
            };
            format!(
                "-- name: Q :exec\nUPDATE {}{ttl} SET {} WHERE {};",
                t.name,
                sets.join(", "),
                wh.text
            )
        }
        75..=84 => {
            let wh = gen_where(rng, t);
            let text = if wh.non_key {
                wh.text
                    .rsplit_once(" AND ")
                    .map(|(a, _)| a.to_string())
                    .unwrap_or(wh.text)
            } else {
                wh.text
            };
            format!("-- name: Q :exec\nDELETE FROM {} WHERE {};", t.name, text)
        }
        _ => {
            let drop_key = rng
                .random_bool(0.1)
                .then(|| rng.random_range(0..t.pk.len()));
            let cols: Vec<&Col> = t
                .cols
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    t.is_key(*i) && !(drop_key.is_some_and(|d| t.pk.get(d) == Some(i)))
                        || (!t.is_key(*i) && rng.random_bool(0.6))
                })
                .map(|(_, c)| c)
                .collect();
            let names = cols
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            let vals = vec!["?"; cols.len()].join(", ");
            format!(
                "-- name: Q :exec\nINSERT INTO {} ({names}) VALUES ({vals});",
                t.name
            )
        }
    };
    Generated {
        text,
        large_collection_update: large,
    }
}

fn literal_elem(rng: &mut impl Rng, ty: &Ty) -> String {
    match ty {
        Ty::Text => format!("'{}'", ["a", "b", "c"].choose(rng).unwrap()),
        _ => rng.random_range(0..100).to_string(),
    }
}

/// Maps analyzer diagnostics to label indices (`None` = diagnostic is not an anti-pattern class).
fn label_of(code: C) -> Option<usize> {
    match code {
        C::IncompleteKey | C::BroadScan => Some(0),
        C::AllowFiltering => Some(1),
        C::NonKeyRestriction => Some(2),
        C::ClusteringGap => Some(3),
        C::OrderBy => Some(4),
        _ => None,
    }
}

/// Outcome of labelling one statement.
pub enum Labelled {
    Ok([bool; N_LABELS]),
    /// The statement failed for a reason unrelated to the classes (generator bug or unsupported CQL).
    Rejected(String),
}

pub fn label(w: &World, g: &Generated) -> Labelled {
    let c = compile(
        &[Input::new("schema.cql", w.ddl.clone())],
        &[Input::new("q.cql", g.text.clone())],
    );
    let mut labels = [false; N_LABELS];
    for d in &c.diagnostics {
        match label_of(d.code) {
            Some(i) => labels[i] = true,
            None if d.is_error() && !matches!(d.code, C::InvalidRestriction) => {
                return Labelled::Rejected(format!("{} {}", d.code.as_str(), d.message));
            }
            None => {}
        }
    }
    labels[5] = g.large_collection_update;
    Labelled::Ok(labels)
}

/// Encoded samples.
#[derive(Default)]
pub struct Dataset {
    pub ids: Vec<Vec<u32>>,
    pub labels: Vec<[bool; N_LABELS]>,
    pub texts: Vec<String>,
    pub rejected: BTreeMap<String, usize>,
}

impl Dataset {
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Per-label positive counts.
    pub fn counts(&self) -> [usize; N_LABELS] {
        let mut c = [0; N_LABELS];
        for l in &self.labels {
            for (i, b) in l.iter().enumerate() {
                c[i] += usize::from(*b);
            }
        }
        c
    }
}

/// Generates `n` labelled samples from `n_worlds` random schemas.
pub fn generate(n: usize, n_worlds: usize, seed: u64) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let worlds: Vec<World> = (0..n_worlds).map(|_| World::random(&mut rng)).collect();
    let mut ds = Dataset::default();
    while ds.len() < n {
        let w = worlds.choose(&mut rng).unwrap();
        let g = gen_statement(&mut rng, w);
        match label(w, &g) {
            Labelled::Ok(l) => {
                // The model never sees the `-- name:` annotation (the lexer drops comments).
                ds.ids.push(encode(&g.text, &w.info));
                ds.labels.push(l);
                ds.texts
                    .push(g.text.lines().skip(1).collect::<Vec<_>>().join(" "));
            }
            Labelled::Rejected(why) => {
                let key = why.split(' ').next().unwrap_or("?").to_string();
                *ds.rejected.entry(key).or_default() += 1;
                if ds.rejected.values().sum::<usize>() > 20 * n.max(100) {
                    panic!(
                        "generator produces too many invalid statements, last: {why}\n{}",
                        g.text
                    );
                }
            }
        }
    }
    ds
}

#[derive(Serialize)]
pub struct JsonRow<'a> {
    pub cql: &'a str,
    pub labels: Vec<&'static str>,
}

impl Dataset {
    /// One JSON object per line: statement text and positive label names.
    pub fn to_jsonl(&self) -> String {
        let mut out = String::new();
        for (t, l) in self.texts.iter().zip(&self.labels) {
            let labels = l
                .iter()
                .enumerate()
                .filter(|(_, b)| **b)
                .map(|(i, _)| LABELS[i])
                .collect();
            out.push_str(&serde_json::to_string(&JsonRow { cql: t, labels }).unwrap());
            out.push('\n');
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic_and_covers_all_labels() {
        let a = generate(3000, 20, 7);
        let b = generate(3000, 20, 7);
        assert_eq!(a.ids, b.ids);
        assert_eq!(a.labels, b.labels);
        let counts = a.counts();
        for (i, c) in counts.iter().enumerate() {
            assert!(
                *c > 30,
                "label {} only {} positives of {}: {:?}",
                LABELS[i],
                c,
                a.len(),
                counts
            );
            assert!(
                *c < a.len() - 30,
                "label {} nearly always set: {:?}",
                LABELS[i],
                counts
            );
        }
    }

    #[test]
    fn hand_checked_labels() {
        let mut rng = StdRng::seed_from_u64(1);
        let w = World::random(&mut rng);
        let t = &w.tables[0];
        let pk = &t.cols[t.pk[0]].name;
        let ok = Generated {
            text: format!("-- name: Q :many\nSELECT * FROM {} WHERE {pk} = ?;", t.name),
            large_collection_update: false,
        };
        let Labelled::Ok(l) = label(&w, &ok) else {
            panic!("rejected")
        };
        // Single-column partition keys are fully restricted here; composite ones are not.
        assert_eq!(l[0], t.pk.len() > 1);
        assert!(!l[1] && !l[5]);
    }
}
