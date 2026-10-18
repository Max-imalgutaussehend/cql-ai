//! `cql-hpc-collect`: builds the cql-hpc benchmark by *measuring* statements on public workload shapes.
//!
//!     cql-hpc-collect --node 127.0.0.1:9042 --engine scylla --out dataset.jsonl [--scale 1.0] [--only nb_tabular]

mod world;

use anyhow::{Result, bail};
use clap::Parser;
use futures::StreamExt;
use rand::prelude::*;
use rand_distr::Zipf;
use scylla::client::session::Session;
use scylla::client::session_builder::SessionBuilder;
use scylla::errors::TranslationError;
use scylla::policies::address_translator::{AddressTranslator, UntranslatedPeer};
use scylla::statement::batch::{Batch, BatchType};
use scylla::value::CqlValue;
use serde::Serialize;
use std::io::Write;
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::Arc;
use std::time::{Duration, Instant};
use world::{Tomb, Ty, World};

struct Pin(SocketAddr);

#[async_trait::async_trait]
impl AddressTranslator for Pin {
    async fn translate_address(
        &self,
        _: &UntranslatedPeer,
    ) -> Result<SocketAddr, TranslationError> {
        Ok(self.0)
    }
}

#[derive(Parser)]
#[command(version, about)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:9042")]
    node: String,
    /// Engine label stored with every sample (`scylla`, `cassandra-5.0`, ...).
    #[arg(long, default_value = "scylla")]
    engine: String,
    #[arg(long, default_value = "dataset.jsonl")]
    out: String,
    /// Multiplies partition counts (0.05 for a smoke run).
    #[arg(long, default_value_t = 1.0)]
    scale: f64,
    /// Only worlds whose id contains this text.
    #[arg(long)]
    only: Option<String>,
    #[arg(long, default_value_t = 40)]
    reps: usize,
}

/// One measured statement.
#[derive(Serialize)]
struct Sample {
    world: String,
    family: &'static str,
    engine: String,
    ddl: String,
    statement: String,
    kind: &'static str,
    // data shape
    partitions: usize,
    rows_per_partition: usize,
    row_bytes: usize,
    table_rows: usize,
    tombstones: bool,
    tomb_mode: &'static str,
    indexed: bool,
    // statement facts
    write: bool,
    lwt: bool,
    limit: Option<u32>,
    in_len: usize,
    batch_n: usize,
    partitions_touched: usize,
    // measurements (microseconds / counts)
    p50_us: f64,
    p90_us: f64,
    p99_us: f64,
    rows_returned: f64,
    bytes_est: f64,
    base_read_us: f64,
    base_write_us: f64,
    twin_p50_us: Option<f64>,
    limit1_p50_us: Option<f64>,
    hot_ratio: Option<f64>,
}

type Gen = Arc<dyn Fn(&mut StdRng) -> Vec<CqlValue> + Send + Sync>;

#[derive(Clone)]
struct Stmt {
    kind: &'static str,
    cql: String,
    pgen: Gen,
    write: bool,
    lwt: bool,
    limit: Option<u32>,
    in_len: usize,
    batch_n: usize,
    partitions_touched: usize,
    /// bytes returned per row (projected columns)
    row_bytes: usize,
    /// run with fewer repetitions (expensive scans)
    heavy: bool,
    /// the same statement with `LIMIT 1`, to attribute cost to result size
    limit1: Option<String>,
}

fn names(cols: &[&world::Col]) -> String {
    cols.iter().map(|c| c.name).collect::<Vec<_>>().join(", ")
}

fn pk_where(w: &World) -> String {
    w.pk.iter()
        .map(|c| format!("{} = ?", c.name))
        .collect::<Vec<_>>()
        .join(" AND ")
}

fn ck_where(w: &World) -> String {
    w.ck.iter()
        .map(|(c, _)| format!("{} = ?", c.name))
        .collect::<Vec<_>>()
        .join(" AND ")
}

/// The statements measured on a world's main table `t`.
fn statements(w: &World) -> Vec<Stmt> {
    let (p, r) = (w.partitions, w.rpp);
    let table = "t";
    let mut out = Vec::new();
    let all_cols: Vec<&world::Col> =
        w.pk.iter()
            .chain(w.ck.iter().map(|(c, _)| c))
            .chain(w.regular.iter())
            .collect();
    let all_bytes = w.row_bytes();
    let has_ck = !w.ck.is_empty();
    let pkw = pk_where(w);
    let full_key = if has_ck {
        format!("{pkw} AND {}", ck_where(w))
    } else {
        pkw.clone()
    };
    let wa = Arc::new(w.clone());

    let key_gen = |full: bool| -> Gen {
        let w = wa.clone();
        Arc::new(move |rng| {
            let (pi, ri) = (
                rng.random_range(0..w.partitions),
                rng.random_range(0..w.rpp),
            );
            let mut v = w.pk_values(pi);
            if full {
                v.extend(w.ck_values(ri));
            }
            v
        })
    };
    let mk = |kind: &'static str, cql: String, pgen: Gen, _rows: usize, parts: usize| Stmt {
        kind,
        cql,
        pgen,
        write: false,
        lwt: false,
        limit: None,
        in_len: 0,
        batch_n: 0,
        partitions_touched: parts,
        row_bytes: all_bytes,
        heavy: false,
        limit1: None,
    };
    if !w.is_counter() {
        out.push(mk(
            "point_read",
            format!("SELECT * FROM {table} WHERE {full_key}"),
            key_gen(true),
            1,
            1,
        ));
        if has_ck {
            let mut s = mk(
                "partition_read",
                format!("SELECT * FROM {table} WHERE {pkw}"),
                key_gen(false),
                r,
                1,
            );
            if r <= 60_000 {
                s.limit1 = Some(format!("SELECT * FROM {table} WHERE {pkw} LIMIT 1"));
                out.push(s);
            }
            for limit in [1u32, 5, 10, 50, 100, 500, 1000, 5000] {
                if (limit as usize) <= r {
                    let mut s = mk(
                        "partition_read_limit",
                        format!("SELECT * FROM {table} WHERE {pkw} LIMIT {limit}"),
                        key_gen(false),
                        limit as usize,
                        1,
                    );
                    s.limit = Some(limit);
                    if limit > 1 {
                        s.limit1 = Some(format!("SELECT * FROM {table} WHERE {pkw} LIMIT 1"));
                    }
                    out.push(s);
                }
            }
            // projection of two regular columns
            if w.regular.len() >= 2 {
                let cols = [&w.regular[0], &w.regular[1]];
                let mut s = mk(
                    "projection_read",
                    format!("SELECT {} FROM {table} WHERE {pkw} LIMIT 100", names(&cols)),
                    key_gen(false),
                    r.min(100),
                    1,
                );
                s.limit = Some(100);
                s.row_bytes = cols.iter().map(|c| c.ty.bytes()).sum();
                out.push(s);
            }
            // clustering range
            let first = &w.ck[0].0;
            if w.ck.len() == 1 {
                let wr = wa.clone();
                let mid = r / 2;
                let pgen: Gen = Arc::new(move |rng| {
                    let mut v = wr.pk_values(rng.random_range(0..wr.partitions));
                    v.extend(wr.ck_values(mid));
                    v
                });
                let mut s = mk(
                    "range_read",
                    format!(
                        "SELECT * FROM {table} WHERE {pkw} AND {} > ? LIMIT 100",
                        first.name
                    ),
                    pgen,
                    r.min(100),
                    1,
                );
                s.limit = Some(100);
                out.push(s);
            }
            // reverse order
            let flip = if w.ck[0].1 { "ASC" } else { "DESC" };
            let mut s = mk(
                "reverse_read",
                format!(
                    "SELECT * FROM {table} WHERE {pkw} ORDER BY {} {flip} LIMIT 10",
                    first.name
                ),
                key_gen(false),
                r.min(10),
                1,
            );
            s.limit = Some(10);
            out.push(s);
            // count
            let mut s = mk(
                "count_partition",
                format!("SELECT count(*) FROM {table} WHERE {pkw}"),
                key_gen(false),
                1,
                1,
            );
            s.row_bytes = 8;
            out.push(s);
            // filtering inside one partition on a regular text column (matches nothing: full partition scan)
            if let Some(tc) = w.regular.iter().find(|c| matches!(c.ty, Ty::Text(_))) {
                let wr = wa.clone();
                let pgen: Gen = Arc::new(move |rng| {
                    let mut v = wr.pk_values(rng.random_range(0..wr.partitions));
                    v.push(CqlValue::Text("no-such-value".into()));
                    v
                });
                let mut s = mk(
                    "filter_in_partition",
                    format!(
                        "SELECT * FROM {table} WHERE {pkw} AND {} = ? ALLOW FILTERING",
                        tc.name
                    ),
                    pgen,
                    0,
                    1,
                );
                s.heavy = r > 20_000;
                out.push(s);
            }
        }
        // IN on a single-column partition key
        if w.pk.len() == 1 {
            for k in [2usize, 5, 10, 25, 50] {
                let wr = wa.clone();
                let pgen: Gen = Arc::new(move |rng| {
                    (0..k)
                        .flat_map(|_| wr.pk_values(rng.random_range(0..wr.partitions)))
                        .collect()
                });
                let marks = vec!["?"; k].join(", ");
                let limit = if has_ck && r > 100 {
                    Some(1000u32)
                } else {
                    None
                };
                let mut s = mk(
                    "in_read",
                    format!(
                        "SELECT * FROM {table} WHERE {} IN ({marks}){}",
                        w.pk[0].name,
                        limit.map(|l| format!(" LIMIT {l}")).unwrap_or_default()
                    ),
                    pgen,
                    (k * r).min(limit.unwrap_or(u32::MAX) as usize),
                    k,
                );
                s.in_len = k;
                s.limit = limit;
                out.push(s);
            }
        }
        // no partition key: scans
        let mut s = mk(
            "full_scan_limit",
            format!("SELECT * FROM {table} LIMIT 100"),
            Arc::new(|_| vec![]),
            100,
            p,
        );
        s.limit = Some(100);
        out.push(s);
        let wr = wa.clone();
        let pgen: Gen = Arc::new(move |rng| wr.pk_values(rng.random_range(0..wr.partitions)));
        let tok = w.pk.iter().map(|c| c.name).collect::<Vec<_>>().join(", ");
        let tok_marks = vec!["?"; w.pk.len()].join(", ");
        let mut s = mk(
            "token_scan",
            format!("SELECT * FROM {table} WHERE token({tok}) > token({tok_marks}) LIMIT 100"),
            pgen,
            100,
            p,
        );
        s.limit = Some(100);
        out.push(s);
        if let Some(tc) = w.regular.iter().find(|c| matches!(c.ty, Ty::Text(_))) {
            let mut s = mk(
                "filter_full_table",
                format!(
                    "SELECT * FROM {table} WHERE {} = ? LIMIT 10 ALLOW FILTERING",
                    tc.name
                ),
                Arc::new(|_| vec![CqlValue::Text("no-such-value".into())]),
                0,
                p,
            );
            s.limit = Some(10);
            s.heavy = true;
            out.push(s);
        }
        // secondary index
        if w.index {
            let tag = w
                .regular
                .iter()
                .find(|c| matches!(c.ty, Ty::Tag(_)))
                .expect("tag column");
            let Ty::Tag(card) = tag.ty else {
                unreachable!()
            };
            let per_value = (w.table_rows() / card as usize).max(1);
            let pgen: Gen =
                Arc::new(move |rng| vec![CqlValue::Int(rng.random_range(0..card) as i32)]);
            let mut s = mk(
                "index_read_limit",
                format!("SELECT * FROM {table} WHERE {} = ? LIMIT 100", tag.name),
                pgen,
                per_value.min(100),
                p,
            );
            s.limit = Some(100);
            out.push(s);
            if per_value <= 3000 {
                let pgen: Gen =
                    Arc::new(move |rng| vec![CqlValue::Int(rng.random_range(0..card) as i32)]);
                out.push(mk(
                    "index_read",
                    format!("SELECT * FROM {table} WHERE {} = ?", tag.name),
                    pgen,
                    per_value,
                    p,
                ));
            }
        }
    }

    // ---- writes
    let ins_cols = names(&all_cols);
    let marks = vec!["?"; all_cols.len()].join(", ");
    let wr = wa.clone();
    let row_gen = move |rng: &mut StdRng, w: &World| -> Vec<CqlValue> {
        let (pi, ri) = (
            rng.random_range(0..w.partitions),
            rng.random_range(0..w.rpp),
        );
        let mut v = w.pk_values(pi);
        v.extend(w.ck_values(ri));
        v.extend(w.regular_values(pi, ri));
        v
    };
    let mkw = |kind: &'static str, cql: String, pgen: Gen, parts: usize| Stmt {
        kind,
        cql,
        pgen,
        write: true,
        lwt: false,
        limit: None,
        in_len: 0,
        batch_n: 0,
        partitions_touched: parts,
        row_bytes: all_bytes,
        heavy: false,
        limit1: None,
    };
    if w.is_counter() {
        let set: Vec<String> = w
            .regular
            .iter()
            .map(|c| format!("{0} = {0} + ?", c.name))
            .collect();
        let wr = wa.clone();
        let pgen: Gen = Arc::new(move |rng| {
            let mut v: Vec<CqlValue> = (0..wr.regular.len())
                .map(|_| CqlValue::Counter(scylla::value::Counter(1)))
                .collect();
            v.extend(wr.pk_values(rng.random_range(0..wr.partitions)));
            v
        });
        out.push(mkw(
            "counter_update",
            format!("UPDATE {table} SET {} WHERE {pkw}", set.join(", ")),
            pgen,
            1,
        ));
        out.push(mk(
            "counter_read",
            format!("SELECT * FROM {table} WHERE {pkw}"),
            key_gen(false),
            1,
            1,
        ));
        return out;
    }
    {
        let wr = wr.clone();
        let rg = row_gen;
        let pgen: Gen = Arc::new(move |rng| rg(rng, &wr));
        out.push(mkw(
            "insert",
            format!("INSERT INTO {table} ({ins_cols}) VALUES ({marks})"),
            pgen,
            1,
        ));
    }
    let wr2 = wa.clone();
    let pgen: Gen = Arc::new(move |rng| row_gen(rng, &wr2));
    out.push(mkw(
        "insert_ttl",
        format!("INSERT INTO {table} ({ins_cols}) VALUES ({marks}) USING TTL 86400"),
        pgen,
        1,
    ));
    let wr3 = wa.clone();
    let pgen: Gen = Arc::new(move |rng| row_gen(rng, &wr3));
    let mut s = mkw(
        "insert_lwt",
        format!("INSERT INTO {table} ({ins_cols}) VALUES ({marks}) IF NOT EXISTS"),
        pgen,
        1,
    );
    s.lwt = true;
    out.push(s);
    if let Some(rc) = w.regular.first() {
        let wr = wa.clone();
        let pgen: Gen = Arc::new(move |rng| {
            let (pi, ri) = (
                rng.random_range(0..wr.partitions),
                rng.random_range(0..wr.rpp),
            );
            let mut v = vec![world::reg_value(&wr.regular[0], pi, ri + 1)];
            v.extend(wr.pk_values(pi));
            v.extend(wr.ck_values(ri));
            v
        });
        out.push(mkw(
            "update",
            format!("UPDATE {table} SET {} = ? WHERE {full_key}", rc.name),
            pgen,
            1,
        ));
    }
    let wr4 = wa.clone();
    let pgen: Gen = Arc::new(move |rng| {
        let (pi, ri) = (
            rng.random_range(0..wr4.partitions),
            rng.random_range(0..wr4.rpp),
        );
        let mut v = wr4.pk_values(pi);
        v.extend(wr4.ck_values(ri + wr4.rpp)); // keys past the data: writes a tombstone without destroying data
        v
    });
    out.push(mkw(
        "delete",
        format!("DELETE FROM {table} WHERE {full_key}"),
        pgen,
        1,
    ));
    // batches
    for n in [5usize, 10, 25, 50, 100] {
        if has_ck && r >= n {
            let wr = wa.clone();
            let pgen: Gen = Arc::new(move |rng| {
                // one partition: n consecutive rows (flattened; the runner splits by statement)
                let pi = rng.random_range(0..wr.partitions);
                let base = rng.random_range(0..wr.rpp - n + 1);
                (0..n)
                    .flat_map(|j| {
                        let mut v = wr.pk_values(pi);
                        v.extend(wr.ck_values(base + j));
                        v.extend(wr.regular_values(pi, base + j));
                        v
                    })
                    .collect()
            });
            let mut s = mkw(
                "batch_same_partition",
                format!("INSERT INTO {table} ({ins_cols}) VALUES ({marks})"),
                pgen,
                1,
            );
            s.batch_n = n;
            out.push(s);
        }
        if p >= n {
            let wr = wa.clone();
            let pgen: Gen = Arc::new(move |rng| {
                (0..n)
                    .flat_map(|_| {
                        let (pi, ri) = (
                            rng.random_range(0..wr.partitions),
                            rng.random_range(0..wr.rpp),
                        );
                        let mut v = wr.pk_values(pi);
                        v.extend(wr.ck_values(ri));
                        v.extend(wr.regular_values(pi, ri));
                        v
                    })
                    .collect()
            });
            let mut s = mkw(
                "batch_multi_partition",
                format!("INSERT INTO {table} ({ins_cols}) VALUES ({marks})"),
                pgen,
                n,
            );
            s.batch_n = n;
            out.push(s);
        }
    }
    out
}

struct Measured {
    p50: f64,
    p90: f64,
    p99: f64,
    rows: f64,
}

fn pct(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

async fn measure(
    session: &Session,
    ks: &str,
    table: &str,
    st: &Stmt,
    reps: usize,
    rng: &mut StdRng,
) -> Result<Measured> {
    let cql = st
        .cql
        .replace(" t ", &format!(" {ks}.{table} "))
        .replace("INTO t ", &format!("INTO {ks}.{table} "))
        .replace("UPDATE t ", &format!("UPDATE {ks}.{table} "))
        .replace("FROM t", &format!("FROM {ks}.{table}"));
    let prepared = session.prepare(cql).await?;
    let reps = if st.heavy { reps.min(8) } else { reps };
    let mut lat = Vec::with_capacity(reps);
    let mut rows = 0.0;
    let n_cols_per_row = if st.batch_n > 0 {
        prepared.get_variable_col_specs().len()
    } else {
        0
    };
    for i in 0..reps + 5 {
        let params = (st.pgen)(rng);
        let t0 = Instant::now();
        let n = if st.batch_n > 0 {
            let mut batch = Batch::new(if st.kind == "batch_same_partition" {
                BatchType::Unlogged
            } else {
                BatchType::Logged
            });
            let mut vals = Vec::new();
            for chunk in params.chunks(n_cols_per_row) {
                batch.append_statement(prepared.clone());
                vals.push(chunk.to_vec());
            }
            session.batch(&batch, vals).await?;
            0
        } else {
            let res = session.execute_unpaged(&prepared, params).await?;
            res.into_rows_result().map(|r| r.rows_num()).unwrap_or(0)
        };
        let dt = t0.elapsed().as_secs_f64() * 1e6;
        if i >= 5 {
            lat.push(dt);
            rows += n as f64;
        }
    }
    lat.sort_by(f64::total_cmp);
    Ok(Measured {
        p50: pct(&lat, 0.5),
        p90: pct(&lat, 0.9),
        p99: pct(&lat, 0.99),
        rows: rows / reps as f64,
    })
}

/// Throughput of 64 concurrent tasks for `secs` with uniform partition choice or all tasks on one hot partition.
async fn throughput(
    session: Arc<Session>,
    w: &World,
    ks: &str,
    write: bool,
    zipf: bool,
    secs: f64,
) -> Result<f64> {
    let table = format!("{ks}.c");
    let all_cols: Vec<&world::Col> =
        w.pk.iter()
            .chain(w.ck.iter().map(|(c, _)| c))
            .chain(w.regular.iter())
            .collect();
    let cql = if write {
        format!(
            "INSERT INTO {table} ({}) VALUES ({})",
            names(&all_cols),
            vec!["?"; all_cols.len()].join(", ")
        )
    } else {
        let key = if w.ck.is_empty() {
            pk_where(w)
        } else {
            format!("{} AND {}", pk_where(w), ck_where(w))
        };
        format!("SELECT * FROM {table} WHERE {key}")
    };
    let prepared = session.prepare(cql).await?;
    let deadline = Instant::now() + Duration::from_secs_f64(secs);
    let mut tasks = futures::stream::FuturesUnordered::new();
    for t in 0..64u64 {
        let (session, prepared, w) = (session.clone(), prepared.clone(), w.clone());
        tasks.push(tokio::spawn(async move {
            let mut rng = StdRng::seed_from_u64(77 + t);
            let z = Zipf::new(w.partitions as f64, 0.99).unwrap();
            let mut n = 0u64;
            while Instant::now() < deadline {
                let pi = if zipf {
                    0
                } else {
                    rng.random_range(0..w.partitions)
                };
                let _ = &z;
                let ri = rng.random_range(0..w.rpp);
                let mut v = w.pk_values(pi);
                v.extend(w.ck_values(ri));
                if write {
                    v.extend(w.regular_values(pi, ri));
                }
                if session.execute_unpaged(&prepared, v).await.is_err() {
                    break;
                }
                n += 1;
            }
            n
        }));
    }
    let mut total = 0;
    while let Some(r) = tasks.next().await {
        total += r?;
    }
    Ok(total as f64 / secs)
}

async fn load(
    session: &Arc<Session>,
    w: &World,
    ks: &str,
    table: &str,
    tombstones: Tomb,
) -> Result<()> {
    let all_cols: Vec<&world::Col> =
        w.pk.iter()
            .chain(w.ck.iter().map(|(c, _)| c))
            .chain(w.regular.iter())
            .collect();
    if w.is_counter() {
        let set: Vec<String> = w
            .regular
            .iter()
            .map(|c| format!("{0} = {0} + ?", c.name))
            .collect();
        let prepared = session
            .prepare(format!(
                "UPDATE {ks}.{table} SET {} WHERE {}",
                set.join(", "),
                pk_where(w)
            ))
            .await?;
        let mut stream = futures::stream::iter(0..w.partitions)
            .map(|p| {
                let (session, prepared, w) = (session.clone(), prepared.clone(), w.clone());
                async move {
                    let mut v: Vec<CqlValue> = (0..w.regular.len())
                        .map(|_| CqlValue::Counter(scylla::value::Counter(10)))
                        .collect();
                    v.extend(w.pk_values(p));
                    session.execute_unpaged(&prepared, v).await
                }
            })
            .buffer_unordered(256);
        while let Some(r) = stream.next().await {
            r?;
        }
        return Ok(());
    }
    let prepared = session
        .prepare(format!(
            "INSERT INTO {ks}.{table} ({}) VALUES ({})",
            names(&all_cols),
            vec!["?"; all_cols.len()].join(", ")
        ))
        .await?;
    let total = w.table_rows();
    let mut stream = futures::stream::iter(0..total)
        .map(|i| {
            let (session, prepared, w) = (session.clone(), prepared.clone(), w.clone());
            async move {
                let (p, r) = (i / w.rpp, i % w.rpp);
                let mut v = w.pk_values(p);
                v.extend(w.ck_values(r));
                v.extend(w.regular_values(p, r));
                session.execute_unpaged(&prepared, v).await
            }
        })
        .buffer_unordered(256);
    while let Some(r) = stream.next().await {
        r?;
    }
    if tombstones != Tomb::None && !w.ck.is_empty() {
        let del = session
            .prepare(format!(
                "DELETE FROM {ks}.{table} WHERE {} AND {}",
                pk_where(w),
                ck_where(w)
            ))
            .await?;
        let rpp = w.rpp;
        let dead = move |i: &usize| match tombstones {
            Tomb::Alternate => i % rpp % 2 == 1,
            Tomb::Prefix => i % rpp < rpp * 95 / 100,
            Tomb::None => false,
        };
        let mut stream = futures::stream::iter(0..total)
            .filter(move |i| futures::future::ready(dead(i)))
            .map(|i| {
                let (session, del, w) = (session.clone(), del.clone(), w.clone());
                async move {
                    let (p, r) = (i / w.rpp, i % w.rpp);
                    let mut v = w.pk_values(p);
                    v.extend(w.ck_values(r));
                    session.execute_unpaged(&del, v).await
                }
            })
            .buffer_unordered(256);
        while let Some(r) = stream.next().await {
            r?;
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let addr = args
        .node
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| anyhow::anyhow!("bad node"))?;
    let session = Arc::new(
        SessionBuilder::new()
            .known_node_addr(addr)
            .address_translator(Arc::new(Pin(addr)))
            .build()
            .await?,
    );
    let mut out = std::fs::File::create(&args.out)?;
    let worlds: Vec<World> = world::all_worlds(args.scale)
        .into_iter()
        .filter(|w| args.only.as_ref().is_none_or(|o| w.id.contains(o)))
        .collect();
    if worlds.is_empty() {
        bail!("no world matches");
    }
    let mut rng = StdRng::seed_from_u64(42);
    let started = Instant::now();
    for (wi, w) in worlds.iter().enumerate() {
        let ks = format!("hpc_{}", w.id.replace(['-', '.'], "_"));
        eprintln!(
            "[{}/{}] {} ({} rows) ...",
            wi + 1,
            worlds.len(),
            w.id,
            w.table_rows()
        );
        session
            .query_unpaged(format!("DROP KEYSPACE IF EXISTS {ks}"), [])
            .await?;
        session.query_unpaged(format!("CREATE KEYSPACE {ks} WITH replication = {{'class': 'NetworkTopologyStrategy', 'replication_factor': 1}}"), []).await?;
        // main table `t` (index, tombstones) and clean twin `c` (baselines, twin comparison)
        session.query_unpaged(w.ddl(&format!("{ks}.t")), []).await?;
        session.query_unpaged(w.ddl(&format!("{ks}.c")), []).await?;
        let t0 = Instant::now();
        load(&session, w, &ks, "c", Tomb::None).await?;
        load(&session, w, &ks, "t", w.tomb).await?;
        if w.index {
            let tag = w
                .regular
                .iter()
                .find(|c| matches!(c.ty, Ty::Tag(_)))
                .expect("tag");
            session
                .query_unpaged(format!("CREATE INDEX ON {ks}.t ({})", tag.name), [])
                .await?;
            tokio::time::sleep(Duration::from_secs(6)).await;
        }
        eprintln!("    loaded in {:.1}s", t0.elapsed().as_secs_f64());

        let all = statements(w);
        let base_read = all
            .iter()
            .find(|s| s.kind == "point_read" || s.kind == "counter_read");
        let base_write = all
            .iter()
            .find(|s| s.kind == "insert" || s.kind == "counter_update");
        let br = match base_read {
            Some(s) => {
                measure(&session, &ks, "c", s, args.reps * 6, &mut rng)
                    .await?
                    .p50
            }
            None => f64::NAN,
        };
        let bw = match base_write {
            Some(s) => {
                measure(&session, &ks, "c", s, args.reps * 6, &mut rng)
                    .await?
                    .p50
            }
            None => f64::NAN,
        };
        let hot_read = if w.partitions >= 50 && !w.is_counter() {
            let u = throughput(session.clone(), w, &ks, false, false, 2.5).await?;
            let z = throughput(session.clone(), w, &ks, false, true, 2.5).await?;
            Some(z / u)
        } else {
            None
        };
        let hot_write = if w.partitions >= 50 && !w.is_counter() {
            let u = throughput(session.clone(), w, &ks, true, false, 2.5).await?;
            let z = throughput(session.clone(), w, &ks, true, true, 2.5).await?;
            Some(z / u)
        } else {
            None
        };
        for st in &all {
            let m = match measure(&session, &ks, "t", st, args.reps, &mut rng).await {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("    skip {} ({e})", st.kind);
                    continue;
                }
            };
            let twin = if (w.tomb != Tomb::None || w.index) && !st.write {
                measure(&session, &ks, "c", st, args.reps, &mut rng)
                    .await
                    .ok()
                    .map(|m| m.p50)
            } else {
                None
            };
            let limit1 = match &st.limit1 {
                Some(q) => {
                    let one = Stmt {
                        cql: q.clone(),
                        limit1: None,
                        ..st.clone()
                    };
                    measure(&session, &ks, "t", &one, args.reps, &mut rng)
                        .await
                        .ok()
                        .map(|m| m.p50)
                }
                None => None,
            };
            let hot = match st.kind {
                "point_read" => hot_read,
                "insert" => hot_write,
                _ => None,
            };
            let sample = Sample {
                world: w.id.clone(),
                family: w.family,
                engine: args.engine.clone(),
                ddl: w.ddl("t"),
                statement: st.cql.clone(),
                kind: st.kind,
                partitions: w.partitions,
                rows_per_partition: w.rpp,
                row_bytes: w.row_bytes(),
                table_rows: w.table_rows(),
                tombstones: w.tomb != Tomb::None,
                tomb_mode: match w.tomb {
                    Tomb::None => "none",
                    Tomb::Alternate => "alternate",
                    Tomb::Prefix => "prefix",
                },
                indexed: w.index,
                write: st.write,
                lwt: st.lwt,
                limit: st.limit,
                in_len: st.in_len,
                batch_n: st.batch_n,
                partitions_touched: st.partitions_touched,
                p50_us: m.p50,
                p90_us: m.p90,
                p99_us: m.p99,
                rows_returned: m.rows,
                bytes_est: m.rows * st.row_bytes as f64,
                base_read_us: br,
                base_write_us: bw,
                twin_p50_us: twin,
                limit1_p50_us: limit1,
                hot_ratio: hot,
            };
            writeln!(out, "{}", serde_json::to_string(&sample)?)?;
        }
        out.flush()?;
        session
            .query_unpaged(format!("DROP KEYSPACE {ks}"), [])
            .await?;
        eprintln!("    done, {:.0}s elapsed", started.elapsed().as_secs_f64());
    }
    Ok(())
}
