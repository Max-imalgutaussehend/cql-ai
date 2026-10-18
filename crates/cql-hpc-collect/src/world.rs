//! World definitions: public CQL workload shapes (NoSQLBench, cassandra-stress, YCSB, TSBS, KillrVideo) as
//! parametrised tables with deterministic data.

use scylla::value::{CqlTimestamp, CqlTimeuuid, CqlValue};
use uuid::Uuid;

#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(dead_code)]
pub enum Ty {
    /// `text` of the given byte width
    Text(usize),
    Int,
    BigInt,
    Double,
    Timestamp,
    Uuid,
    TimeUuid,
    Counter,
    /// `int` with the given cardinality (an index target)
    Tag(u32),
}

impl Ty {
    pub fn cql(self) -> &'static str {
        match self {
            Ty::Text(_) => "text",
            Ty::Int | Ty::Tag(_) => "int",
            Ty::BigInt => "bigint",
            Ty::Double => "double",
            Ty::Timestamp => "timestamp",
            Ty::Uuid => "uuid",
            Ty::TimeUuid => "timeuuid",
            Ty::Counter => "counter",
        }
    }

    pub fn bytes(self) -> usize {
        match self {
            Ty::Text(w) => w,
            Ty::Int | Ty::Tag(_) => 4,
            Ty::BigInt | Ty::Double | Ty::Timestamp | Ty::Counter => 8,
            Ty::Uuid | Ty::TimeUuid => 16,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Col {
    pub name: &'static str,
    pub ty: Ty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tomb {
    None,
    Alternate,
    Prefix,
}

#[derive(Clone, Debug)]
pub struct World {
    pub id: String,
    pub family: &'static str,
    pub pk: Vec<Col>,
    pub ck: Vec<(Col, bool)>,
    pub regular: Vec<Col>,
    pub partitions: usize,
    pub rpp: usize,
    /// How the main table is made tombstone-heavy: nothing, every second row, or the first 95% of every partition (queue pattern).
    pub tomb: Tomb,
    /// Index the first `Tag` column of the main table.
    pub index: bool,
}

fn c(name: &'static str, ty: Ty) -> Col {
    Col { name, ty }
}

const DATA: [&str; 8] = [
    "data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7",
];
const FIELD: [&str; 10] = [
    "field0", "field1", "field2", "field3", "field4", "field5", "field6", "field7", "field8",
    "field9",
];
const USAGE: [&str; 10] = [
    "usage_user",
    "usage_system",
    "usage_idle",
    "usage_nice",
    "usage_iowait",
    "usage_irq",
    "usage_softirq",
    "usage_steal",
    "usage_guest",
    "usage_guest_nice",
];

fn world(
    id: String,
    family: &'static str,
    pk: Vec<Col>,
    ck: Vec<(Col, bool)>,
    regular: Vec<Col>,
    partitions: usize,
    rpp: usize,
) -> World {
    World {
        id,
        family,
        pk,
        ck,
        regular,
        partitions,
        rpp,
        tomb: Tomb::None,
        index: false,
    }
}

/// All worlds of the benchmark. `scale` shrinks partition counts for quick runs.
pub fn all_worlds(scale: f64) -> Vec<World> {
    let s = |n: usize| ((n as f64 * scale) as usize).max(10);
    let mut out = Vec::new();
    // NoSQLBench cql-keyvalue2: key -> value
    for (p, w) in [(s(20_000), 64), (s(200_000), 64), (s(50_000), 1024)] {
        out.push(world(
            format!("nb_keyvalue_p{p}_w{w}"),
            "nb_keyvalue",
            vec![c("key", Ty::Text(12))],
            vec![],
            vec![c("value", Ty::Text(w))],
            p,
            1,
        ));
    }
    // NoSQLBench cql-tabular2: part, clust, data0..7
    for (p, r, w) in [
        (s(2000), 10, 16),
        (s(1000), 100, 16),
        (s(200), 1000, 16),
        (s(40), 10_000, 16),
        (s(10), 40_000, 16),
        (s(500), 200, 128),
    ] {
        let regular = DATA.iter().map(|n| c(n, Ty::Text(w))).collect();
        out.push(world(
            format!("nb_tabular_p{p}_r{r}_w{w}"),
            "nb_tabular",
            vec![c("part", Ty::Text(10))],
            vec![(c("clust", Ty::Text(12)), false)],
            regular,
            p,
            r,
        ));
    }
    // NoSQLBench cql-iot / cql-timeseries2: ((machine_id, sensor_name), time DESC)
    for (p, r) in [
        (s(500), 200),
        (s(200), 1000),
        (s(50), 8000),
        (s(10), 40_000),
    ] {
        out.push(world(
            format!("nb_iot_p{p}_r{r}"),
            "nb_iot",
            vec![c("machine_id", Ty::Uuid), c("sensor_name", Ty::Text(6))],
            vec![(c("time", Ty::Timestamp), true)],
            vec![
                c("sensor_value", Ty::Double),
                c("station_id", Ty::Uuid),
                c("data", Ty::Text(100)),
            ],
            p,
            r,
        ));
    }
    // cassandra-stress "blogposts"-style: domain, published_date, url
    for (p, r) in [(s(1000), 50), (s(100), 1000), (s(20), 5000)] {
        out.push(world(
            format!("cs_blogposts_p{p}_r{r}"),
            "cs_blogposts",
            vec![c("domain", Ty::Text(14))],
            vec![(c("published_date", Ty::TimeUuid), true)],
            vec![
                c("url", Ty::Text(40)),
                c("author", Ty::Text(16)),
                c("title", Ty::Text(60)),
                c("body", Ty::Text(1024)),
            ],
            p,
            r,
        ));
    }
    // YCSB usertable
    for (p, w) in [(s(50_000), 100), (s(200_000), 100)] {
        let regular = FIELD.iter().map(|n| c(n, Ty::Text(w))).collect();
        out.push(world(
            format!("ycsb_usertable_p{p}"),
            "ycsb",
            vec![c("y_id", Ty::Text(14))],
            vec![],
            regular,
            p,
            1,
        ));
    }
    // TSBS cpu: series_id, timestamp_ns, ten usage fields
    for (p, r) in [(s(500), 500), (s(100), 3000)] {
        let regular = USAGE.iter().map(|n| c(n, Ty::Double)).collect();
        out.push(world(
            format!("tsbs_cpu_p{p}_r{r}"),
            "tsbs_cpu",
            vec![c("series_id", Ty::Text(16))],
            vec![(c("timestamp_ns", Ty::BigInt), false)],
            regular,
            p,
            r,
        ));
    }
    // KillrVideo comments: ((videoid), commentid DESC)
    for (p, r) in [(s(5000), 20), (s(200), 1000), (s(10), 30_000)] {
        out.push(world(
            format!("kv_comments_p{p}_r{r}"),
            "kv_comments",
            vec![c("videoid", Ty::Uuid)],
            vec![(c("commentid", Ty::TimeUuid), true)],
            vec![c("userid", Ty::Uuid), c("comment", Ty::Text(200))],
            p,
            r,
        ));
    }
    // Tombstone-heavy twins of tabular and iot
    for (p, r) in [(s(200), 1000), (s(40), 10_000)] {
        let regular = DATA.iter().map(|n| c(n, Ty::Text(16))).collect();
        let mut w = world(
            format!("nb_tabular_tomb_p{p}_r{r}"),
            "nb_tabular_tomb",
            vec![c("part", Ty::Text(10))],
            vec![(c("clust", Ty::Text(12)), false)],
            regular,
            p,
            r,
        );
        w.tomb = Tomb::Alternate;
        out.push(w);
    }
    // Queue-pattern tombstones: the first 95% of every partition is deleted, readers start at the dead rows
    for (p, r) in [(s(100), 2000), (s(40), 20_000), (s(10), 60_000)] {
        let regular = DATA.iter().take(4).map(|n| c(n, Ty::Text(16))).collect();
        let mut w = world(
            format!("nb_queue_tomb_p{p}_r{r}"),
            "nb_queue_tomb",
            vec![c("part", Ty::Text(10))],
            vec![(c("clust", Ty::Text(12)), false)],
            regular,
            p,
            r,
        );
        w.tomb = Tomb::Prefix;
        out.push(w);
    }
    // Indexed tables (write amplification, index reads)
    for (p, r, card) in [(s(2000), 50, 100u32), (s(500), 200, 20_000)] {
        let mut regular: Vec<Col> = vec![c("tag", Ty::Tag(card))];
        regular.extend(DATA.iter().take(4).map(|n| c(n, Ty::Text(32))));
        let mut w = world(
            format!("nb_tabular_idx_p{p}_r{r}_card{card}"),
            "nb_tabular_idx",
            vec![c("part", Ty::Text(10))],
            vec![(c("clust", Ty::Text(12)), false)],
            regular,
            p,
            r,
        );
        w.index = true;
        out.push(w);
    }
    // cassandra-stress counters
    for p in [s(20_000), s(200_000)] {
        out.push(world(
            format!("cs_counter_p{p}"),
            "cs_counter",
            vec![c("key", Ty::Text(12))],
            vec![],
            vec![
                c("c1", Ty::Counter),
                c("c2", Ty::Counter),
                c("c3", Ty::Counter),
            ],
            p,
            1,
        ));
    }
    // each shape at half, normal and double partition count: more data shapes for the same workload
    let mut expanded = Vec::new();
    for w in out {
        for m in [0.5f64, 1.0, 2.0] {
            let mut v = w.clone();
            v.partitions = ((w.partitions as f64 * m) as usize).max(10);
            if m != 1.0 {
                v.id = v.id.replacen(
                    &format!("_p{}", w.partitions),
                    &format!("_p{}", v.partitions),
                    1,
                );
            }
            if v.table_rows() <= 900_000 && (m == 1.0 || v.partitions != w.partitions) {
                expanded.push(v);
            }
        }
    }
    expanded
}

impl World {
    pub fn table_rows(&self) -> usize {
        self.partitions * self.rpp
    }

    pub fn row_bytes(&self) -> usize {
        self.pk.iter().map(|c| c.ty.bytes()).sum::<usize>()
            + self.ck.iter().map(|(c, _)| c.ty.bytes()).sum::<usize>()
            + self.regular.iter().map(|c| c.ty.bytes()).sum::<usize>()
    }

    pub fn is_counter(&self) -> bool {
        self.regular.iter().any(|c| c.ty == Ty::Counter)
    }

    pub fn ddl(&self, table: &str) -> String {
        let mut cols: Vec<String> = Vec::new();
        for col in self
            .pk
            .iter()
            .chain(self.ck.iter().map(|(c, _)| c))
            .chain(self.regular.iter())
        {
            cols.push(format!("{} {}", col.name, col.ty.cql()));
        }
        let pk = self
            .pk
            .iter()
            .map(|c| c.name)
            .collect::<Vec<_>>()
            .join(", ");
        let mut key = if self.pk.len() > 1 {
            format!("({pk})")
        } else {
            pk
        };
        for (c, _) in &self.ck {
            key.push_str(&format!(", {}", c.name));
        }
        let mut s = format!(
            "CREATE TABLE {table} ({}, PRIMARY KEY ({key}))",
            cols.join(", ")
        );
        if !self.ck.is_empty() {
            let order: Vec<String> = self
                .ck
                .iter()
                .map(|(c, desc)| format!("{} {}", c.name, if *desc { "DESC" } else { "ASC" }))
                .collect();
            s.push_str(&format!(" WITH CLUSTERING ORDER BY ({})", order.join(", ")));
        }
        s.push(';');
        s
    }

    pub fn pk_values(&self, p: usize) -> Vec<CqlValue> {
        self.pk.iter().map(|c| pk_value(c, p)).collect()
    }

    pub fn ck_values(&self, r: usize) -> Vec<CqlValue> {
        self.ck.iter().map(|(c, _)| ck_value(c, r)).collect()
    }

    pub fn regular_values(&self, p: usize, r: usize) -> Vec<CqlValue> {
        self.regular.iter().map(|c| reg_value(c, p, r)).collect()
    }
}

fn timeuuid(r: usize) -> Uuid {
    let t = r as u64 + 1;
    Uuid::from_fields(
        (t & 0xffff_ffff) as u32,
        ((t >> 32) & 0xffff) as u16,
        (((t >> 48) & 0x0fff) as u16) | 0x1000,
        &[0x80, 0x01, 1, 2, 3, 4, 5, 6],
    )
}

pub fn pk_value(c: &Col, p: usize) -> CqlValue {
    match c.ty {
        Ty::Text(_) if c.name == "sensor_name" => CqlValue::Text(format!("s{}", p % 5)),
        Ty::Text(_) => CqlValue::Text(format!("{}{p:08}", &c.name[..1])),
        Ty::Uuid => CqlValue::Uuid(Uuid::from_u128(
            0x1000_0000_0000_0000_0000_0000_0000_0000 + p as u128,
        )),
        Ty::BigInt => CqlValue::BigInt(p as i64),
        _ => CqlValue::Int(p as i32),
    }
}

pub fn ck_value(c: &Col, r: usize) -> CqlValue {
    match c.ty {
        Ty::Text(_) => CqlValue::Text(format!("c{r:010}")),
        Ty::BigInt => CqlValue::BigInt(1_700_000_000_000_000_000 + r as i64 * 1_000_000_000),
        Ty::Timestamp => CqlValue::Timestamp(CqlTimestamp(1_700_000_000_000 + r as i64 * 1000)),
        Ty::TimeUuid => CqlValue::Timeuuid(CqlTimeuuid::from(timeuuid(r))),
        Ty::Int | Ty::Tag(_) => CqlValue::Int(r as i32),
        _ => CqlValue::Int(r as i32),
    }
}

pub fn reg_value(c: &Col, p: usize, r: usize) -> CqlValue {
    match c.ty {
        Ty::Text(w) => {
            let mut s = format!("{}-{p}-{r}-", c.name);
            while s.len() < w {
                s.push_str("abcdefghij");
            }
            s.truncate(w.max(1));
            CqlValue::Text(s)
        }
        Ty::Int => CqlValue::Int(((p * 131 + r) % 1000) as i32),
        Ty::Tag(card) => CqlValue::Int(((p * 7 + r) % card as usize) as i32),
        Ty::BigInt => CqlValue::BigInt((p * 1000 + r) as i64),
        Ty::Double => CqlValue::Double(((p + r) % 1000) as f64 * 0.5),
        Ty::Uuid => CqlValue::Uuid(Uuid::from_u128(
            0x2000_0000_0000_0000_0000_0000_0000_0000 + (p as u128 % 97),
        )),
        Ty::Timestamp => CqlValue::Timestamp(CqlTimestamp(1_700_000_000_000 + (p + r) as i64)),
        Ty::TimeUuid => CqlValue::Timeuuid(CqlTimeuuid::from(timeuuid(r))),
        Ty::Counter => CqlValue::Counter(scylla::value::Counter(1)),
    }
}
