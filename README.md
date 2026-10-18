# cql-ai

**A tiny Transformer that predicts CQL performance bottlenecks in CI, trained on measurements from public high-performance CQL workloads.**

> Part of the **cqlz family**
>
> | Project | Direction | What it does |
> |---|---|---|
> | [`cqlc`](https://github.com/Max-imalgutaussehend/cqlc) | schema-first | CQL → typed Rust, checks Cassandra 3.x–5.0 CQL offline |
> | [`cqlx`](https://github.com/Max-imalgutaussehend/cqlx) | code-first | Rust structs → schema, writes and type-state queries |
> | **`cql-ai`** | **performance** | Predicts bottlenecks from the statement and the data shape (this repo) |

People who use CQL from Rust want more than type safety: they want it fast at scale. `cql-ai` reads your `.cql`
files in CI and warns when a statement is likely to be a *performance* problem on your data:

```text
$ cql-ai check queries/ --schema schema/
warning: AI predicts a large result bottleneck (100% confidence, about 40x the cost of a point read) on line 1
 --> queries/events.cql:1:1
  |
1 | SELECT * FROM events WHERE part = ?
  |
  = help: cost is dominated by the number of rows/bytes returned (materialisation, serialisation, network): add a LIMIT, page, or project fewer columns
  = note: data shape: 200 partitions x 10k rows per partition
```

## What it predicts

Six bottleneck classes plus a cost estimate (`cost` = how many times a point read, or a plain insert for writes).
Labels are **measured outcomes**, not rules: every statement of the training set was executed on a real database.

| Label | Operational definition (relative to a baseline measured on the same table) |
|---|---|
| `LargeResult` | read ≥ 3× a point read, returns ≥ 100 rows or ≥ 32 KiB, and the same statement with `LIMIT 1` is ≥ 2.5× faster |
| `ServerScan` | read ≥ 3× a point read but returns < 100 rows / < 32 KiB (the server scans much more than it returns: `count`, filtering) |
| `FanOut` | read ≥ 3× a point read and touches more than one partition (`IN`, token scans, secondary indexes, filtering) |
| `TombstoneScan` | ≥ 2× slower than the same statement on a clean twin table (queue pattern: the first 95 % of every partition deleted) |
| `WriteCost` | non-LWT write ≥ 1.8× a plain insert (in the data: batches, mostly across partitions) |
| `LightweightTxn` | `IF NOT EXISTS` and friends, ≥ 1.5× a plain insert |

Everything is a ratio against a baseline measured on the same machine and table, so labels do not depend on the hardware used for collection.

## The data: measured, from public workloads

`cql-hpc-collect` builds real tables shaped like the public benchmarks below, loads data, and measures every statement
(40 repetitions, p50/p90/p99, rows returned, counterfactuals such as `LIMIT 1` and a clean twin table).

| Workload shape | Source |
|---|---|
| key-value, tabular, time-series (IoT) | [NoSQLBench](https://github.com/nosqlbench/nosqlbench) baselines `cql-keyvalue2`, `cql-tabular2`, `cql-iot` (Apache-2.0, copied in `data/public/`) |
| blog posts, counters | [cassandra-stress](https://github.com/apache/cassandra) profiles (Apache-2.0, copied in `data/public/`) |
| user table | [YCSB](https://github.com/brianfrankcooper/YCSB) core workloads |
| `cpu` time series | [TSBS](https://github.com/timescale/tsbs) layout |
| comments | [KillrVideo](https://github.com/KillrVideo/killrvideo-data) model (Apache-2.0) |

90 worlds per engine (partitions from 10 to 400,000, 1 to 60,000 rows per partition, tombstone-heavy variants, indexed variants), about 2,650
measured statements each, on **ScyllaDB (latest, 4 shards)** and **Apache Cassandra 5.0**, both in Docker on one Apple M3.
The dataset is published as [`Max-imumlikelihood/cql-hpc-bench`](https://huggingface.co/datasets/Max-imumlikelihood/cql-hpc-bench).

## Model

`cqlc-core`'s lexer turns a statement into integer ids (no text tokenizer; identifiers become schema-role tokens). A numeric context vector adds
what tokens cannot carry: data shape (partitions, rows per partition, row bytes, tombstones, index) and literals (`LIMIT`, `IN` list length, batch size).
4-layer Transformer encoder, d=128, 578k parameters, a bottleneck head (6 sigmoids) and a cost head (`log10` of the cost ratio).
Built with [Burn](https://burn.dev); weights are plain safetensors.

## Results

Trained on ScyllaDB measurements only (1,597 statements), evaluated three ways that the model never saw during training.
Macro-F1 over the labels present; cost error in `log10` units (0.135 means a typical error of ×1.37).

| Test set | Statements | Model + data stats | Model, no stats | Keyword-rule baseline | Cost MAE | Spearman |
|---|---|---|---|---|---|---|
| unseen **worlds** (same workloads, other data shapes) | 631 | **0.855** | 0.683 | 0.557 | 0.135 | 0.920 |
| unseen **families** (YCSB, KillrVideo comments) | 756 | **0.889** | 0.674 | 0.624 | 0.136 | 0.891 |
| other **engine** (trained on ScyllaDB, tested on Cassandra 5.0) | 2,614 | **0.802** | 0.624 | 0.516 | 0.179 | 0.848 |

What this says: the model clearly beats a keyword heuristic, and **data shape is what makes the difference**: the same query is a problem on a 10,000-row partition
and fine on a 10-row one. Without data statistics the model falls back to roughly rule-level accuracy. The weak spots are `WriteCost` (precision 0.5–0.6; the baselines are noisy) and
`TombstoneScan` on the other engine (precision 0.48).

## Use it

```sh
cargo install cql-ai-cli
cql-ai check queries/ --schema schema/ [--stats stats.toml] [--threshold 0.5] [--min-cost 20] [--deny-warnings] [--format json]
```

Tell it about your data in the schema (it only needs orders of magnitude):

```sql
-- hpc: partitions=200000 rows=10000 row_bytes=300 tombstones=false
CREATE TABLE events (...);
```

or in a TOML file (`--stats`):

```toml
[events]
partitions = 200000
rows_per_partition = 10000
row_bytes = 300
```

`-- hpc: batch=25` above a statement tells the model it is written in a batch of 25. Without statistics the model only sees the statement and says so.

The model (2.3 MB) is fetched from Hugging Face once into `~/.cache/cql-ai/` (`CQL_AI_CACHE`; `--model <dir>` for local files; `--offline`; `CQL_AI_HF_REPO` to override).
A check of a 4-statement file takes about 0.2 s wall clock including process start.

GitHub Actions: copy [`templates/cql-ai.yml`](templates/cql-ai.yml).

## Reproduce

```sh
docker run -d -p 9042:9042 scylladb/scylla --smp 4 --memory 6G --overprovisioned 1 --developer-mode 1
cargo run --release -p cql-hpc-collect -- --node 127.0.0.1:9042 --engine scylla --out data/cql-hpc-scylla.jsonl     # ~10 min
cargo run --release -p cql-ai-train -- stats --data data/cql-hpc-scylla.jsonl
cargo run --release -p cql-ai-train -- train --data data/cql-hpc-scylla.jsonl --data data/cql-hpc-cassandra.jsonl --train-engine scylla
```

## Honest limitations

* **One machine, one node, Docker.** All measurements come from a single node on a laptop VM. Cluster effects (replication, network, multi-DC, compaction under load) are not in the data.
  Ratios are relative, but a different cluster can rank statements differently.
* **No `HotKey` class.** A first version measured throughput with all clients on one partition. On a single ScyllaDB node it did **not** degrade (ratio ≥ 0.9, caching helps), so there is no signal to learn and the label was dropped.
  Real hot-partition problems need a cluster and write-heavy load.
* **Tombstones** are only the queue pattern (first 95 % of a partition deleted); alternating deletes on ScyllaDB made reads *faster* and are not labelled.
* **Thresholds are choices** (3×, 100 rows, ...), written down in `crates/cql-ai-train/src/labels.rs`. Measurement noise on Docker for Mac is real (single-digit to tens of percent).
* The model predicts *relative* cost for the statement shapes in the data (reads, writes, counters, LWT, batches, scans, `IN`, secondary indexes). Vector search, UDTs, `GROUP BY` and others are out of distribution.
* It does not replace `cqlc check`, which is exact and about 400× faster for the anti-patterns it knows; `cql-ai` answers a different question: *how slow will this be on my data?*

## Publishing

```sh
python3 scripts/push_to_hf.py                      # model -> Max-imumlikelihood/cql-hpc-bottleneck-transformer
python3 scripts/push_to_hf.py --dataset            # data/*.jsonl -> Max-imumlikelihood/cql-hpc-bench
```

## License

Apache-2.0
