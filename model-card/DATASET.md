---
license: apache-2.0
task_categories:
  - tabular-regression
  - text-classification
tags:
  - cql
  - cassandra
  - scylladb
  - performance
  - benchmark
---

# cql-hpc-bench

Measured performance of CQL statements on tables shaped like public benchmark workloads. One JSON object per statement.

* `cql-hpc-scylla.jsonl`: ScyllaDB (latest), 4 shards, 90 worlds, ~2,670 statements.
* `cql-hpc-cassandra.jsonl`: Apache Cassandra 5.0, 90 worlds (half partition counts), ~2,614 statements.
* Workload shapes: NoSQLBench (`cql-keyvalue2`, `cql-tabular2`, `cql-iot`), cassandra-stress (blog posts, counters), YCSB (`usertable`), TSBS (`cpu`), KillrVideo (comments), plus tombstone-heavy and indexed variants.
* Fields: `world`, `family`, `engine`, `ddl`, `statement`, `kind`, data shape (`partitions`, `rows_per_partition`, `row_bytes`, `table_rows`, `tombstones`, `tomb_mode`, `indexed`),
  statement facts (`write`, `lwt`, `limit`, `in_len`, `batch_n`, `partitions_touched`), measurements (`p50_us`, `p90_us`, `p99_us`, `rows_returned`, `bytes_est`),
  baselines on the same table (`base_read_us`, `base_write_us`), counterfactuals (`twin_p50_us` on a clean twin table, `limit1_p50_us` with `LIMIT 1`) and `hot_ratio`.
* Collected with `cql-hpc-collect` (https://github.com/Max-imalgutaussehend/cql-ai) on one Apple M3 in Docker, 40 repetitions per statement.

Limitations: single node, laptop VM, noise of Docker on macOS; absolute microseconds are not portable, ratios are the intended signal.
Workload definitions the shapes are based on are Apache-2.0 licensed (NoSQLBench, Apache Cassandra, KillrVideo).
