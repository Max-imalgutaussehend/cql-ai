---
license: apache-2.0
library_name: burn
tags:
  - cql
  - cassandra
  - scylladb
  - performance
  - safetensors
datasets:
  - Max-imumlikelihood/cql-hpc-bench
---

# cql-hpc-bottleneck-transformer

Tiny (578k parameters) Transformer that predicts **CQL performance bottlenecks** from `cqlc-core` lexer tokens plus the shape of the data
(partitions, rows per partition, row bytes, tombstones, index) and statement literals (`LIMIT`, `IN` length, batch size).
Part of the [cqlz family](https://github.com/Max-imalgutaussehend/cql-ai).

* Output: six bottleneck probabilities (`LargeResult`, `ServerScan`, `FanOut`, `TombstoneScan`, `WriteCost`, `LightweightTxn`) and `log10(cost / baseline)`.
* Trained on **measured** latencies of public workload shapes (NoSQLBench, cassandra-stress, YCSB, TSBS, KillrVideo) on ScyllaDB and Cassandra 5.0:
  [`cql-hpc-bench`](https://huggingface.co/datasets/Max-imumlikelihood/cql-hpc-bench).
* Files: `model.safetensors`, `config.json` (Burn `ModelConfig`). Use with `cargo install cql-ai-cli && cql-ai check queries/ --schema schema/`.

## Evaluation (model trained on ScyllaDB only; Macro-F1)

| Test set | with data stats | without | keyword baseline |
|---|---|---|---|
| unseen worlds | 0.855 | 0.683 | 0.557 |
| unseen workload families | 0.889 | 0.674 | 0.624 |
| other engine (Cassandra 5.0) | 0.802 | 0.624 | 0.516 |

The released weights are trained on both engines pooled (same recipe). See the repository README for definitions, thresholds and limitations
(single node in Docker on one machine; no hot-key label because no signal was observable there).
