---
license: apache-2.0
library_name: burn
tags:
  - cql
  - cassandra
  - scylladb
  - static-analysis
  - safetensors
pipeline_tag: text-classification
---

# cql-anti-pattern-transformer

Tiny (576k parameters) Transformer encoder that predicts CQL anti-patterns from **`cqlc-core` lexer
tokens** (no text tokenizer). Part of the [cqlz family](https://github.com/Max-imalgutaussehend/cql-ai).

* Input: a CQL statement lexed by `cqlc-core`, identifiers replaced by schema role tokens.
* Output: 6 independent probabilities — `MissingPartitionKey`, `InefficientAllowFiltering`,
  `NonKeyRestriction`, `ClusteringGap`, `OrderByMismatch`, `LargeCollectionUpdate`.
* Files: `model.safetensors` (weights, Burn module paths), `config.json` (Burn `ModelConfig`).
* Use with the `cql-ai` CLI: `cargo install cql-ai-cli && cql-ai check queries/ --schema schema/`.

## Training data

100,000 synthetic statements over 600 random schemas (6 epochs, AdamW), labelled by `cqlc-core`'s analyzer
(plus a generator rule for `LargeCollectionUpdate`). See the repository for the generator.

## Evaluation

10,000 fresh statements from unseen schemas, threshold 0.5:

| Label | Precision | Recall | F1 |
|---|---|---|---|
| MissingPartitionKey | 0.879 | 0.891 | 0.885 |
| InefficientAllowFiltering | 1.000 | 1.000 | 1.000 |
| NonKeyRestriction | 1.000 | 1.000 | 1.000 |
| ClusteringGap | 1.000 | 1.000 | 1.000 |
| OrderByMismatch | 0.997 | 0.954 | 0.975 |
| LargeCollectionUpdate | 1.000 | 0.998 | 0.999 |

Macro-F1 0.977, exact match 0.949. Synthetic distribution only.

## Limitations

Synthetic training data; needs the table schema to assign key roles; not a substitute for
`cqlc`'s deterministic analysis.
