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

100,000 synthetic statements over 600 random schemas, labelled by `cqlc-core`'s analyzer
(plus a generator rule for `LargeCollectionUpdate`). See the repository for the generator.

## Evaluation

CARD_RESULTS_PLACEHOLDER

## Limitations

Synthetic training data; needs the table schema to assign key roles; not a substitute for
`cqlc`'s deterministic analysis.
