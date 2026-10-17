# cql-ai

**A from-scratch, ~0.6M-parameter Transformer that flags CQL anti-patterns in CI.**

> Part of the **cqlz family**
>
> | Project | Direction | What it does |
> |---|---|---|
> | [`cqlc`](https://github.com/Max-imalgutaussehend/cqlc) | schema-first | Static CQL → typed Rust code generator (like sqlc) |
> | [`cqlx`](https://github.com/Max-imalgutaussehend/cqlx) | code-first | Rust structs → schema + type-state query builders |
> | **`cql-ai`** | analysis | Neural anti-pattern detector, built on `cqlc-core`'s lexer (this repo) |

```text
$ cql-ai check queries/ --schema schema/
warning: AI detects 97% probability of Missing Partition Key on line 4
  --> queries/events.cql:4:1
   |
 4 | SELECT body FROM events WHERE tenant = ?
   |
   = help: partition key not fully restricted: the query can fan out across partitions (hot or scatter-gather reads)

cql-ai: checked 12 statement(s) in 3 file(s), 1 warning(s)
```

## The trick: no text tokenizer

Instead of BPE over characters, queries go through the **`cqlc-core` lexer**. Every token kind has a
fixed integer id; keywords have their own ids; identifiers are either

* a **role token** (`COL_PK`, `COL_CK0..3`, `COL_REG`, `COL_STATIC`, `COL_LIST/SET/MAP`, `COL_COUNTER`, `COL_UNKNOWN`)
  when the table's schema says what the column is, or
* one of 64 hash buckets otherwise.

So `SELECT body FROM events WHERE tenant = ? AND ts > ?` becomes about 15 ids from a vocabulary of
~240, and the model sees *structure and key roles*, not strings. Sequences are 5–10× shorter than
character/BPE tokens and the vocabulary is tiny, so the model can be tiny too.

```text
[CLS] SELECT COL_REG FROM TABLE WHERE COL_PK = BIND AND COL_CK0 > BIND
```

Consequence you should know about: **the model needs the schema** (`CREATE TABLE`s) to know which
columns are keys. `cql-ai check` takes DDL from `--schema` or from `CREATE` statements inside the
checked files, and refuses to run without any.

## Model

`cql-ai-model` (built with [Burn](https://burn.dev) 0.22): token + learned position embeddings →
**4 × Transformer encoder layers** (d_model 128, 4 heads, FFN 256, pre-norm, padding mask) →
LayerNorm → masked mean-pool → linear head with **6 sigmoid outputs**. 576k parameters, 2.3 MB.

The head is *multi-label* (a query can have several problems), not softmax multi-class:

| Label | Meaning | Ground truth |
|---|---|---|
| `MissingPartitionKey` | partition key not fully restricted | `cqlc` `CQL111` / `CQL201` |
| `InefficientAllowFiltering` | `ALLOW FILTERING` | `cqlc` `CQL202` |
| `NonKeyRestriction` | restriction on a non-key column | `cqlc` `CQL203` |
| `ClusteringGap` | clustering restriction skips a column | `cqlc` `CQL204` |
| `OrderByMismatch` | `ORDER BY` against the clustering order | `cqlc` `CQL205` |
| `LargeCollectionUpdate` | `UPDATE` overwrites a whole collection | rule in the generator (not a `cqlc` rule) |

## Training data

`cql-ai-train` contains a randomised **AST → CQL text** generator: random schemas (2–5 tables,
simple/composite partition keys, 0–3 clustering columns, scalar and collection columns), then
SELECT / INSERT / UPDATE / DELETE with random omissions and mistakes. Each statement is compiled by
`cqlc-core` against its schema and **labelled by the analyzer's diagnostics**; statements the
analyzer rejects for unrelated reasons are discarded. Validation uses schemas the model never saw.

```sh
cargo run --release -p cql-ai-train -- generate --count 100000 --out data.jsonl   # inspect the dataset
cargo run --release -p cql-ai-train --features wgpu -- train \
    --train-count 100000 --val-count 5000 --epochs 10 --batch-size 256 --lr 2e-3 --out artifacts
cargo run --release -p cql-ai-train -- eval --model artifacts                     # fresh unseen schemas
```

Training uses AdamW with warm-up and cosine decay on BCE-with-logits. `--features wgpu` trains through
Burn's wgpu backend (Metal/Vulkan/DX12); without it everything runs on the pure-Rust CPU backend.

RESULTS_PLACEHOLDER

## Use it

```sh
cargo install cql-ai-cli          # installs `cql-ai`
cql-ai check queries/ --schema schema/ [--threshold 0.5] [--deny-warnings] [--format json]
```

On first run the CLI downloads `model.safetensors` + `config.json` from
[`Max-imalgutaussehend/cql-anti-pattern-transformer`](https://huggingface.co/Max-imalgutaussehend/cql-anti-pattern-transformer)
into `~/.cache/cql-ai/` (override with `CQL_AI_CACHE`; `--model <dir>` uses local files; `--offline`
never touches the network; `HF_TOKEN` is sent if set). It then lexes every statement, runs the model
on CPU and prints compiler-style diagnostics. Exit codes: `0` ok, `1` warnings with `--deny-warnings`, `2` error.

### GitHub Actions

Copy [`templates/cql-ai.yml`](templates/cql-ai.yml) to `.github/workflows/cql-ai.yml`. It caches the
model, installs the release binary and runs `cql-ai check queries/ --schema schema/ --deny-warnings`.
With a warm cache the check itself takes TIMING_PLACEHOLDER.

## Publishing (maintainer notes)

```sh
# 1. weights to Hugging Face (needs a write token for Max-imalgutaussehend)
export HF_TOKEN=hf_...
python3 scripts/push_to_hf.py --dry-run      # inspect
python3 scripts/push_to_hf.py                # or: scripts/push_to_hf.sh (curl + jq)

# 2. crates, in dependency order
cargo publish -p cql-ai-model
cargo publish -p cql-ai-train
cargo publish -p cql-ai-cli

# 3. binary for the CI template
git tag v0.1.0 && git push --tags            # release.yml attaches cql-ai-<target>.tar.gz
```

## Honest limitations

* The model learns what `cqlc`'s analyzer already decides deterministically, plus one extra rule.
  It is a fast, fuzzy, schema-aware linter that tolerates partial queries and gives probabilities;
  it is not a replacement for `cqlc` and it does not know your data distribution (real *hot
  partition* detection needs cardinality statistics that no query text contains).
* Training data is synthetic. Constructs the generator never emits (`IN`, `LWT`, UDTs, `BATCH`,
  functions in WHERE, …) are out of distribution; expect lower confidence there.
* Weights are plain safetensors: nothing from the network is executed.

## License

Apache-2.0
