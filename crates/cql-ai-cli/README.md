# cql-ai-cli

Installs `cql-ai`, a CI linter that flags CQL anti-patterns with a tiny Transformer running on
[cqlc-core](https://crates.io/crates/cqlc-core) lexer tokens. Weights are fetched once from
[Hugging Face](https://huggingface.co/Max-imumlikelihood/cql-anti-pattern-transformer) and cached in `~/.cache/cql-ai/`.

```sh
cargo install cql-ai-cli
cql-ai check queries/ --schema schema/ --deny-warnings
```

Part of the **cqlz family**; see the [repository](https://github.com/Max-imalgutaussehend/cql-ai).
