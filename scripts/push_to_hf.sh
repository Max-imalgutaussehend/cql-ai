#!/usr/bin/env bash
# cURL variant of push_to_hf.py. Needs: curl, base64, jq.   HF_TOKEN=hf_... scripts/push_to_hf.sh
set -euo pipefail
REPO="${REPO:-Max-imumlikelihood/cql-anti-pattern-transformer}"
DIR="${DIR:-artifacts}"
CARD="${CARD:-model-card/README.md}"
ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face token with write access}"

# 409 = repository already exists.
code=$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$ENDPOINT/api/repos/create" \
  -H "Authorization: Bearer $HF_TOKEN" -H 'Content-Type: application/json' \
  -d "{\"type\":\"model\",\"name\":\"${REPO#*/}\"}")
[[ "$code" == 200 || "$code" == 409 ]] || { echo "creating repo failed: HTTP $code" >&2; exit 1; }

{
  echo '{"key":"header","value":{"summary":"Upload cql-ai anti-pattern transformer","description":""}}'
  for pair in "model.safetensors:$DIR/model.safetensors" "config.json:$DIR/config.json" "README.md:$CARD"; do
    name="${pair%%:*}"; path="${pair#*:}"
    jq -nc --arg path "$name" --arg content "$(base64 < "$path" | tr -d '\n')" \
      '{key:"file",value:{path:$path,encoding:"base64",content:$content}}'
  done
} | curl -sS --fail-with-body -X POST "$ENDPOINT/api/models/$REPO/commit/main" \
  -H "Authorization: Bearer $HF_TOKEN" -H 'Content-Type: application/x-ndjson' --data-binary @-
echo
echo "uploaded: $ENDPOINT/$REPO"
