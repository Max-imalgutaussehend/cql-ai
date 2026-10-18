#!/usr/bin/env python3
"""Publish the trained model to Hugging Face (stdlib only, no huggingface_hub needed).

    export HF_TOKEN=hf_...            # a *write* token for Max-imalgutaussehend
    python3 scripts/push_to_hf.py     # uploads artifacts/ + model-card/README.md
    python3 scripts/push_to_hf.py --dry-run

Creates the model repo if needed and uploads model.safetensors, config.json and the model card in
one commit through the documented NDJSON commit API (files here are a few MB, so they are sent inline
as base64; larger files would need the LFS flow).
"""
import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_REPO = "Max-imumlikelihood/cql-anti-pattern-transformer"


def request(url, token, data=None, content_type="application/json"):
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO, help="namespace/name of the model repo")
    ap.add_argument("--dir", default="artifacts", help="directory with model.safetensors and config.json")
    ap.add_argument("--card", default="model-card/README.md", help="model card to upload as README.md")
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build the request, print a summary, send nothing")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not token and not args.dry_run:
        sys.exit("error: set HF_TOKEN to a Hugging Face token with write access")

    files = {
        "model.safetensors": os.path.join(args.dir, "model.safetensors"),
        "config.json": os.path.join(args.dir, "config.json"),
        "README.md": args.card,
    }
    for name, path in files.items():
        if not os.path.isfile(path):
            sys.exit(f"error: missing {path} (needed as {name})")

    lines = [{"key": "header", "value": {"summary": "Upload cql-ai anti-pattern transformer", "description": ""}}]
    for name, path in files.items():
        with open(path, "rb") as f:
            content = base64.b64encode(f.read()).decode()
        lines.append({"key": "file", "value": {"content": content, "path": name, "encoding": "base64"}})
    body = ("\n".join(json.dumps(line) for line in lines) + "\n").encode()

    namespace, _, name = args.repo.partition("/")
    if args.dry_run:
        print(f"would create {args.repo} (if missing) and commit {', '.join(files)} ({len(body)} bytes)")
        return

    status, text = request(f"{args.endpoint}/api/whoami-v2", token)
    if status != 200:
        sys.exit(f"error: token rejected ({status}): {text}")
    who = json.loads(text)
    orgs = [o.get("name") for o in who.get("orgs", [])]
    if namespace != who.get("name") and namespace not in orgs:
        sys.exit(
            f"error: this token belongs to Hugging Face user '{who.get('name')}', not '{namespace}'.\n"
            f"       Re-run with --repo {who.get('name')}/{name}  (and set CQL_AI_HF_REPO for the CLI, see README)."
        )

    status, text = request(
        f"{args.endpoint}/api/repos/create",
        token,
        json.dumps({"type": "model", "name": name, "organization": None, "private": args.private}).encode(),
    )
    if status not in (200, 409):  # 409: repo already exists
        sys.exit(f"error: creating repo failed ({status}): {text}")
    print(f"repo {args.repo}: {'exists' if status == 409 else 'created'}")

    status, text = request(
        f"{args.endpoint}/api/models/{args.repo}/commit/main", token, body, "application/x-ndjson"
    )
    if status != 200:
        sys.exit(f"error: commit failed ({status}): {text}")
    print(f"uploaded: {args.endpoint}/{args.repo}")


if __name__ == "__main__":
    main()
