#!/usr/bin/env python3
"""
Backfill statement_pure for lean theorems using a local LLM.

Reads theorems without statement_pure from the KB, sends them to the LLM
in parallel to generate plain-math restatements, then writes back.

Supports two backends:
  --backend llama   llama.cpp /completion API (default: KB_LLM_URL)
  --backend ollama  Ollama /api/generate API (default: http://localhost:11434)

Usage:
    python scripts/backfill_statement_pure.py [--backend ollama] [--model qwen3:0.6b]
        [--llm-url URL] [--workers N] [--project NAME] [--limit N] [--dry-run]

Concurrency: uses a thread pool (--workers) to saturate the GPU.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).parent.parent))

from kb import KnowledgeBase

PROMPT = (
    "Restate the following Lean 4 theorem in pure mathematical language. "
    "No Lean syntax, no type annotations, no imports. Use standard math notation. "
    "One sentence, under 30 words.\n\n"
    "Lean:\n{statement}\n\nMath:"
)


def restate_llama(url: str, tid: str, lean_name: str, statement: str) -> tuple[str, str, str | None]:
    prompt = (
        "<|im_start|>user\n" + PROMPT.format(statement=statement[:600]) + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    try:
        req = Request(url, data=json.dumps({
            "prompt": prompt, "n_predict": 80, "temperature": 0.1,
            "stop": ["<|im_end|>"],
        }).encode(), headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=30) as r:
            result = json.loads(r.read())["content"].strip().strip('"').strip("'")
        return tid, lean_name, result or None
    except Exception as e:
        return tid, lean_name, None


def restate_ollama(url: str, model: str, tid: str, lean_name: str, statement: str) -> tuple[str, str, str | None]:
    prompt = PROMPT.format(statement=statement[:600])
    try:
        req = Request(f"{url}/api/generate", data=json.dumps({
            "model": model, "prompt": prompt, "stream": False,
            "think": False,  # disable thinking for qwen3 thinking models
            "options": {"num_predict": 80, "temperature": 0.1},
        }).encode(), headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
            result = data.get("response", "").strip()
        result = result.strip().strip('"').strip("'")
        return tid, lean_name, result or None
    except Exception as e:
        return tid, lean_name, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["llama", "ollama"], default="ollama")
    parser.add_argument("--model", default="qwen3:0.6b",
                        help="Model name (ollama backend, default: qwen3:0.6b)")
    parser.add_argument("--llm-url", default=None,
                        help="Override endpoint URL")
    parser.add_argument("--workers", type=int, default=32,
                        help="Parallel LLM requests (default: 32)")
    parser.add_argument("--project", default=None, help="Filter by project")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    kb = KnowledgeBase()
    conn = kb._theorems.conn

    if args.backend == "ollama":
        url = args.llm_url or "http://localhost:11434"
        model = args.model
        call = lambda tid, name, stmt: restate_ollama(url, model, tid, name, stmt)
        print(f"Backend: ollama  model={model}  url={url}")
    else:
        import os
        url = args.llm_url or os.environ.get("KB_LLM_URL", "http://tardis:9510/completion")
        call = lambda tid, name, stmt: restate_llama(url, tid, name, stmt)
        print(f"Backend: llama   url={url}")

    where = "WHERE statement_pure IS NULL OR statement_pure = ''"
    params: list = []
    if args.project:
        where += " AND project = ?"
        params.append(args.project)
    if args.limit:
        where += f" LIMIT {args.limit}"

    rows = conn.execute(
        f"SELECT id, lean_name, statement FROM lean_theorems {where}", params
    ).fetchall()
    print(f"Theorems to backfill: {len(rows)}")

    if args.dry_run:
        for tid, lean_name, stmt in rows[:5]:
            print(f"  {lean_name}: {stmt[:80]}")
        return

    updated = failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(call, tid, lean_name, stmt): (tid, lean_name)
            for tid, lean_name, stmt in rows
        }
        for i, fut in enumerate(as_completed(futures), 1):
            tid, lean_name, pure = fut.result()
            if pure:
                conn.execute(
                    "UPDATE lean_theorems SET statement_pure=? WHERE id=?",
                    (pure, tid),
                )
                updated += 1
            else:
                failed += 1

            if i % 100 == 0:
                conn.commit()
                elapsed = time.time() - t0
                rate = i / elapsed
                remaining = (len(rows) - i) / rate if rate > 0 else 0
                print(f"  {i}/{len(rows)} done  {rate:.1f}/s  ETA {remaining/60:.1f}min")

    conn.commit()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Updated: {updated}")
    print(f"  Failed:  {failed}")
    print(f"  Rate:    {updated/elapsed:.1f}/s")

    # Re-embed updated theorems (statement_pure is used as embed_text)
    if updated > 0:
        print(f"\nRe-embedding {updated} updated theorems...")
        re_embedded = 0
        rows2 = conn.execute(
            "SELECT id, statement_pure FROM lean_theorems "
            "WHERE statement_pure IS NOT NULL AND statement_pure != ''"
        ).fetchall()
        for tid, pure in rows2:
            emb = kb._theorems.embedding_service.embed(pure)
            conn.execute("DELETE FROM lean_theorems_vec WHERE id=?", (tid,))
            conn.execute(
                "INSERT INTO lean_theorems_vec (id, embedding) VALUES (?,?)",
                (tid, emb),
            )
            re_embedded += 1
            if re_embedded % 100 == 0:
                conn.commit()
                print(f"  re-embedded {re_embedded}/{len(rows2)}")
        conn.commit()
        print(f"  Re-embedded {re_embedded} theorems.")


if __name__ == "__main__":
    main()
