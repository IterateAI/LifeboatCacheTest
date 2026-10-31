"""Prepare the public agentic traces (sammshen/lmcache-agentic-traces) for the KV-cache offload benchmark
(sammshen/lmcache-agentic-traces: SWE-bench Verified, GAIA, WildClawBench) for agentic_replay.py.

Each dataset row is one LLM call with the session's full cumulative message list; row N's input
extends row N-1's. This keeps, per session, only what each turn adds (as rendered text), so the
replay rebuilds every prompt as a strict extension of the previous one without holding 2.4 GB of
repeated history in memory. Output: one JSON object per session per line (gzip):
  {"session_id", "source", "turns": [{"delta": str, "output_length": int, "pre_gap": float}, ...]}

usage: agentic_prep.py --out sessions.jsonl.gz [--parquet-dir DIR] [--shards 0,1,2,3,4] [--max-turns 25]
Without --parquet-dir the shards are downloaded from huggingface.co (about 2.4 GB for all five).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import urllib.request

REPO = "sammshen/lmcache-agentic-traces"
URL = "https://huggingface.co/datasets/{repo}/resolve/main/data/train-{i:05d}-of-00005.parquet"


def render(msg: dict) -> str:
    """One message as plain text. Deterministic, so turn N's prompt text is exactly turn N-1's
    plus what turn N adds (the cache only sees a growing prefix, as with the real agent)."""
    parts = [f"<|{msg.get('role', 'user')}|>\n"]
    content = msg.get("content")
    if isinstance(content, list):  # content blocks
        content = "\n".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    if content:
        parts.append(str(content))
    for call in msg.get("tool_calls") or []:
        fn = (call or {}).get("function") or {}
        parts.append(f"\n<tool_call>{fn.get('name', '')}({fn.get('arguments', '')})</tool_call>")
    parts.append("\n")
    return "".join(parts)


def fetch(i: int, cache_dir: str) -> str:
    path = os.path.join(cache_dir, f"train-{i:05d}-of-00005.parquet")
    if not os.path.exists(path):
        tmp = path + ".part"
        print(f"downloading shard {i} -> {path}", file=sys.stderr, flush=True)
        urllib.request.urlretrieve(URL.format(repo=REPO, i=i), tmp)
        os.replace(tmp, path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parquet-dir", default=os.path.expanduser("~/.cache/lifeboat-agentic-traces"))
    ap.add_argument("--shards", default="0,1,2,3,4")
    ap.add_argument("--max-turns", type=int, default=25)
    a = ap.parse_args()
    import pyarrow.parquet as pq

    os.makedirs(a.parquet_dir, exist_ok=True)
    sessions: dict[str, list] = {}
    for i in (int(s) for s in a.shards.split(",")):
        table = pq.read_table(fetch(i, a.parquet_dir), columns=["session_id", "input", "pre_gap", "output_length"])
        for row in table.to_pylist():
            sessions.setdefault(row["session_id"], []).append(row)
    n_turns = n_bad = 0
    with gzip.open(a.out, "wt", encoding="utf-8") as out:
        for sid in sorted(sessions):
            # turn order: each turn's message list extends the previous one's
            rows = sorted(sessions[sid], key=lambda r: len(r["input"]) if not isinstance(r["input"], str)
                          else len(json.loads(r["input"])))
            turns, prev_msgs = [], 0
            for row in rows[: a.max_turns]:
                msgs = row["input"]
                if isinstance(msgs, str):
                    msgs = json.loads(msgs)
                if len(msgs) < prev_msgs:  # not an extension of the previous turn: stop here
                    n_bad += 1
                    break
                turns.append({"delta": "".join(render(m) for m in msgs[prev_msgs:]),
                              "output_length": int(row["output_length"] or 0),
                              "pre_gap": float(row["pre_gap"] or 0.0)})
                prev_msgs = len(msgs)
            if turns:
                n_turns += len(turns)
                out.write(json.dumps({"session_id": sid, "source": sid.split("__", 1)[0], "turns": turns}) + "\n")
    print(f"{a.out}: {len(sessions)} sessions, {n_turns} turns (first {a.max_turns} per session)"
          + (f", {n_bad} sessions cut where a turn didn't extend the previous one" if n_bad else ""))


if __name__ == "__main__":
    main()
