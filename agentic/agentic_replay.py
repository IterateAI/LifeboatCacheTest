"""Closed-loop agentic replay, the workload of the Intel / HPE / Metrum AI "KV Cache Offload"
report: N users at once, each replaying agent sessions turn by turn (a turn's prompt is the
previous one plus what the agent appended), ROUNDS requests per user, output capped at MAX_OUT
tokens per request. Against any OpenAI-compatible server (vLLM, SGLang, the Lifeboat product).

Reports what the report reports: mean / P99 time to first token, sustained output tokens/s
(output tokens over wall-clock), wall-clock for the run, round-1 vs last-round TTFT, prefix-cache
hit rate (cached prompt tokens from the server's usage), and energy per token (GPU power from
nvidia-smi, CPU package energy from RAPL where readable).

usage: agentic_replay.py --url http://127.0.0.1:18080 --model bench --sessions sessions.jsonl.gz
           --users 256 [--rounds 25] [--max-out 512] [--context 40192] [--api-key K]
           [--out DIR] [--label NAME] [--gaps]
Writes DIR/<label>-requests.csv and DIR/<label>-summary.json and prints the summary.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import gzip
import json
import os
import statistics
import subprocess
import threading
import time

import aiohttp

CHARS_PER_TOKEN = 3.5  # only to skip turns that would exceed --context before sending them


def load_sessions(path: str) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


class Energy:
    """GPU power sampled once a second (nvidia-smi), CPU package energy from RAPL counters."""

    def __init__(self) -> None:
        self.gpu_j = 0.0
        self._stop = threading.Event()
        self._rapl = sorted(glob.glob("/sys/class/powercap/intel-rapl:[0-9]*/energy_uj"))
        self._rapl0 = self._read_rapl()
        self.cpu_j: float | None = None
        self._t = threading.Thread(target=self._sample, daemon=True)
        self._t.start()

    def _read_rapl(self) -> list[int] | None:
        try:
            return [int(open(p).read()) for p in self._rapl] or None
        except OSError:
            return None  # usually root-only

    def _sample(self) -> None:
        last = time.monotonic()
        while not self._stop.wait(1.0):
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                watts = sum(float(x) for x in out.split() if x.replace(".", "", 1).isdigit())
            except (OSError, subprocess.SubprocessError):
                return
            now = time.monotonic()
            self.gpu_j += watts * (now - last)
            last = now

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=5)
        end = self._read_rapl()
        if self._rapl0 and end:
            rng = []
            for p in self._rapl:
                try:
                    rng.append(int(open(os.path.join(os.path.dirname(p), "max_energy_range_uj")).read()))
                except OSError:
                    rng.append(2 ** 32)
            self.cpu_j = sum(((e - s) % r) for s, e, r in zip(self._rapl0, end, rng)) / 1e6


async def ask(http: aiohttp.ClientSession, a, prompt: str, max_tokens: int) -> dict:
    body = {"model": a.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.monotonic()
    first, n_out, usage, err = None, 0, {}, None
    try:
        async with http.post(f"{a.url}/v1/completions", json=body) as r:
            if r.status != 200:
                err = f"HTTP {r.status}: {(await r.text())[:160]}"
            else:
                async for raw in r.content:
                    line = raw.decode().strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    d = json.loads(line[6:])
                    if d.get("choices") and d["choices"][0].get("text"):
                        first = first or time.monotonic() - t0
                        n_out += 1
                    if d.get("usage"):
                        usage = d["usage"]
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        err = repr(e)[:160]
    end = time.monotonic() - t0
    return {"ttft": first, "e2e": end, "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "out_tokens": usage.get("completion_tokens", n_out), "error": err}


async def user(k: int, a, sessions: list[dict], http, rows: list, t_start: float) -> None:
    """User k replays sessions k, k+N, k+2N, ... turn by turn until it made a.rounds requests."""
    done, s_idx = 0, k
    while done < a.rounds:
        sess = sessions[s_idx % len(sessions)]
        s_idx += a.users
        prompt = f"[user {k}]\n"  # keeps users' sessions apart when a session is reused
        for t, turn in enumerate(sess["turns"]):
            if done >= a.rounds:
                break
            prompt += turn["delta"]
            if len(prompt) / CHARS_PER_TOKEN > a.context - a.max_out:
                break  # this session outgrew the context window: the user moves on
            if a.gaps and t and turn["pre_gap"]:
                await asyncio.sleep(turn["pre_gap"])
            max_tokens = max(1, min(a.max_out, turn["output_length"] or a.max_out))
            r = await ask(http, a, prompt + "<|assistant|>\n", max_tokens)
            done += 1
            rows.append({"user": k, "session": sess["session_id"], "turn": t + 1, "round": done,
                         "t_start": round(time.monotonic() - t_start - r["e2e"], 3), **r})
            if r["error"]:
                break
        if s_idx >= k + a.users * len(sessions):
            break  # every session tried


def summarize(rows: list[dict], wall: float, energy: Energy, a) -> dict:
    ok = [r for r in rows if not r["error"] and r["ttft"] is not None]
    ttft = [r["ttft"] for r in ok]
    out_tok = sum(r["out_tokens"] or 0 for r in ok)
    prompt = sum(r["prompt_tokens"] or 0 for r in ok)
    cached = sum(r["cached_tokens"] or 0 for r in ok)
    by_round: dict[int, list[float]] = {}
    for r in ok:
        by_round.setdefault(r["round"], []).append(r["ttft"])
    last = max(by_round) if by_round else None
    r1 = statistics.mean(by_round[1]) if 1 in by_round else None
    rl = statistics.mean(by_round[last]) if last else None
    joules = energy.gpu_j + (energy.cpu_j or 0.0)
    return {"label": a.label, "users": a.users, "rounds": a.rounds, "max_out": a.max_out, "context": a.context,
            "requests": len(rows), "ok": len(ok), "errors": len(rows) - len(ok), "wall_s": round(wall, 1),
            "ttft_mean_s": round(statistics.mean(ttft), 3) if ttft else None,
            "ttft_p50_s": pct(ttft, 50), "ttft_p99_s": pct(ttft, 99),
            "out_tok_s_sustained": round(out_tok / wall, 2) if wall else None,
            "output_tokens": out_tok, "prompt_tokens": prompt,
            "prefix_hit_rate": round(cached / prompt, 4) if prompt else None,
            "ttft_round1_s": round(r1, 3) if r1 else None, f"ttft_round{last}_s": round(rl, 3) if rl else None,
            "round1_to_last_pct": round(100 * (rl - r1) / r1, 1) if r1 and rl else None,
            "gpu_joules": round(energy.gpu_j, 1), "cpu_joules": round(energy.cpu_j, 1) if energy.cpu_j else None,
            "tok_per_joule": round(out_tok / joules, 4) if joules else None}


async def main_async(a) -> None:
    sessions = load_sessions(a.sessions)
    if a.sources:
        sessions = [s for s in sessions if s["source"] in a.sources.split(",")]
    headers = {"Authorization": f"Bearer {a.api_key}"} if a.api_key else {}
    timeout = aiohttp.ClientTimeout(total=a.timeout)
    rows: list[dict] = []
    energy = Energy()
    t0 = time.monotonic()
    async with aiohttp.ClientSession(headers=headers, timeout=timeout,
                                     connector=aiohttp.TCPConnector(limit=0)) as http:
        await asyncio.gather(*(user(k, a, sessions, http, rows, t0) for k in range(a.users)))
    wall = time.monotonic() - t0
    energy.stop()
    s = summarize(rows, wall, energy, a)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.label}-requests.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]) if rows else ["user"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, f"{a.label}-summary.json"), "w") as fh:
        json.dump(s, fh, indent=1)
    print(json.dumps(s), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--users", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--max-out", type=int, default=512)
    ap.add_argument("--context", type=int, default=131072, help="the server's max model length")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--sources", default="", help="e.g. swebench,gaia,wildclaw (default: all)")
    ap.add_argument("--gaps", action="store_true", help="wait each turn's recorded tool time (pre_gap)")
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--out", default="agentic-results")
    ap.add_argument("--label", default="run")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
