"""The Intel / HPE / Metrum AI KV-cache-offload grid, plus LifeboatCache, plus a DRAM sweep.

For every cell (engine, configuration, DRAM GB, users) the server starts fresh with empty cache
tiers (as in the report: caches cleared and the server restarted before each concurrency
change), agentic_replay.py runs, and a row lands in intel_summary.csv. Afterwards:
intel_sla.csv (most users within a P99-TTFT target of 30 / 300 / 900 s per configuration) and
intel_dram.csv (the same user count across DRAM sizes: how throughput and TTFT scale with DRAM).

Configurations (names follow the report; DRAM = host-RAM KV tier per GPU, NVMe = local disk):
  vllm     gpu_only            vLLM prefix cache only
           lmcache_dram        vLLM + LMCache, DRAM only             (report: "+ CPU DRAM")
           lmcache_dram_nvme   vLLM + LMCache, DRAM + LMCache's NVMe (report: "+ DRAM + NVMe")
           lifeboat            vLLM offloading, DRAM + LifeboatCache on NVMe
  sglang   radix               SGLang radix cache only
           lifeboat            SGLang HiCache, DRAM + LifeboatCache on NVMe
  product  radix / lifeboat    the same two inside the Lifeboat product image (docker)

usage: intel_sweep.py --model-dir DIR --sessions sessions.jsonl.gz --out results/
          [--engines vllm,sglang,product] [--configs gpu_only,lmcache_dram,...] [--dram 16,32,64]
          [--users 32,128,256,512] [--rounds 25] [--max-out 512] [--context 40192] [--tp 1]
          [--kv-dtype auto] [--sglang-python PATH] [--image iterateai/lifeboat:2.2.61-amd64]
Resumable: a cell whose summary exists is skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = HERE  # serve.sh sits next to this script
VLLM_MODES = {"gpu_only": "vllm", "lmcache_dram": "lmcache", "lmcache_dram_nvme": "lmcache_fs",
              "lifeboat": "lb_native", "lmcache_lifeboat": "lmcache_lb"}
ENGINE_CONFIGS = {"vllm": ["gpu_only", "lmcache_dram", "lmcache_dram_nvme", "lifeboat"],
                  "sglang": ["radix", "lifeboat"], "product": ["radix", "lifeboat"]}
NO_DRAM = {"gpu_only", "radix"}  # configurations without a host-RAM tier: one DRAM column
PORTS = {"vllm": 18080, "sglang": 30010, "product": 30010}


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def wait_ready(port: int, proc_alive, timeout: float = 1800) -> None:
    t = time.time()
    while time.time() - t < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            return
        except OSError:
            if not proc_alive():
                raise RuntimeError("server exited during startup")
            time.sleep(3)
    raise RuntimeError("server did not become ready")


class Server:
    """One server for one cell; stop() leaves the GPU free for the next."""

    def __init__(self, a, engine: str, config: str, dram: int, home: str) -> None:
        self.a, self.engine, self.config, self.dram, self.home = a, engine, config, dram, home
        self.proc = None
        self.port = PORTS[engine]
        self.api_key = ""

    def start(self) -> None:
        a, env = self.a, dict(os.environ)
        os.makedirs(self.home, exist_ok=True)
        if self.engine == "vllm":
            env.update(MODEL_DIR=a.model_dir, TP=str(a.tp), KV_DTYPE=a.kv_dtype, CPU_GB=str(self.dram or 1),
                       MAX_LEN=str(a.context), MAX_SEQS=str(a.max_seqs), GPU_UTIL=str(a.gpu_util),
                       SERVE_EXTRA=a.extra_args, CHUNK=str(a.chunk))
            out = subprocess.run([os.path.join(MATRIX, "serve.sh"), "custom", VLLM_MODES[self.config]],
                                 env=env, capture_output=True, text=True)
            if "ready" not in out.stdout:
                raise RuntimeError(f"serve.sh failed: {out.stdout[-400:]} {out.stderr[-400:]}")
            root = env.get("EVAL_ROOT", os.path.expanduser("~/lifeboat-eval"))
            keys = os.path.join(root, "vllm-api-keys")
            self.api_key = open(keys).read().split()[0] if os.path.exists(keys) else ""
            return
        lb = self.config == "lifeboat"
        if self.engine == "sglang":
            if lb:
                env.update(LIFEBOAT_CACHE="1", LIFEBOAT_CACHE_HOME=self.home, LIFEBOAT_SGLANG_HOST_GB=str(self.dram))
            else:
                env["LIFEBOAT_CACHE"] = "0"
            cmd = [a.sglang_python, "-m", "sglang.launch_server", "--model-path", a.model_dir,
                   "--served-model-name", "bench", "--host", "127.0.0.1", "--port", str(self.port),
                   "--tp", str(a.tp), "--context-length", str(a.context), "--enable-cache-report",
                   "--mem-fraction-static", str(a.gpu_util)] + a.extra_args.split()
            if sys.platform == "darwin":  # SGLang's Apple-silicon (MLX) runtime
                env["SGLANG_USE_MLX"] = "1"
                cmd.append("--disable-cuda-graph")
            self.log = open(os.path.join(self.home, "server.log"), "w")
            self.proc = subprocess.Popen(cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            wait_ready(self.port, lambda: self.proc.poll() is None)
            return
        # the Lifeboat product image (SGLang inside)
        name = "lb-intel"
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        envs = ["-e", "HOME=/opt/lifeboat/.lifeboat", "-e", "FLASHINFER_DISABLE_VERSION_CHECK=1"]
        mounts = ["-v", f"{a.models_root}:/models:ro"]
        if lb:
            envs += ["-e", "LIFEBOAT_CACHE=1", "-e", "PYTHONPATH=/opt/lbcache", "-e", "LIFEBOAT_CACHE_HOME=/lbc",
                     "-e", f"LIFEBOAT_SGLANG_HOST_GB={self.dram}"]
            mounts += ["-v", f"{a.lbc_pylib}:/opt/lbcache:ro", "-v", f"{self.home}:/lbc"]
        model_in = "/models/" + os.path.relpath(a.model_dir, a.models_root)
        subprocess.run(["docker", "run", "-d", "--name", name, "--device", "nvidia.com/gpu=all", "--network", "host",
                        "--shm-size", "8g", "--user", "lifeboat:lifeboat", *mounts, *envs,
                        "--entrypoint", "/opt/lifeboat/venv/bin/python", a.image, "-m", "lifeboat.engine",
                        "--model-path", model_in, "--served-model-name", "bench", "--host", "127.0.0.1",
                        "--port", str(self.port), "--tp", str(a.tp), "--context-length", str(a.context),
                        "--mem-fraction-static", str(a.gpu_util), *a.extra_args.split()], check=True,
                       capture_output=True)
        alive = lambda: subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name],  # noqa: E731
                                       capture_output=True, text=True).stdout.strip() == "true"
        wait_ready(self.port, alive)

    def stop(self) -> None:
        if self.engine == "vllm":
            subprocess.run([os.path.join(MATRIX, "serve.sh"), "custom", "stop"], capture_output=True)
        elif self.engine == "sglang" and self.proc is not None:
            try:
                os.killpg(self.proc.pid, 15)
                self.proc.wait(120)
            except (ProcessLookupError, PermissionError):
                pass  # already exited (macOS reports a gone group as EPERM)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, 9)
        elif self.engine == "product":
            subprocess.run(["docker", "rm", "-f", "lb-intel"], capture_output=True)


def run_cell(a, engine: str, config: str, dram: int, users: int) -> dict | None:
    label = f"{engine}-{config}-dram{dram}-u{users}"
    summary = os.path.join(a.out, "cells", f"{label}-summary.json")
    if os.path.exists(summary):
        return json.load(open(summary))
    home = os.path.join(a.out, "homes", f"{engine}-{config}-dram{dram}")
    subprocess.run(["rm", "-rf", home])
    srv = Server(a, engine, config, dram, home)
    log("cell", label, "starting server")
    try:
        srv.start()
        cmd = [sys.executable, os.path.join(HERE, "agentic_replay.py"), "--url", f"http://127.0.0.1:{srv.port}",
               "--model", "bench", "--sessions", a.sessions, "--users", str(users), "--rounds", str(a.rounds),
               "--max-out", str(a.max_out), "--context", str(a.context), "--out", os.path.join(a.out, "cells"),
               "--label", label] + (["--api-key", srv.api_key] if srv.api_key else []) + (["--gaps"] if a.gaps else [])
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            log("cell", label, "replay failed:", out.stderr[-400:])
            return None
        s = json.load(open(summary))
        if engine == "sglang" and srv.proc is not None and srv.proc.poll() is not None:
            s["server_crashed"] = True  # partial: the engine died during the run
            json.dump(s, open(summary, "w"), indent=1)
        log("cell", label, f"ttft mean {s['ttft_mean_s']}s p99 {s['ttft_p99_s']}s, "
            f"{s['out_tok_s_sustained']} tok/s, hit {s['prefix_hit_rate']}, wall {s['wall_s']}s")
        return s
    except Exception as e:  # noqa: BLE001 - one failed cell must not end the sweep
        log("cell", label, "failed:", e)
        return None
    finally:
        srv.stop()


def saved_rows(a) -> list[dict]:
    """Every cell summary under <out>/cells, so a resumed or split run's tables cover all cells."""
    import glob
    import re
    rows = []
    for path in sorted(glob.glob(os.path.join(a.out, "cells", "*-summary.json"))):
        m = re.match(r"(\w+)-(\w+)-dram(\d+)-u(\d+)-summary\.json$", os.path.basename(path))
        if m:
            rows.append({"engine": m[1], "config": m[2], "dram_gb": int(m[3]), **json.load(open(path))})
    rows.sort(key=lambda r: (r["engine"], r["config"], r["dram_gb"], r["users"]))
    return rows


def write_tables(a, rows: list[dict]) -> None:
    if not rows:
        return
    cols = ["engine", "config", "dram_gb", "users"]
    for r in rows:  # every column any cell has (server_crashed only where it happened)
        cols += [k for k in r if k not in cols and k != "label"]
    with open(os.path.join(a.out, "intel_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    sla = []
    for key in sorted({(r["engine"], r["config"], r["dram_gb"]) for r in rows}):
        cell = {"engine": key[0], "config": key[1], "dram_gb": key[2]}
        for target in (30, 300, 900):
            ok = [r["users"] for r in rows if (r["engine"], r["config"], r["dram_gb"]) == key
                  and r.get("ttft_p99_s") is not None and r["ttft_p99_s"] <= target]
            cell[f"max_users_p99_le_{target}s"] = max(ok) if ok else 0
        sla.append(cell)
    with open(os.path.join(a.out, "intel_sla.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(sla[0]))
        w.writeheader()
        w.writerows(sla)
    keep = ["engine", "config", "dram_gb", "users", "ttft_mean_s", "ttft_p99_s", "out_tok_s_sustained",
            "prefix_hit_rate", "wall_s", "tok_per_joule"]
    dram = sorted((r for r in rows if r["config"] not in NO_DRAM),
                  key=lambda r: (r["engine"], r["config"], r["users"], r["dram_gb"]))
    with open(os.path.join(a.out, "intel_dram.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, keep, extrasaction="ignore")
        w.writeheader()
        w.writerows(dram)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engines", default="vllm")
    ap.add_argument("--configs", default="", help="default: every configuration of each engine")
    ap.add_argument("--dram", default="16,32,64,128", help="host-RAM KV tier sizes, GB per GPU")
    ap.add_argument("--users", default="32,128,256,512")
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--max-out", type=int, default=512)
    ap.add_argument("--context", type=int, default=131072)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--kv-dtype", default="auto", help="the report used FP16 KV (auto)")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--max-seqs", type=int, default=512)
    ap.add_argument("--gpu-util", type=float, default=0.80, help="the report used 0.80")
    ap.add_argument("--extra-args", default="", help="extra engine arguments, e.g. --moe-backend triton")
    ap.add_argument("--gaps", action="store_true")
    ap.add_argument("--sglang-python", default=sys.executable)
    ap.add_argument("--image", default="iterateai/lifeboat:2.2.61-amd64")
    ap.add_argument("--models-root", default=os.path.expanduser("~/models"))
    ap.add_argument("--lbc-pylib", default="", help="lifeboatcache installed with pip --target, for the product image")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "cells"), exist_ok=True)
    configs = set(a.configs.split(",")) if a.configs else None
    rows: list[dict] = []
    for engine in a.engines.split(","):
        for config in ENGINE_CONFIGS[engine]:
            if configs and config not in configs:
                continue
            for dram in ([0] if config in NO_DRAM else [int(x) for x in a.dram.split(",")]):
                for users in (int(x) for x in a.users.split(",")):
                    s = run_cell(a, engine, config, dram, users)
                    if s:
                        rows.append({"engine": engine, "config": config, "dram_gb": dram, **s})
                        write_tables(a, saved_rows(a))
    log("sweep done:", len(rows), "cells ->", a.out)


if __name__ == "__main__":
    main()
