#!/usr/bin/env python3
"""Lifeboat Cache verification: runs your model on your vLLM or SGLang, with and without Lifeboat
Cache, and writes the results as CSV.

What it measures, sized to this machine's GPUs, RAM and disk:
  capacity    GPU KV pool (tokens) and how many users of the test context fit on the GPU at once
  throughput  total output tokens/s and time to first token at 1..64 concurrent users
  returning   users who come back with a new question after their KV left the GPU — the case a
              KV cache tier exists for: recompute (no offload) vs the engine's own offload vs
              Lifeboat Cache (and LMCache / LMCache + Lifeboat Cache on vLLM when available)
  overflow    more users than the GPU holds (about 2x its KV pool), so eviction is real
  restart     a returning user after the server restarts

Nothing leaves this machine. A server already using the GPU is stopped only with your consent
and restarted afterwards with the same command and environment.

usage: lbc_verify.py [--yes] [--full] [--engine-python PATH] [--model PATH_OR_ID] [--workdir DIR]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import glob
import http.client
import json
import os
import random
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time

VERSION = "1.0.0"
K = 1024
PORT = 18555
DASH_PORT = 18556
QUESTIONS = ["\n\nQuestion: In one paragraph, what is the main purpose of the code above?\nAnswer:",
             "\n\nQuestion: List the most important classes or functions defined above.\nAnswer:",
             "\n\nQuestion: What would you refactor first in the code above, and why?\nAnswer:"]
LABELS = {"vllm": "vLLM, no offload", "vllm_offload": "vLLM native offload (CPU + disk)",
          "lifeboat": "Lifeboat Cache", "lmcache": "LMCache", "lmcache_lifeboat": "LMCache + Lifeboat Cache",
          "sglang": "SGLang, no offload", "sglang_hicache": "SGLang HiCache (file backend)"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


# ---------------------------------------------------------------- this machine
def system_info(workdir):
    gpus = []
    for line in sh(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version",
                    "--format=csv,noheader,nounits"]).splitlines():
        i, name, total, used, drv = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(i), "name": name, "mem_mib": int(total), "used_mib": int(used), "driver": drv})
    cuda = re.search(r"CUDA Version: ([0-9.]+)", sh(["nvidia-smi"]))
    mem = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        mem[k] = int(v.split()[0]) * 1024
    shm = shutil.disk_usage("/dev/shm") if os.path.isdir("/dev/shm") else None
    disk = shutil.disk_usage(workdir)
    fs = sh(["df", "-T", workdir]).splitlines()[-1].split()[1] if sh(["df", "-T", workdir]) else "?"
    cpu = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")), "?")
    return {"gpus": gpus, "cuda_driver": cuda.group(1) if cuda else "?", "cpu": cpu, "cores": os.cpu_count(),
            "ram_gb": mem["MemTotal"] / 2**30, "ram_avail_gb": mem["MemAvailable"] / 2**30,
            "shm_gb": shm.total / 2**30 if shm else 0, "shm_free_gb": shm.free / 2**30 if shm else 0,
            "disk_free_gb": disk.free / 2**30, "fs": fs}


# ---------------------------------------------------------------- the engine you run
def proc_args(pid):
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace").split("\0")[:-1]
    except OSError:
        return []


def find_servers():
    """vLLM / SGLang API servers running on this machine (parent processes only)."""
    found = []
    for d in glob.glob("/proc/[0-9]*"):
        pid = int(d.rsplit("/", 1)[1])
        a = proc_args(pid)
        s = " ".join(a)
        if " serve " not in f" {s} " and "api_server" not in s and "sglang.launch_server" not in s:
            continue
        kind = "sglang" if "sglang" in s else "vllm" if "vllm" in s else None
        if kind is None or pid == os.getpid():
            continue
        try:
            ppid = int(open(f"{d}/stat").read().rsplit(")", 1)[1].split()[1])
        except OSError:
            continue
        found.append({"pid": pid, "ppid": ppid, "kind": kind, "argv": a})
    pids = {f["pid"] for f in found}
    return [f for f in found if f["ppid"] not in pids]


def split_launch(kind, argv):
    """argv -> (launcher prefix, model, the rest of the flags)."""
    model, rest, launcher = None, [], []
    i = 0
    if "serve" in argv:
        j = argv.index("serve")
        launcher = argv[:j + 1]
        if kind == "vllm" and j + 1 < len(argv) and not argv[j + 1].startswith("-"):
            model, i = argv[j + 1], j + 2
        else:
            i = j + 1
    else:
        j = next((k for k, x in enumerate(argv) if x == "-m"), None)
        if j is not None and j + 1 < len(argv):
            launcher, i = argv[:j + 2], j + 2
        else:
            launcher, i = argv[:1], 1
    while i < len(argv):
        x = argv[i]
        if x in ("--model", "--model-path") and i + 1 < len(argv):
            model, i = argv[i + 1], i + 2
            continue
        if x.startswith(("--model=", "--model-path=")):
            model, i = x.split("=", 1)[1], i + 1
            continue
        rest.append(x)
        i += 1
    return launcher, model, rest


def strip_flags(args, drop):
    """Remove flags (and their values) this tool sets itself."""
    out, i = [], 0
    while i < len(args):
        x = args[i]
        name = x.split("=", 1)[0]
        if name in drop or any(name.startswith(p) for p in drop if p.endswith("-")):
            i += 1
            if "=" not in x and i < len(args) and not args[i].startswith("--"):
                i += 1
            continue
        out.append(x)
        i += 1
    return out


def flag(args, *names, default=None):
    for i, x in enumerate(args):
        for n in names:
            if x == n and i + 1 < len(args):
                return args[i + 1]
            if x.startswith(n + "="):
                return x.split("=", 1)[1]
    return default


def has_module(py, mod):
    return subprocess.run([py, "-c", f"import importlib.util as u,sys;sys.exit(u.find_spec('{mod}') is None)"],
                          capture_output=True).returncode == 0


def pkg_version(py, name):
    return sh([py, "-c", f"from importlib.metadata import version;print(version('{name}'))"]) or None


def find_engine_python(kind=None):
    cands = ["python3", "python", os.path.join(os.environ.get("CONDA_PREFIX", "/nonexistent"), "bin/python"),
             "/opt/conda/bin/python", "/usr/local/bin/python3", "/usr/bin/python3", "/opt/venv/bin/python",
             "/workspace/.venv/bin/python", os.path.expanduser("~/.venv/bin/python")]
    for c in cands:
        p = shutil.which(c) or (c if os.path.exists(c) else None)
        if not p:
            continue
        for k in ([kind] if kind else ["vllm", "sglang"]):
            if has_module(p, k):
                return p, k
    return None, None


def engine_python_of(target):
    """The interpreter that runs the server, as launched (a venv's python, not the resolved
    base interpreter, which wouldn't see the venv's packages)."""
    a0 = target["argv"][0]
    try:
        cwd = os.readlink(f"/proc/{target['pid']}/cwd")
    except OSError:
        cwd = "/"
    path = a0 if os.path.isabs(a0) else shutil.which(a0) or os.path.join(cwd, a0)
    cands = []
    if os.path.basename(path).startswith("python"):
        cands.append(path)
    else:
        try:
            line = open(path, "rb").readline().decode(errors="ignore")
            if line.startswith("#!"):
                parts = line[2:].split()
                cands.append(shutil.which(parts[1]) if os.path.basename(parts[0]) == "env" and len(parts) > 1 else parts[0])
        except OSError:
            pass
    cands.append(os.path.realpath(f"/proc/{target['pid']}/exe"))
    for c in cands:
        if c and os.path.exists(c) and has_module(c, target["kind"]):
            return c
    return None


def in_other_namespace(pid):
    try:
        return os.readlink(f"/proc/{pid}/ns/mnt") != os.readlink("/proc/self/ns/mnt")
    except OSError:
        return True


def systemd_unit(pid):
    try:
        for line in open(f"/proc/{pid}/cgroup"):
            m = re.search(r"/([^/]+\.service)$", line.strip())
            if m and not m.group(1).startswith(("user@", "session-")):
                return m.group(1)
    except OSError:
        pass
    return None


def ask(question):
    try:
        with open("/dev/tty") as tty:
            print(question + " [y/N] ", end="", flush=True)
            return tty.readline().strip().lower() in ("y", "yes")
    except OSError:
        return False


# ---------------------------------------------------------------- servers
class Server:
    def __init__(self, ctx, config, keep_tier=False):
        self.ctx, self.config, self.keep = ctx, config, keep_tier
        self.home = os.path.join(ctx.work, "tiers", config)
        self.log_path = os.path.join(ctx.work, "logs", f"{config}-{int(time.time())}.log")
        self.proc = None

    def command(self):
        c = self.ctx
        env = dict(os.environ, **c.engine_env)
        if not self.keep:
            shutil.rmtree(self.home, ignore_errors=True)
        os.makedirs(self.home, exist_ok=True)
        env.pop("LIFEBOAT_CACHE", None)
        args = list(c.base_args)
        if c.kind == "vllm":
            env["VLLM_SERVER_DEV_MODE"] = "1"  # /reset_prefix_cache
            args += ["--host", "127.0.0.1", "--port", str(PORT), "--served-model-name", "verify",
                     "--enable-prefix-caching", "--enable-prompt-tokens-details"]
            if self.config == "vllm_offload":
                kt = {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
                      "kv_connector_extra_config": {"spec_name": "TieringOffloadingSpec",
                                                    "cpu_bytes_to_use": int(c.cpu_gb * 2**30),
                                                    "secondary_tiers": [{"type": "fs", "root_dir": self.home + "/fs"}]}}
                args += ["--kv-transfer-config", json.dumps(kt)]
            elif self.config == "lmcache":
                srv = shlex.join(["--l1-size-gb", str(c.cpu_gb), "--eviction-policy", "LRU"])
                kt = {"kv_connector": "LMCacheMPConnector", "kv_role": "kv_both",
                      "kv_connector_extra_config": {"lmcache.mp.host": "tcp://localhost", "lmcache.mp.port": 6555,
                                                    "lmcache.mp.autostart": True,
                                                    "lmcache.mp.autostart.server_args": srv,
                                                    "lmcache.mp.autostart.wait_timeout": 180}}
                args += ["--kv-transfer-config", json.dumps(kt)]
            elif self.config in ("lifeboat", "lmcache_lifeboat"):
                self.lifeboat_env(env, "native" if self.config == "lifeboat" else "lmcache")
        else:
            args += ["--host", "127.0.0.1", "--port", str(PORT), "--served-model-name", "verify",
                     "--enable-cache-report"]
            if self.config == "sglang_hicache":
                env["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = self.home + "/file"
                args += ["--enable-hierarchical-cache", "--hicache-storage-backend", "file",
                         "--hicache-size", str(int(c.cpu_gb))]
                if not flag(args, "--page-size"):
                    args += ["--page-size", "64"]
            elif self.config == "lifeboat":
                self.lifeboat_env(env, None)
                env["LIFEBOAT_SGLANG_HOST_GB"] = str(int(c.cpu_gb))
        return c.launcher + ([c.model] if c.kind == "vllm" and c.launcher[-1] == "serve" else
                             ["--model-path" if c.kind == "sglang" else "--model", c.model]) + args, env

    def lifeboat_env(self, env, vllm_mode):
        c = self.ctx
        cfg = os.path.join(self.home, "lifeboat.yaml")
        lines = ["storage:", "  write_mode: low_latency", "  fallback_local:", f"    path: {self.home}/kv",
                 f"    capacity_gb: {int(c.disk_gb)}", "eviction:", "  admission:", "    enabled: false",
                 "tenancy:", "  enabled: false", "telemetry:", "  prometheus:", f"    port: {DASH_PORT + 1}",
                 "dashboard:", f"  port: {DASH_PORT}", f"  auth_token: {c.dash_token}", "engine:",
                 f"  native_cpu_gb: {c.cpu_gb}", f"  lmcache_l1_size_gb: {c.cpu_gb}"]
        if vllm_mode:
            lines.append(f"  vllm_mode: {vllm_mode}")
        with open(cfg, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        env.update(LIFEBOAT_CACHE="1", LIFEBOAT_CACHE_CONFIG=cfg, LIFEBOAT_CACHE_HOME=self.home)

    def start(self):
        cmd, env = self.command()
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        log(f"starting {LABELS[self.config]}")
        t0 = time.time()
        self.proc = subprocess.Popen(cmd, env=env, stdout=open(self.log_path, "w"), stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True, cwd=self.ctx.work)
        deadline = time.time() + self.ctx.start_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.config} exited while starting:\n{self.tail()}")
            if http_get("/v1/models") is not None:
                self.ready_s = round(time.time() - t0, 1)
                warmup()
                log(f"  ready in {self.ready_s}s")
                return self
            time.sleep(2)
        self.stop()
        raise RuntimeError(f"{self.config} did not become ready in {self.ctx.start_timeout}s:\n{self.tail()}")

    def tail(self, n=25):
        try:
            return "".join(open(self.log_path, errors="ignore").readlines()[-n:])
        except OSError:
            return ""

    def text(self):
        try:
            return open(self.log_path, errors="ignore").read()
        except OSError:
            return ""

    def kv_tokens(self):
        t = self.text()
        m = re.search(r"GPU KV cache size: ([0-9,]+) tokens", t) or re.search(r"max_total_num_tokens[=: ]+(\d+)", t)
        return int(m.group(1).replace(",", "")) if m else None

    def attached(self):
        """Did the cache under test actually attach (a silently ignored plugin would make the
        numbers meaningless)? Lifeboat: its log line, or its dashboard answering."""
        if self.config not in ("lifeboat", "lmcache_lifeboat"):
            return True
        for _ in range(30):
            t = self.text()
            if "LIFEBOAT_CACHE=1: Lifeboat Cache via" in t or "HiCache enabled with the Lifeboat storage backend" in t \
                    or lifeboat_stats(self.ctx) is not None:
                return True
            time.sleep(2)
        return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(90)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        wait_gpu_free()


def wait_gpu_free(limit_s=180):
    t0 = time.time()
    while time.time() - t0 < limit_s:
        used = [int(x) for x in sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()]
        if not used or max(used) < 2000:
            return
        time.sleep(2)


# ---------------------------------------------------------------- client (stdlib only)
def http_get(path, port=PORT, headers=None, timeout=5):
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        c.request("GET", path, headers=headers or {})
        r = c.getresponse()
        body = r.read()
        return json.loads(body) if r.status == 200 else None
    except (OSError, ValueError, http.client.HTTPException):
        return None


def http_post(path, body=b"", timeout=120):
    try:
        c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
        c.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        r = c.getresponse()
        r.read()
        return r.status == 200
    except (OSError, http.client.HTTPException):
        return False


def complete(prompt, max_tokens):
    body = json.dumps({"model": "verify", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
                       "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}})
    t0 = time.perf_counter()
    first = last = None
    usage = {}
    try:
        c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=3600)
        c.request("POST", "/v1/completions", body=body, headers={"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            return {"error": f"http {r.status}: {r.read()[:200]!r}"}
        while True:
            line = r.readline()
            if not line:
                break
            line = line.decode(errors="ignore").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("choices") and d["choices"][0].get("text"):
                now = time.perf_counter()
                first = first or now
                last = now
            if d.get("usage"):
                usage = d["usage"]
    except (OSError, ValueError, http.client.HTTPException) as e:
        return {"error": repr(e)[:200]}
    end = time.perf_counter()
    first = first or end
    n = usage.get("completion_tokens") or 0
    return {"ttft_s": first - t0, "e2e_s": end - t0, "t0": t0, "t_end": end, "prompt_tokens": usage.get("prompt_tokens"),
            "out_tokens": n, "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "decode_tok_s": (n - 1) / (last - first) if n > 1 and last and last > first else None}


def warmup():
    complete("Hello", 4)


def burst(prompts, max_tokens=64):
    with cf.ThreadPoolExecutor(max(1, len(prompts))) as ex:
        return list(ex.map(lambda p: complete(p, max_tokens), prompts))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))] if xs else None


def summarize(rs):
    ok = [r for r in rs if "error" not in r]
    out = {"requests": len(rs), "errors": len(rs) - len(ok)}
    if ok:
        tt = [r["ttft_s"] for r in ok]
        ptoks = sum(r["prompt_tokens"] or 0 for r in ok)
        cached = [r["cached_tokens"] for r in ok if r["cached_tokens"] is not None]
        wall = max(r["t_end"] for r in ok) - min(r["t0"] for r in ok)
        out.update(ttft_mean_s=sum(tt) / len(tt), ttft_p50_s=pct(tt, 50), ttft_p90_s=pct(tt, 90), ttft_max_s=max(tt),
                   prompt_tokens=ptoks, cached_frac=(sum(cached) / ptoks) if cached and ptoks else None,
                   prefill_tok_s=ptoks / max(1e-9, max(r["t0"] + r["ttft_s"] for r in ok) - min(r["t0"] for r in ok)),
                   wall_s=wall)
    if len(ok) < len(rs):
        out["first_error"] = next(r["error"] for r in rs if "error" in r)
    return out


# ---------------------------------------------------------------- prompts
class Prompts:
    """Exactly-sized prompts from source code text: a unique header per user + a shared-style
    body + a question; the same user's header + body is identical across visits."""

    def __init__(self, model, trust_remote_code):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        import importlib.util

        roots = [os.path.dirname(importlib.util.find_spec(m).origin) for m in ("json", "http", "email", "asyncio")]
        text, n = [], 0
        for root in roots:
            for f in sorted(glob.glob(os.path.join(root, "**/*.py"), recursive=True)):
                t = open(f, errors="ignore").read()
                text.append(t)
                n += len(t)
        self.ids = self.tok.encode("\n".join(text), add_special_tokens=False)
        self.q = [len(self.tok.encode(q, add_special_tokens=False)) for q in QUESTIONS]

    def make(self, tag, size, turn):
        head = f"<session {tag}>\nRepository snapshot. Read the code below.\n\n"
        body_n = max(16, size - len(self.tok.encode(head, add_special_tokens=False)) - max(self.q) - 8)
        off = random.Random(tag).randrange(max(1, len(self.ids) - body_n)) if body_n < len(self.ids) else 0
        reps = -(-(off + body_n) // len(self.ids))
        body = (self.ids * reps)[off:off + body_n]
        return head + self.tok.decode(body) + QUESTIONS[turn]


# ---------------------------------------------------------------- tests
def reset_gpu_cache(kind):
    for _ in range(60):
        if http_post("/reset_prefix_cache" if kind == "vllm" else "/flush_cache"):
            return True
        time.sleep(3)
    return False


def tier_bytes(root):
    n = 0
    for dp, _, fs in os.walk(root):
        for f in fs:
            try:
                n += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return n


def settle(server, limit_s=600):
    """Wait until background writes to the tier directory stop."""
    last, stable, t0 = -1, 0, time.time()
    while time.time() - t0 < limit_s:
        cur = tier_bytes(server.home)
        stable = stable + 1 if cur == last else 0
        if stable >= 3:
            return
        last = cur
        time.sleep(2)


def drop_page_cache(root):
    """Later reads of the tier come from disk, not RAM (works without root for our own files)."""
    for dp, _, fs in os.walk(root):
        for f in fs:
            try:
                fd = os.open(os.path.join(dp, f), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.fdatasync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
            finally:
                os.close(fd)


def returning(ctx, server, n, size, tag, reset=True):
    tags = [f"{tag}-u{u}" for u in range(n)]
    log(f"  {n} users x {size // K}K: first visit")
    cold = summarize(burst([ctx.P.make(t, size, 0) for t in tags]))
    settle(server)
    if reset:
        reset_gpu_cache(ctx.kind)
    drop_page_cache(server.home)
    log("  ... returning with a new question")
    back = summarize(burst([ctx.P.make(t, size, 1) for t in tags]))
    log(f"  first visit {fmt(cold.get('ttft_mean_s'))}, returning {fmt(back.get('ttft_mean_s'))} "
        f"({pctfmt(back.get('cached_frac'))} from cache)")
    return {"cold": cold, "return": back}


def throughput(ctx, levels, seconds):
    rows = []
    for n in levels:
        deadline = time.perf_counter() + seconds
        counter = iter(range(10**9))

        def client():
            out = []
            while time.perf_counter() < deadline:
                r = complete(ctx.P.make(f"tp-{next(counter)}-{random.random()}", 2 * K, 0), 128)
                out.append(r)
                if "error" in r:
                    time.sleep(1)
            return out

        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(n) as ex:
            rs = [r for f in [ex.submit(client) for _ in range(n)] for r in f.result()]
        wall = time.perf_counter() - t0
        ok = [r for r in rs if "error" not in r]
        dec = [r["decode_tok_s"] for r in ok if r["decode_tok_s"]]
        tt = [r["ttft_s"] for r in ok]
        row = {"users": n, "requests": len(ok), "errors": len(rs) - len(ok),
               "out_tok_s_total": sum(r["out_tokens"] for r in ok) / wall,
               "decode_tok_s_per_user": sum(dec) / len(dec) if dec else None,
               "ttft_p50_s": pct(tt, 50), "ttft_p90_s": pct(tt, 90)}
        log(f"  {n:>3} users: {row['out_tok_s_total']:.0f} tok/s total, "
            f"{fmt(row['decode_tok_s_per_user'], '')} tok/s each, p90 first token {fmt(row['ttft_p90_s'])}")
        rows.append(row)
    return rows


def fmt(v, unit="s"):
    return f"{v:.2f}{unit}" if isinstance(v, (int, float)) else "-"


def pctfmt(v):
    return f"{100 * v:.0f}%" if isinstance(v, (int, float)) else "-"


def lifeboat_stats(ctx):
    s = http_get("/api/stats", DASH_PORT, {"Authorization": "Bearer " + ctx.dash_token})
    return (s or {}).get("counts")


# ---------------------------------------------------------------- driver
class Ctx:
    pass


def plan_sizes(ctx, kv):
    """Test context and user counts for this GPU and model."""
    size = 64 * K
    while size > 4 * K and (size + 1024 > ctx.max_len or size > kv):
        size //= 2
    fit = max(1, min(16, int(0.5 * kv / size)))
    return size, fit


def run(args):
    work = os.path.abspath(os.path.expanduser(args.workdir))
    os.makedirs(work, exist_ok=True)
    ctx = Ctx()
    ctx.work, ctx.start_timeout = work, args.start_timeout
    ctx.dash_token = secrets.token_urlsafe(24)
    sysinfo = system_info(work)
    gpus = sysinfo["gpus"]
    if not gpus:
        sys.exit("no NVIDIA GPU visible (nvidia-smi)")
    log(f"{len(gpus)}x {gpus[0]['name']}, driver {gpus[0]['driver']} (CUDA {sysinfo['cuda_driver']}), "
        f"{sysinfo['cores']} cores, {sysinfo['ram_gb']:.0f} GB RAM, /dev/shm {sysinfo['shm_gb']:.0f} GB, "
        f"{sysinfo['disk_free_gb']:.0f} GB free ({sysinfo['fs']})")

    # --- the engine and model: a running server, else --engine-python / --model
    running = find_servers()
    target = None
    if args.pid:
        target = next((s for s in running if s["pid"] == args.pid), None)
        if not target:
            sys.exit(f"no vLLM / SGLang server with pid {args.pid}")
    elif running:
        target = running[0]
        if len(running) > 1:
            log(f"{len(running)} servers found, using pid {target['pid']} (choose with --pid)")
    original = None
    if target:
        launcher, model, rest = split_launch(target["kind"], target["argv"])
        if in_other_namespace(target["pid"]):
            sys.exit(f"Your {target['kind']} server (pid {target['pid']}) runs in a container. Run this inside "
                     "that container, or stop the container and run it here with --engine-python and --model.")
        ctx.kind = target["kind"]
        ctx.py = args.engine_python or engine_python_of(target)
        if not ctx.py:
            ctx.py, _ = find_engine_python(ctx.kind)
        ctx.model = args.model or model
        ctx.launcher = launcher if os.path.basename(launcher[0]).startswith("python") or launcher[-1] == "serve" \
            else [ctx.py, "-m", "vllm.entrypoints.openai.api_server" if ctx.kind == "vllm" else "sglang.launch_server"]
        ctx.base_args = rest
        try:
            raw = open(f"/proc/{target['pid']}/environ", "rb").read().decode(errors="replace").split("\0")
            ctx.orig_env = dict(x.split("=", 1) for x in raw if "=" in x)
            ctx.orig_cwd = os.readlink(f"/proc/{target['pid']}/cwd")
        except OSError:
            ctx.orig_env, ctx.orig_cwd = dict(os.environ), os.getcwd()
        ctx.engine_env = {k: v for k, v in ctx.orig_env.items()
                          if k.startswith(("CUDA_", "HF_", "HUGGING", "VLLM_", "SGLANG_", "TRANSFORMERS_", "NCCL_"))}
        unit = systemd_unit(target["pid"])
        if unit:
            sys.exit(f"Your server (pid {target['pid']}) is managed by {unit}. Stop it with "
                     f"'sudo systemctl stop {unit}', run this again with --model {ctx.model} "
                     f"--engine-python {ctx.py}, then 'sudo systemctl start {unit}'.")
        original = {"argv": target["argv"], "env": ctx.orig_env, "cwd": ctx.orig_cwd, "pid": target["pid"]}
    else:
        ctx.py, ctx.kind = (args.engine_python, None) if args.engine_python else find_engine_python()
        if ctx.py and not ctx.kind:
            ctx.kind = "vllm" if has_module(ctx.py, "vllm") else "sglang" if has_module(ctx.py, "sglang") else None
        if not ctx.py or not ctx.kind:
            sys.exit("No running vLLM / SGLang server and no Python with vllm or sglang found. "
                     "Pass --engine-python /path/to/python --model <path or HF id>.")
        if not args.model:
            sys.exit("No running server to take the model from: pass --model <path or HF id>.")
        ctx.model = args.model
        vbin = os.path.join(os.path.dirname(ctx.py), "vllm")
        ctx.launcher = [vbin, "serve"] if ctx.kind == "vllm" and os.path.exists(vbin) else \
            [ctx.py, "-m", "vllm.entrypoints.openai.api_server" if ctx.kind == "vllm" else "sglang.launch_server"]
        ctx.base_args, ctx.engine_env = list(args.extra_args), {}
    if not ctx.py:
        sys.exit("could not find the Python that runs your engine: pass --engine-python")
    drop = {"--host", "--port", "--api-key", "--served-model-name", "--kv-transfer-config", "--enable-prefix-caching",
            "--no-enable-prefix-caching", "--enable-prompt-tokens-details", "--enable-hierarchical-cache",
            "--hicache-", "--enable-cache-report", "--disable-log-requests"}
    ctx.base_args = strip_flags(ctx.base_args, drop) + list(args.extra_args if target else [])
    engine_ver = pkg_version(ctx.py, ctx.kind)
    log(f"engine: {ctx.kind} {engine_ver} ({ctx.py}); model: {ctx.model}")
    if ctx.kind == "sglang" and "--disable-radix-cache" in ctx.base_args:
        log("note: this launch disables the radix cache; SGLang's HiCache (and so Lifeboat Cache) needs it, "
            "so the cache configurations run with it on")
        ctx.base_args = [a for a in ctx.base_args if a != "--disable-radix-cache"]

    # --- we re-run inside the engine's Python (for its tokenizer); hand over now
    # compare environments, not binaries: a venv's python is a symlink to the base interpreter
    if sh([ctx.py, "-c", "import sys;print(sys.prefix)"]) != sys.prefix and not args.in_engine:
        install(ctx, args)
        own = ["--in-engine", "--engine-python", ctx.py] + (["--pid", str(target["pid"])] if target else [])
        os.execv(ctx.py, [ctx.py, os.path.abspath(__file__), *own, *sys.argv[1:]])
    if not args.in_engine:
        install(ctx, args)

    # --- the GPU must be ours for the run
    if original:
        if not (args.yes or ask(f"Your {ctx.kind} server (pid {original['pid']}, {ctx.model}) uses the GPU. Stop it "
                                f"for the test (~{'90' if args.full else '35'} min) and restart it afterwards "
                                "with the same command?")):
            sys.exit("Not stopped. Re-run with --yes, or stop the server yourself and pass --model / --engine-python.")
        stop_original(original)
    busy = [g for g in system_info(work)["gpus"] if g["used_mib"] > 2000]
    if busy:
        restore(original, work)
        sys.exit(f"GPU memory is in use by another process ({busy[0]['used_mib']} MiB on GPU {busy[0]['index']}); "
                 "free the GPU and run this again")
    try:
        return measure(ctx, args, sysinfo, engine_ver)
    finally:
        restore(original, work)


def install(ctx, args):
    """lifeboatcache (and LMCache for vLLM) into the engine's environment, keeping the engine,
    torch and everything already installed at their versions."""
    py = ctx.py
    pins = [f"{p}=={v}" for p in ("vllm", "sglang", "torch", "transformers", "lmcache", "flashinfer-python")
            if (v := pkg_version(py, p))]
    cons = os.path.join(os.path.expanduser(args.workdir), "constraints.txt")
    open(cons, "w").write("\n".join(pins) + "\n")
    want = [f"lifeboatcache{'==' + args.lifeboat_version if args.lifeboat_version else ''}"]
    if pkg_version(py, "lifeboatcache") != args.lifeboat_version or not args.lifeboat_version:
        log("installing lifeboatcache into the engine environment (engine and torch pinned)")
        pip(py, ["-U", *want, "-c", cons])
    if ctx.kind == "vllm" and not args.no_lmcache and not has_module(py, "lmcache"):
        log("trying LMCache 0.5.x (skipped if it would change any installed package)")
        if not pip(py, ["lmcache>=0.5.6,<0.6", "-c", cons], quiet_fail=True):
            log("  LMCache not installed: it needs a different torch / vLLM than this environment has")
    log(f"lifeboatcache {pkg_version(py, 'lifeboatcache')}" +
        (f", LMCache {pkg_version(py, 'lmcache')}" if has_module(py, "lmcache") else ""))


def pip(py, args, quiet_fail=False):
    base = [py, "-m", "pip", "install", "-q", "--disable-pip-version-check"]
    for extra in ([], ["--break-system-packages"]):
        r = subprocess.run(base + extra + args, capture_output=True, text=True)
        if r.returncode == 0:
            return True
        if "externally-managed" not in r.stderr:
            break
    if not quiet_fail:
        sys.exit(f"pip install failed:\n{r.stderr[-2000:]}")
    return False


def stop_original(o):
    log(f"stopping your server (pid {o['pid']})")
    try:
        os.killpg(os.getpgid(o["pid"]), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(o["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    for _ in range(90):
        if not os.path.exists(f"/proc/{o['pid']}"):
            break
        time.sleep(1)
    wait_gpu_free()


def restore(o, work):
    if not o or o.get("restored"):
        return
    o["restored"] = True
    logf = os.path.join(work, "your-server-restarted.log")
    log(f"restarting your server: {shlex.join(o['argv'])[:200]}")
    try:
        subprocess.Popen(o["argv"], env=o["env"], cwd=o["cwd"], stdout=open(logf, "a"), stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
        log(f"  your server is starting again (output: {logf})")
    except OSError as e:
        log(f"  could not restart it ({e}); start it again with:\n  {shlex.join(o['argv'])}")


def measure(ctx, args, sysinfo, engine_ver):
    random.seed(0)
    ctx.trust = "--trust-remote-code" in ctx.base_args
    ctx.max_len = int(flag(ctx.base_args, "--max-model-len", "--context-length", default=0) or 0) or model_max_len(ctx.model)
    ngpu = len(sysinfo["gpus"])
    ctx.cpu_gb = max(2, int(min(30, 0.25 * sysinfo["ram_avail_gb"], (sysinfo["shm_free_gb"] - 4) / max(1, ngpu))))
    ctx.disk_gb = max(10, int(min(500, 0.5 * sysinfo["disk_free_gb"])))
    log(f"cache sizes: {ctx.cpu_gb} GB host RAM tier, {ctx.disk_gb} GB disk tier; model context {ctx.max_len // K}K")
    ctx.P = Prompts(ctx.model, ctx.trust)
    lm = ctx.kind == "vllm" and has_module(ctx.py, "lmcache")
    configs = (["vllm", "vllm_offload", "lifeboat"] + (["lmcache", "lmcache_lifeboat"] if lm else [])
               if ctx.kind == "vllm" else ["sglang", "sglang_hicache", "lifeboat"])
    if args.only:
        configs = [c for c in configs if c in args.only.split(",")]
    budget = 300 if args.full else 120
    levels = [1, 8, 32, 64] if not args.full else [1, 8, 32, 64, 128, 256]
    res = {"tool_version": VERSION, "engine": ctx.kind, "engine_version": engine_ver, "model": ctx.model,
           "lifeboatcache": pkg_version(ctx.py, "lifeboatcache"), "lmcache": pkg_version(ctx.py, "lmcache"),
           "system": sysinfo, "args": ctx.base_args, "configs": {}}
    out = os.path.join(ctx.work, "results-" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    plan = None
    for config in configs:
        r = res["configs"][config] = {"label": LABELS[config]}
        try:
            srv = Server(ctx, config).start()
        except RuntimeError as e:
            r["skipped"] = str(e).splitlines()[0]
            r["log_tail"] = str(e)[-3000:]
            log(f"  skipped: {r['skipped']}")
            continue
        try:
            r["attached"] = srv.attached()
            if not r["attached"]:
                r["skipped"] = "the cache did not attach (see the server log); numbers would be plain vLLM"
                log(f"  {r['skipped']}")
                continue
            r["ready_s"], r["kv_tokens"] = srv.ready_s, srv.kv_tokens()
            if plan is None:
                kv = r["kv_tokens"] or 100_000
                size, fit = plan_sizes(ctx, kv)
                plan = res["plan"] = {"gpu_kv_tokens": kv, "context": size, "users_fit": fit,
                                      "users_on_gpu_at_once": kv // size}
                log(f"plan: {size // K}K-token users; {kv:,} KV tokens on the GPU = {kv // size} such users at once")
            size = plan["context"]
            if config in ("vllm", "sglang", "lifeboat"):
                log("total concurrency")
                r["throughput"] = throughput(ctx, levels, 40 if args.full else 20)
            if config in ("vllm", "sglang"):
                # without offload a returning user is recomputed: its first visit is the reference
                log("first visits (a returning user costs the same without offload)")
                r["returning"] = returning(ctx, srv, plan["users_fit"], size, f"{config}-fit")
                if "overflow_users" not in plan:
                    rate = r["returning"]["cold"].get("prefill_tok_s") or 5000
                    want = int(2 * plan["gpu_kv_tokens"] / size) + 1
                    plan["overflow_users"] = max(0, min(want, int(budget * rate / size), 128))
                    plan["overflow_x_gpu"] = round(plan["overflow_users"] * size / plan["gpu_kv_tokens"], 2)
                continue
            log("returning users (GPU cache reset in between)")
            r["returning"] = returning(ctx, srv, plan["users_fit"], size, f"{config}-fit")
            if plan.get("overflow_x_gpu", 0) >= 1.25:
                log(f"overflow: {plan['overflow_users']} users, {plan['overflow_x_gpu']}x the GPU's KV pool")
                r["overflow"] = returning(ctx, srv, plan["overflow_users"], size, f"{config}-ovf", reset=False)
            if config not in ("lmcache",):  # LMCache alone keeps KV in RAM only
                settle(srv)
                srv.stop()
                srv = Server(ctx, config, keep_tier=True).start()
                drop_page_cache(srv.home)
                log("after a full restart")
                r["restart"] = summarize(burst([ctx.P.make(f"{config}-fit-u0", size, 2)]))
                log(f"  returning user {fmt(r['restart'].get('ttft_mean_s'))} "
                    f"({pctfmt(r['restart'].get('cached_frac'))} from cache)")
            if config in ("lifeboat", "lmcache_lifeboat"):
                r["lifeboat_counts"] = lifeboat_stats(ctx)
        finally:
            srv.stop()
            shutil.rmtree(srv.home, ignore_errors=True)
            json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1, default=str)
    write_csv(res, out)
    tar = out + ".tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(out, arcname=os.path.basename(out))
    print_summary(res)
    log(f"results: {out}/ (CSV) and {tar}")
    return 0


def model_max_len(model):
    for p in (os.path.join(model, "config.json"),):
        try:
            c = json.load(open(p))
            c = c.get("text_config", c)
            return int(c.get("max_position_embeddings") or 32768)
        except (OSError, ValueError):
            pass
    try:
        from transformers import AutoConfig

        c = AutoConfig.from_pretrained(model)
        c = getattr(c, "text_config", c)
        return int(getattr(c, "max_position_embeddings", 32768))
    except Exception:  # noqa: BLE001
        return 32768


# ---------------------------------------------------------------- output
def write_csv(res, out):
    base = {"engine": f"{res['engine']} {res['engine_version']}", "model": res["model"],
            "gpu": res["system"]["gpus"][0]["name"], "gpus": len(res["system"]["gpus"])}
    rows = {"summary": [], "throughput": [], "returning": [], "system": []}
    plan = res.get("plan") or {}
    ref = None
    for cfg, r in res["configs"].items():
        if cfg in ("vllm", "sglang"):
            ref = (r.get("returning") or {}).get("cold", {}).get("ttft_mean_s")
    offload_ref = None
    for cfg in ("vllm_offload", "sglang_hicache"):
        offload_ref = offload_ref or ((res["configs"].get(cfg) or {}).get("returning") or {}).get(
            "return", {}).get("ttft_mean_s")
    for cfg, r in res["configs"].items():
        b = {**base, "config": r["label"]}
        for lv in r.get("throughput", []):
            rows["throughput"].append({**b, **lv})
        for test in ("returning", "overflow"):
            for ph, s in (r.get(test) or {}).items():
                rows["returning"].append({**b, "test": test, "phase": ph, "context_tokens": plan.get("context"),
                                          "users": s.get("requests"), **{k: v for k, v in s.items() if k != "requests"}})
        if r.get("restart"):
            rows["returning"].append({**b, "test": "restart", "phase": "return", "context_tokens": plan.get("context"),
                                      "users": 1, **{k: v for k, v in r["restart"].items() if k != "requests"}})
        ret = ((r.get("returning") or {}).get("return") or {}).get("ttft_mean_s")
        if cfg in ("vllm", "sglang"):
            ret = None
        tp = r.get("throughput") or []
        rows["summary"].append({
            **b, "status": "skipped: " + r["skipped"] if r.get("skipped") else "ok",
            "gpu_kv_tokens": r.get("kv_tokens"), "context_tokens": plan.get("context"),
            "users_on_gpu_at_once": plan.get("users_on_gpu_at_once"),
            "peak_out_tok_s": max((x["out_tok_s_total"] for x in tp), default=None),
            "returning_users": plan.get("users_fit"),
            "returning_ttft_s": ret if ret is not None else ref if cfg in ("vllm", "sglang") else None,
            "x_faster_than_recompute": round(ref / ret, 2) if ref and ret else None,
            "x_faster_than_engine_offload": round(offload_ref / ret, 2) if offload_ref and ret and cfg not in (
                "vllm_offload", "sglang_hicache") else None,
            "returning_cached_frac": ((r.get("returning") or {}).get("return") or {}).get("cached_frac"),
            "overflow_users": plan.get("overflow_users") if r.get("overflow") else None,
            "overflow_ttft_s": ((r.get("overflow") or {}).get("return") or {}).get("ttft_mean_s"),
            "restart_ttft_s": (r.get("restart") or {}).get("ttft_mean_s"),
            "server_start_s": r.get("ready_s")})
    s = res["system"]
    for k, v in [("engine", base["engine"]), ("model", res["model"]), ("lifeboatcache", res["lifeboatcache"]),
                 ("lmcache", res["lmcache"]), ("gpu", base["gpu"]), ("gpus", base["gpus"]),
                 ("gpu_mem_mib", s["gpus"][0]["mem_mib"]), ("driver", s["gpus"][0]["driver"]),
                 ("cuda_driver", s["cuda_driver"]), ("cpu", s["cpu"]), ("cores", s["cores"]),
                 ("ram_gb", round(s["ram_gb"])), ("shm_gb", round(s["shm_gb"])), ("disk_free_gb", round(s["disk_free_gb"])),
                 ("filesystem", s["fs"]), ("engine_args", " ".join(res["args"])), ("tool_version", VERSION)]:
        rows["system"].append({"item": k, "value": v})
    for name, rr in rows.items():
        if not rr:
            continue
        cols = list(dict.fromkeys(c for r in rr for c in r))
        with open(os.path.join(out, f"{name}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            for r in rr:
                w.writerow({c: round(v, 4) if isinstance(v, float) else v for c, v in r.items()})


def print_summary(res):
    plan = res.get("plan") or {}
    print("\n" + "=" * 78)
    print(f"Lifeboat Cache verification — {res['engine']} {res['engine_version']}, {res['model']}")
    print(f"{len(res['system']['gpus'])}x {res['system']['gpus'][0]['name']}; "
          f"{plan.get('context', 0) // K}K-token users, {plan.get('users_fit')} returning")
    print("-" * 78)
    print(f"{'configuration':<34}{'returning user, first token':>28}{'after restart':>16}")
    for cfg, r in res["configs"].items():
        if r.get("skipped"):
            print(f"{r['label']:<34}{'skipped':>28}")
            continue
        ret = r.get("returning", {}).get("cold" if cfg in ("vllm", "sglang") else "return", {})
        note = " (recompute)" if cfg in ("vllm", "sglang") else f" ({pctfmt(ret.get('cached_frac'))} cached)"
        print(f"{r['label']:<34}{fmt(ret.get('ttft_mean_s')) + note:>28}{fmt((r.get('restart') or {}).get('ttft_mean_s')):>16}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="stop a running server without asking (it is restarted after)")
    ap.add_argument("--full", action="store_true", help="longer run (~90 min): bigger overflow, more concurrency")
    ap.add_argument("--engine-python", help="the Python that runs vLLM / SGLang (found automatically)")
    ap.add_argument("--model", help="model path or HF id (default: the one your running server serves)")
    ap.add_argument("--pid", type=int, help="which running server to take the model and flags from")
    ap.add_argument("--workdir", default="~/.lifeboatcache-verify", help="results and cache tiers (local NVMe)")
    ap.add_argument("--lifeboat-version", default="", help="pin a lifeboatcache version (default: latest)")
    ap.add_argument("--no-lmcache", action="store_true", help="don't try to install LMCache")
    ap.add_argument("--only", help="comma-separated configurations to run")
    ap.add_argument("--start-timeout", type=int, default=1200, help="seconds to wait for a server to start")
    ap.add_argument("--in-engine", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("extra_args", nargs="*", help="extra engine flags after --, e.g. -- --max-model-len 65536")
    args = ap.parse_args()
    if not sys.platform.startswith("linux"):
        sys.exit("Linux with an NVIDIA GPU is required")
    sys.exit(run(args))


if __name__ == "__main__":
    main()
