# Agentic benchmark tools

The workload of Intel's published KV-cache offload study: real agent sessions (SWE-bench
Verified, GAIA, WildClawBench) replayed turn by turn against an OpenAI-compatible server, across
configurations, host-DRAM sizes and numbers of concurrent agents. These are the tools behind
[`../reports/`](../reports/).

| File | What it does |
|---|---|
| `agentic_prep.py` | Downloads `sammshen/lmcache-agentic-traces` (2.4 GB parquet) and writes `sessions.jsonl.gz` (~19 MB): what each turn adds |
| `agentic_replay.py` | N agents at once; each replays sessions turn by turn. Mean / P99 time to first token, sustained output tokens/s, prefix-cache hit rate, energy per token. Writes `<label>-summary.json` and `<label>-requests.csv` |
| `intel_sweep.py` | The grid: configuration × DRAM GB × agents, a fresh server with empty caches per cell. Writes `intel_summary.csv`, `intel_sla.csv`, `intel_dram.csv`. Resumable |
| `serve.sh` | Starts vLLM in one configuration and waits until it is ready (used by `intel_sweep.py`) |

## Run it

Linux with NVIDIA GPUs, Python 3.12, local NVMe for the disk tiers, and `/dev/shm` larger than
the biggest DRAM size you test (vLLM keeps its CPU tier there).

```bash
mkdir -p ~/lifeboat-eval && cd ~/lifeboat-eval
uv venv -p 3.12 .venv
uv pip install --python .venv/bin/python "vllm==0.32.0" "lmcache==0.5.6" lifeboatcache aiohttp pyarrow huggingface_hub
.venv/bin/vllm download-kernels            # prebuilt FlashInfer kernels, if the host has no CUDA toolkit
.venv/bin/python /path/to/agentic/agentic_prep.py --out sessions.jsonl.gz

export EVAL_ROOT=~/lifeboat-eval           # .venv, keys, logs and the disk tiers live here
# 4 GPUs, the run in reports/agentic-4x-rtxpro6000-qwen3-235b-a22b
.venv/bin/python /path/to/agentic/intel_sweep.py --model-dir /models/Qwen3-235B-A22B-FP8 --tp 4 \
    --sessions sessions.jsonl.gz --out results/q235 --rounds 10 --max-out 256 --context 40960 \
    --configs gpu_only,lmcache_dram,lifeboat --dram 64,128,256 --users 32,64,128
```

`serve.sh` writes an API key (`vllm-api-keys`) and a dashboard token (`dash-token`) into
`EVAL_ROOT` on first use. Settings come from the environment: `CPU_GB` (set per cell by the
sweep), `LB_CAPACITY_GB` (LifeboatCache's NVMe tier, default 250 — size it to your disk for many
agents on a large model), `NATIVE_BLOCK_TOKENS`, `GPU_UTIL`, `MAX_SEQS`, `KV_DTYPE`, `SERVE_EXTRA`
(extra `vllm serve` flags). Configurations: `gpu_only`, `lmcache_dram`, `lmcache_dram_nvme`,
`lifeboat`.
