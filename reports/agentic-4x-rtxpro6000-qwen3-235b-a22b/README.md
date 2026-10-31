# Agentic workload on 4 × RTX PRO 6000: vLLM alone vs LMCache vs LifeboatCache

Measured 2026-10-30 and 2026-10-31. Real agent sessions (`sammshen/lmcache-agentic-traces`)
replayed turn by turn on a four-GPU model, with and without KV-cache offload.
Raw per-request and per-cell results are in the subdirectories; the customer deck is
`LifeboatCache_vLLM_Agentic_4xRTXPRO6000.pptx`. Tools: [`../../agentic/`](../../agentic/).

## Setup

| | |
|---|---|
| Server | AWS g7e.24xlarge: 4 × NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB), Intel Xeon Platinum 8559C, 1 TB RAM, 2 × 3.5 TB local NVMe (RAID 0) |
| Software | vLLM 0.32.0, LMCache 0.5.6, LifeboatCache 0.1.23, torch 2.13.0, NVIDIA driver 595 |
| Model | Qwen/Qwen3-235B-A22B-FP8, tensor parallel 4, KV cache bf16, `gpu-memory-utilization` 0.80 (387,536 tokens of GPU KV) |
| Workload | 10 rounds per agent, 256 output tokens per request (end-of-sequence ignored), 40,960-token context |
| Per cell | fresh server, empty caches and tiers |
| DRAM | the server's host-RAM KV tier, the same for both caches (vLLM `cpu_bytes_to_use` and LMCache `--l1-size-gb` are both server-wide): 64 / 128 / 256 / 512 GB = 16 / 32 / 64 / 128 GB per GPU |
| Configurations | `gpu_only` = vLLM alone; `lmcache_dram` = vLLM + LMCache (DRAM); `lifeboat` = vLLM native offloading (DRAM) + LifeboatCache on local NVMe |

DeepGEMM was disabled (`VLLM_USE_DEEP_GEMM=0`) in every configuration: its JIT did not compile
with the operating system's compiler. A small, identical number of requests fails in every
configuration (sessions longer than the context).

## Results

Each cell: mean time to first token / output tokens per second.

### Equal DRAM (128 GB)

| Agents | vLLM alone | vLLM + LMCache | vLLM + LifeboatCache |
|---|---|---|---|
| 32 | 0.9 s / 204 | 1.1 s / 206 | **0.9 s / 204** |
| 64 | 26.7 s / 94 | 17.1 s / 148 | **9.1 s / 219** |
| 128 | 137.5 s / 70 | 113.5 s / 87 | **42.7 s / 208** (2 TB NVMe tier) |

Share of prompt tokens served from cache at 64 agents: vLLM alone 12%, LMCache 53%,
LifeboatCache 75%. p99 time to first token: 71 s, 52 s, 26 s.

### Energy, latency over the session, capacity within a latency target (128 GB)

| | vLLM alone | vLLM + LMCache | vLLM + LifeboatCache |
|---|---|---|---|
| Output tokens per joule (GPU), 64 / 128 agents | 0.071 / 0.053 | 0.113 / 0.065 | **0.171 / 0.159** |
| Mean TTFT, round 1 → round 10, 64 agents | 2.1 → 29.5 s | 2.0 → 20.2 s | **2.1 → 8.9 s** |
| Mean TTFT, round 1 → round 10, 128 agents | 10.7 → 182.9 s | 10.8 → 154.2 s | **10.8 → 45.4 s** |
| Most agents with p99 TTFT within 30 s (of 32 / 64 / 128 tested) | 32 | 32 | **64** |
| Wall-clock for the same 640 / 1,280 requests | 12.9 / 34.8 min | 8.1 / 28.3 min | **5.5 / 11.8 min** |

GPU energy from `nvidia-smi` (CPU package energy was not readable on this cloud server).
LifeboatCache at 128 agents with the 2 TB NVMe tier.

### Long agent sessions: 256 and 512 agents (2026-10-31)

25 rounds per agent, up to 512 output tokens per round, 40,192-token context; LifeboatCache with
256 GB of DRAM and a 3 TB NVMe tier (`LB_CAPACITY_GB=3000`).

| Agents | DRAM | Mean TTFT | P99 TTFT | Output tok/s | Cached | Completed |
|---|---|---|---|---|---|---|
| 256 | 256 GB | 158 s | 242 s | **222.6** | 89% | 6,391 / 6,400 |
| 256 | 64 GB | 178 s | 254 s | 199.0 | 85% | 6,392 / 6,400 |
| 512 | 256 GB | 598 s | 1,365 s | **122.4** | 90% | 12,789 / 12,800 |

More DRAM: +12% throughput from 64 to 256 GB at 256 agents. Requests that did not complete
exceeded the context. Data: `long-sessions/cells/`.

### Host DRAM sweep (64 agents)

| DRAM | vLLM + LMCache | vLLM + LifeboatCache |
|---|---|---|
| 64 GB | 24.8 s / 104 | 9.2 s / 219 |
| 128 GB | 17.1 s / 148 | 9.1 s / 219 |
| 256 GB | 10.0 s / 217 | 9.1 s / 220 |
| 512 GB | 10.0 s / 219 | 16.1 s / 123 ¹ |

vLLM alone at 64 agents: 26.7 s / 94. At 64 agents both caches reach the workload's reuse ceiling
(about 75% of prompt tokens from cache) once enough KV is kept; LifeboatCache keeps on local NVMe
what does not fit in DRAM.

¹ At 512 GB (a 550 GB region, over half of the server's RAM) vLLM 0.32 could not pin its CPU
offload tier on three of the four workers during startup and copied through pageable memory, so
LifeboatCache's native mode ran slower. The same region pinned without error from four processes
in a standalone test on the idle machine; the cause during startup is being investigated. 256 GB
pinned and ran at full speed.

### 128 agents and the NVMe tier size

| LifeboatCache NVMe tier | DRAM | Result | Cached |
|---|---|---|---|
| 250 GB (the launcher's default) | 64 GB | 72.8 s / 131 | 56% |
| 250 GB | 128 GB | 80.3 s / 119 | 50% |
| 2 TB | 128 GB | **42.7 s / 208** | 79% |

128 agents' KV on this model is far more than 250 GB; give the tier the space your NVMe has
(`LB_CAPACITY_GB` in `agentic/serve.sh`, `fallback_local.capacity_gb` in `lifeboat.yaml`).

### Repeatability (64 agents, 128 GB)

| | Run 1 | Run 2 |
|---|---|---|
| vLLM + LMCache | 17.1 s / 148 | 17.0 s / 146 |
| vLLM + LifeboatCache | 9.1 s / 219 | 9.2 s / 219 |

## Files

| Path | Contents |
|---|---|
| `q235/` | the grid: `sweep_summary.csv` (one row per cell), `cells/*-summary.json`, `cells/*-requests.csv` (every request: TTFT, end-to-end time, prompt / cached / output tokens) |
| `q235-r2/` | second run of the 64-agent, 128 GB cells |
| `q235-cap2000/` | LifeboatCache, 128 agents, 128 GB, 2 TB NVMe tier |
| `long-sessions/` | LifeboatCache, 25-round sessions at 256 and 512 agents |

Less DRAM, less performance; more DRAM, better performance — achieved just with LifeboatCache on
top of vLLM, SGLang or Lifeboat.
