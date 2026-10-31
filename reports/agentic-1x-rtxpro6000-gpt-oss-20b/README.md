# gpt-oss-20b on 1 × RTX PRO 6000: vLLM alone vs LMCache vs LifeboatCache

Measured 2026-10-31 with the tools in [`../../agentic/`](../../agentic/).

| | |
|---|---|
| Server | 1 × RTX PRO 6000 Blackwell Workstation Edition (96 GB), 61 GB RAM, local NVMe |
| Software | vLLM 0.32.0, LMCache 0.5.6, LifeboatCache 0.1.23 (128 agents) and 0.1.24 (256 agents) |
| Model | openai/gpt-oss-20b (MXFP4), 128,000-token context, `gpu-memory-utilization` 0.80 |
| Workload | 128 and 256 agents, 25 rounds per agent, up to 512 output tokens per round; fresh server and empty tiers per run |
| Host DRAM tier | 24 GB for both caches; LifeboatCache NVMe tier 70 GB |

| 128 agents | Mean TTFT | P99 TTFT | Output tok/s | Prompt tokens from cache | Output tokens per joule (GPU) | Completed |
|---|---|---|---|---|---|---|
| vLLM alone | 0.49 s | 2.5 s | 1,117 | 80% | 1.90 | 3,199 / 3,200 |
| vLLM + LMCache | 0.64 s | 2.8 s | 1,144 | 82% | 1.95 | 3,200 / 3,200 |
| vLLM + LifeboatCache | **0.44 s** | **2.4 s** | **1,425** | **89%** | **2.55** | 3,200 / 3,200 |

| 256 agents | Mean TTFT | P99 TTFT | Output tok/s | Prompt tokens from cache | Output tokens per joule (GPU) | Completed |
|---|---|---|---|---|---|---|
| vLLM alone | 70.7 s | 115 s | 342 | 7% | 0.57 | 6,400 / 6,400 |
| vLLM + LMCache | 68.7 s | 112 s | 349 | 9% | 0.58 | 6,400 / 6,400 |
| vLLM + LifeboatCache 0.1.24 | **54.4 s** | 162 s | 349 | **40%** | **0.74** | 6,399 / 6,400 |

On sliding-window models LifeboatCache 0.1.24 turns vLLM's async scheduling off (with it on, vLLM
0.32's offload copies raced the model under a full CPU tier and crashed the engine); at 256 agents
that keeps throughput level with vLLM alone, and the cache shows as lower TTFT and energy per token.

`cells/`: per-request CSVs and per-run summaries.
