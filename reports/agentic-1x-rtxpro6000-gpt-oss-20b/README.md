# gpt-oss-20b on 1 × RTX PRO 6000: vLLM alone vs LMCache vs LifeboatCache

Measured 2026-10-31 with the tools in [`../../agentic/`](../../agentic/).

| | |
|---|---|
| Server | 1 × RTX PRO 6000 Blackwell Workstation Edition (96 GB), 61 GB RAM, local NVMe |
| Software | vLLM 0.32.0, LMCache 0.5.6, LifeboatCache 0.1.23 |
| Model | openai/gpt-oss-20b (MXFP4), 128,000-token context, `gpu-memory-utilization` 0.80 |
| Workload | 128 agents, 25 rounds per agent, up to 512 output tokens per round; fresh server and empty tiers per run |
| Host DRAM tier | 24 GB for both caches; LifeboatCache NVMe tier 70 GB |

| 128 agents | Mean TTFT | P99 TTFT | Output tok/s | Prompt tokens from cache | Output tokens per joule (GPU) | Completed |
|---|---|---|---|---|---|---|
| vLLM alone | 0.49 s | 2.5 s | 1,117 | 80% | 1.90 | 3,199 / 3,200 |
| vLLM + LMCache | 0.64 s | 2.8 s | 1,144 | 82% | 1.95 | 3,200 / 3,200 |
| vLLM + LifeboatCache | **0.44 s** | **2.4 s** | **1,425** | **89%** | **2.55** | 3,200 / 3,200 |

`cells/`: per-request CSVs and per-run summaries.
