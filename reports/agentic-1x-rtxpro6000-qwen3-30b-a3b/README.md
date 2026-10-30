# Agentic workload on 1 × RTX PRO 6000: vLLM alone vs LMCache vs LifeboatCache

Measured 2026-10-30, same workload and tools as the
[4-GPU run](../agentic-4x-rtxpro6000-qwen3-235b-a22b/). Customer deck:
`LifeboatCache_vLLM_Agentic_RTXPRO6000.pptx`.

| | |
|---|---|
| Server | rented cloud GPU server: 1 × RTX PRO 6000 Blackwell Server Edition (96 GB), AMD EPYC 9554, 1.5 TB RAM, local NVMe (`/dev/shm` capped at 88 GB, so no 128 GB cells) |
| Software | vLLM 0.32.0, LMCache 0.5.6, LifeboatCache 0.1.23, torch 2.13.0 |
| Model | Qwen/Qwen3-30B-A3B, KV cache bf16, `gpu-memory-utilization` 0.80 |
| Workload | 10 rounds per agent, 256 output tokens per request, 40,960-token context; fresh server and empty tiers per cell |

Mean time to first token / output tokens per second at 64 agents:

| DRAM | vLLM alone | vLLM + LMCache | vLLM + LifeboatCache 0.1.23 |
|---|---|---|---|
| 16 GB | 25.4 s / 180 | 25.2 s / 185 | **13.5 s / 310** |
| 32 GB | 25.4 s / 180 | 21.8 s / 211 (run 2: 22.2 / 208) | **13.0 s / 323** (run 2: 12.7 / 325) |
| 64 GB | 25.4 s / 180 | 16.6 s / 268 | **12.8 s / 323** |

At 16 agents all three configurations are within noise (0.27–0.33 s, 293–296 tok/s). 2 / 3 / 6
requests fail at 16 / 32 / 64 agents in every configuration (sessions longer than the context).

| Path | Contents |
|---|---|
| `r1/` | vLLM alone and LMCache at 32 GB, 16 / 32 / 64 agents |
| `final/` | LifeboatCache 0.1.23 at 16 / 32 / 64 GB; LMCache at 16 / 64 GB |
| `r2/` | second run at 32 GB, 32 and 64 agents |

`r1/` also holds LifeboatCache 0.1.22 cells (16-token offload blocks), superseded by 0.1.23,
which offloads 256-token blocks by default.
