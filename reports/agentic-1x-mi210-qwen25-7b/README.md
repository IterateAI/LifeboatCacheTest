# Qwen2.5-7B on 1 × AMD Instinct MI210: vLLM, SGLang and Lifeboat, with and without LifeboatCache

Measured 2026-11-01 to 11-03 with the tools in [`../../agentic/`](../../agentic/).

| | |
|---|---|
| Server | 1 × AMD Instinct MI210 (64 GB, gfx90a), ROCm 7.2, AMD EPYC 9555P, 250 GB RAM, local NVMe |
| Software | vLLM 0.32.0 (`vllm/vllm-openai-rocm:v0.32.0`), SGLang 0.5.19 and Lifeboat 2.2.62; LMCache 0.5.3 rebuilt for gfx90a; LifeboatCache 0.1.24 / 0.1.25, and 0.1.28 for LMCache + LifeboatCache |
| Model | Qwen/Qwen2.5-7B-Instruct (bf16), 32,768-token context, `gpu-memory-utilization` 0.80 |
| Workload | 128 and 256 agents, 10 rounds per agent, up to 512 output tokens per round; fresh server and empty tiers per run |
| Host DRAM tier | 64 GB for every offload configuration; NVMe on the same host |

## vLLM: every KV offload option

| vLLM with | 128 agents: mean TTFT / P99 / tok/s / cached | 256 agents: mean TTFT / P99 / tok/s / cached |
|---|---|---|
| nothing (GPU prefix cache only) | 260.0 s / 473.9 s / 45 / 2% | 698.1 s / 1,110.7 s / 41 / 0% |
| vLLM's own offloading (CPU + filesystem tier) | 55.3 s / 94.9 s / 148 / 80% | 172.7 s / 363.7 s / 152 / 83% |
| LMCache, DRAM | 158.7 s / 373.8 s / 66 / 29% | 636.9 s / 977.0 s / 45 / 0% |
| LMCache, DRAM + NVMe | 65.3 s / 123.0 s / 128 / 73% | 350.1 s / 596.7 s / 78 / 49% |
| LMCache + LifeboatCache as its storage tier | 68.4 s / 123.7 s / 123 / 72% | 325.7 s / 615.8 s / 85 / 54% |
| **LifeboatCache** (native mode) | **55.0 s / 97.7 s / 147 / 79%** | **179.9 s / 378.1 s / 145 / 83%** |
| LifeboatCache, OSCAR + EpiCache | 55.6 s / 102.0 s / 147 / 79% | 182.2 s / 371.5 s / 143 / 82% |

vLLM + LifeboatCache: 3.2x the output throughput of vLLM alone at 128 agents, 3.6x at 256. vLLM's
own CPU + filesystem offloading is level with it on this model (within 1% at 128, 5% ahead at 256).
As LMCache's storage tier, LifeboatCache is 4% behind LMCache's own NVMe tier at 128 agents and 9%
ahead at 256.

## SGLang and Lifeboat

| Engine, agents | No offload | + LifeboatCache | + LifeboatCache, OSCAR + EpiCache |
|---|---|---|---|
| SGLang 0.5.19, 128 | 93.3 s / 71 tok/s / 9% | **84.9 s / 84 / 28%** | 84.9 s / 84 / 28% |
| Lifeboat 2.2.62, 128 | 93.2 s / 72 / 9% | **84.2 s / 86 / 28%** | 84.5 s / 86 / 28% |
| SGLang 0.5.19, 256 | 410.5 s / 58 / 1% | 402.6 s / 60 / 2% | 412.5 s / 58 / 2% |
| Lifeboat 2.2.62, 256 | 410.6 s / 58 / 1% | 404.3 s / 59 / 2% | 410.5 s / 59 / 2% |

The 19–20% gain at 128 agents comes from SGLang's host-DRAM tier. SGLang 0.5.19 writes pages to a
storage tier but reads almost none back (with LifeboatCache or SGLang's own file backend), so at 256
agents, once sessions outgrow DRAM, there is no gain. SGLang needs `--attention-backend triton` on
the MI210; Lifeboat runs with what its control plane picks (Triton attention, torch.compile).

## Notes

- **OSCAR and EpiCache:** OSCAR halved the bytes stored on SGLang / Lifeboat at the same speed (vLLM
  native mode stores raw bytes). `cap40/`: vLLM + LifeboatCache with a 40 GB NVMe tier at 128 agents,
  so eviction runs — LRU 76.1 s / 110 tok/s, OSCAR + EpiCache 77.0 s / 109 tok/s, 66% cached for both.
- **LMCache on an MI210:** the LMCache 0.5.3 in vLLM's ROCm image has GPU kernels for gfx942 / gfx950
  only; on gfx90a its KV transfer crashes and vLLM keeps serving with no cache. Rebuild it in the
  image: `PYTORCH_ROCM_ARCH=gfx90a pip install --no-build-isolation --no-binary lmcache lmcache==0.5.3`.
  LifeboatCache 0.1.27+ refuses to start in LMCache mode on a build without kernels for the GPU.
- 14 of 1,280 (128 agents) and 14 of 2,560 (256) requests fail in every cell (16 with vLLM's own
  offloading): prompts over the 32,768-token context.

`cells/`: per-request CSVs and per-run summaries (`<engine>-<config>-dram<GB>-u<agents>`; `product` is
Lifeboat, `radix` / `gpu_only` no offload, `lifeboat` LifeboatCache, `lifeboat_oe` with OSCAR + EpiCache).
