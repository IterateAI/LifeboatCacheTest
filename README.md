# Lifeboat Cache verification

Check what [Lifeboat Cache](https://pypi.org/project/lifeboatcache/) does for **your** model on
**your** GPUs, with the vLLM or SGLang you already run. One line:

```bash
curl -fsSL https://raw.githubusercontent.com/IterateAI/LifeboatCacheTest/main/run.sh | bash
```

It finds your running vLLM / SGLang server, takes its model and flags, asks before stopping it, runs
the tests, starts your server again with the same command, and prints a summary. The results are
written as CSV files.

To skip the question (for example over a non-interactive SSH session), or to choose options:

```bash
curl -fsSL https://raw.githubusercontent.com/IterateAI/LifeboatCacheTest/main/run.sh | bash -s -- --yes
curl -fsSL https://raw.githubusercontent.com/IterateAI/LifeboatCacheTest/main/run.sh | bash -s -- --full
```

## What it measures

Each test runs on your engine three ways — without offload, with the engine's own offload, and
with Lifeboat Cache:

| Engine | Configurations |
|---|---|
| vLLM on NVIDIA | no offload · vLLM native offload (CPU + disk) · Lifeboat Cache · LMCache · LMCache + Lifeboat Cache |
| SGLang on NVIDIA | no offload (radix cache) · SGLang HiCache with its file backend · Lifeboat Cache |
| vLLM on a Mac (CPU backend) | no offload · Lifeboat Cache (vLLM's own offload and LMCache need a GPU) |
| SGLang on a Mac (MLX backend) | no offload · Lifeboat Cache (the MLX runtime has no HiCache) |

| Test | What it shows |
|---|---|
| Capacity | How much KV fits on your GPU, and how many users of the test length fit at once |
| Concurrency | Total output tokens/s, tokens/s per user and time to first token at 1 to 64 users (to 256 with `--full`) |
| Returning users | Users come back with a new question after their KV left the GPU: time to first token and the share of the prompt served from cache |
| Overflow | About twice as many users as the GPU holds, so the GPU really evicts them |
| Restart | A returning user after the server restarts |

The prompt length (up to 64K tokens), the number of users and the cache sizes are picked from your
model's context length, the GPU's KV space, RAM, `/dev/shm` and free disk. A quick run takes about
30 minutes; `--full` about 90.

## What it changes on your machine

- **Python packages:** it installs `lifeboatcache` into the Python environment that runs your
  engine. On vLLM it also tries LMCache 0.5.x, and skips it if installing would change anything.
  Your engine, torch and transformers are pinned, so their versions don't change.
- **Your server:** it's stopped only with your consent (or `--yes`) and started again at the end
  with the same command, environment and working directory, also if the test fails. Its output
  then goes to `~/.lifeboatcache-verify/your-server-restarted.log`.
- **Files:** everything else stays in `~/.lifeboatcache-verify` (results, logs, the temporary
  cache tiers, which are deleted after each test). Put it on local NVMe with `--workdir` if your home
  directory isn't.

Nothing is sent anywhere: no telemetry, and the results stay on your machine.

## Results

`~/.lifeboatcache-verify/results-<time>/` and the same as a `.tar.gz`:

| File | Contents |
|---|---|
| `summary.csv` | One row per configuration: returning-user time to first token, speed-up over recompute and over the engine's own offload, overflow, restart, peak throughput |
| `returning.csv` | Every returning / overflow / restart phase: mean, p50, p90 and max time to first token, cached share, prefill rate, errors |
| `throughput.csv` | Every concurrency level per configuration |
| `system.csv` | GPU, driver, CPU, RAM, disk, engine and package versions, the engine flags used |
| `results.json` | Everything, including server log tails for configurations that didn't start |

## Requirements

- Linux with NVIDIA GPUs and `nvidia-smi`, or macOS on Apple silicon; Python 3 and `curl`.
- **vLLM 0.32.x**, or **SGLang 0.5.19 or newer**, installed, with the model you want to test.
  Other vLLM versions run, but Lifeboat Cache's native mode targets 0.32; a configuration that
  doesn't attach is reported as skipped rather than counted.
- Free disk for the cache tiers (the tool uses up to half the free space, at most 500 GB).

Lifeboat Cache (0.1.13+) runs with vLLM and SGLang on NVIDIA GPUs (measured), on AMD and Intel
GPUs (supported by the engines, not measured yet), with vLLM's CPU backend (Apple silicon and CPU
servers) and with SGLang's MLX backend on Apple silicon (full-attention models).

### On a Mac

The same line works on Apple silicon with vLLM's CPU build or SGLang's MLX runtime. Unified
memory holds the model, the engine's KV pool and its working buffers together, so the tests stay
small there: prompts up to 16K tokens, 2 to 3 returning users, no overflow test, and SGLang's test
servers start with `--mem-fraction-static 0.3` unless your flags set it (larger settings ran a
64 GB M4 Max out of Metal memory with several long users). Measured with this line on an M4 Max,
Qwen3-0.6B (vLLM with lifeboatcache 0.1.14, SGLang with 0.1.15):

| Engine | Returning users | Without offload | With Lifeboat Cache | After a restart |
|---|---|---|---|---|
| vLLM 0.32 (CPU) | 3 x 8K | 49.8 s | 1.11 s (99% cached, 45x) | 0.55 s |
| SGLang 0.5.22 (MLX) | 2 x 16K | 7.9 s | 1.48 s (100% cached, 5.4x) | 0.88 s |

## Options

| Option | Meaning |
|---|---|
| `--yes` | Stop the running server without asking (it is started again afterwards) |
| `--full` | Longer run: bigger overflow, concurrency up to 256 |
| `--model PATH_OR_ID` | Model to test when no server is running (or a different one) |
| `--engine-python PATH` | The Python that runs vLLM / SGLang, if it isn't found automatically |
| `--pid PID` | Which running server to take the model and flags from, if there are several |
| `--workdir DIR` | Where results and temporary cache tiers go (default `~/.lifeboatcache-verify`) |
| `--lifeboat-version X.Y.Z` | Test a specific lifeboatcache version (default: latest) |
| `--no-lmcache` | Don't try to install LMCache |
| `--only a,b` | Run only these configurations (`vllm`, `vllm_offload`, `lifeboat`, `lmcache`, `lmcache_lifeboat`, `sglang`, `sglang_hicache`) |
| `-- FLAGS` | Extra engine flags, e.g. `-- --max-model-len 65536 --tensor-parallel-size 2` |

Without a running server: `... | bash -s -- --model Qwen/Qwen3-30B-A3B` (the model must be
downloaded already, or downloadable by your engine).

## Not supported

- **A server that renames its process** (SGLang does) hides its environment: it is restarted with
  the Python it runs in, its command line and this shell's environment (plus `SGLANG_USE_MLX=1` on a
  Mac), and the tool says so before stopping it.
- **A server inside a container** can't be stopped and restarted safely from outside. Run the line
  inside that container, or stop the container and use `--model` / `--engine-python` here.
- **A server run by systemd:** the tool prints the `systemctl` commands to stop it and start it again.
- **Hybrid (Mamba / linear-attention) models on SGLang:** restores after a restart are unreliable
  at the moment; on vLLM they work.

Questions or results to share: open an issue in this repository.
