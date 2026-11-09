# Shared KV pool over RDMA — Soft-RoCE on one host (2026-11-09)

**What this shows:** LifeboatCache's shared KV memory pool (`lifeboat-cache-pool` + the `nixl`
tier, 0.1.29) moving KV blocks with real RDMA verbs: UCX picked `rc_verbs/rxe0:1` for every
transfer, with TCP and shared memory excluded (`UCX_TLS=rc_v,self`), and every block read back
byte-identical.

**What it does not show:** RDMA *hardware* speed. Soft-RoCE (the kernel's `rdma_rxe`) runs the
RDMA protocol in software on the CPU; on this host it tops out near 12 Gb/s. A 400 Gb/s
InfiniBand or RoCE NIC does the same work in hardware. These numbers are not a forecast for
one — see `docs/RDMA_VALIDATION.md` for the hardware run.

## Setup

- Host: the AMD MI210 box (128 cores, Rocky Linux 9, kernel 5.14). GPUs not used.
- Soft-RoCE device `rxe0` on `eno1`; pool and benchmark client on the same host, so traffic
  loops back in the kernel and never reaches the 1 Gb/s wire.
- Containers: `python:3.12-slim` + `nixl[cu12]==1.5.2` (UCX 1.23) + Debian `rdma-core` 56.1.
  NIXL's wheel bundles its own renamed `libibverbs`, which cannot load the system's Soft-RoCE
  provider ("no userspace device-specific driver found for rxe0"); the image points it at the
  system library (same 1.14 ABI) — see "Reproduce".
- Pool: 8 GB, `--transport nixl`. Client: `benchmarks/pool_bench.py`, 4 threads, 2 GB per run
  (4,096 x 0.5 MB, 1,024 x 2 MB, 256 x 8 MB blocks), 3 runs per cell; medians below.

## Results

Raw RDMA ceiling of this Soft-RoCE device (`perftest`, 2 MB messages, same host):
`ib_write_bw` 12.23 Gb/s (1.53 GB/s), `ib_read_bw` 9.77 Gb/s (1.22 GB/s).

| Block | Pool write over RDMA | Pool read over RDMA | Write p50 / p99 | Read p50 / p99 |
|---|---|---|---|---|
| 0.5 MB | 0.99 GB/s | 0.65 GB/s | 1.6 / 5.7 ms | 3.2 / 4.2 ms |
| 2 MB | 1.03 GB/s | 0.66 GB/s | 5.8 / 23.6 ms | 12.7 / 15.0 ms |
| 8 MB | 1.05 GB/s | 0.66 GB/s | 23.1 / 93.8 ms | 50.8 / 66.1 ms |

- The pool reaches about **two thirds of the device's raw write rate and about half its raw read
  rate**, at every block size. 1, 4, 8 and 16 client threads give the same GB/s (2 MB blocks:
  1.00–1.06 write, 0.63–0.66 read): the software RDMA engine is the limit, not the pool.
- **0 mismatched blocks** in 21 runs (12 over RDMA).
- The pool's TCP data path on the same host ran 6–22 GB/s. That is a loopback memory copy, not a
  network, and is not comparable with Soft-RoCE or with RDMA hardware.

Not measured here: RDMA hardware throughput, engine throughput with a shared pool, or anything
between two hosts.

## Reproduce

```bash
# Soft-RoCE (root; here via a privileged container), then remove it afterwards
modprobe rdma_rxe && rdma link add rxe0 type rxe netdev eno1
#   ... runs ...
rdma link delete rxe0 && modprobe -r rdma_rxe

# image: python:3.12-slim, pip install "nixl[cu12]==1.5.2" numpy pytest pyyaml lifeboatcache,
# apt install rdma-core ibverbs-providers ibverbs-utils iproute2 kmod perftest, then
ln -sfn /usr/lib/x86_64-linux-gnu/libibverbs /usr/lib64/libibverbs
for f in $SITE/nixl_cu1*.libs/libibverbs-*.so*; do ln -sf /usr/lib/x86_64-linux-gnu/libibverbs.so.1 "$f"; done

# containers: --network host --device /dev/infiniband --ulimit memlock=-1
#             -e UCX_TLS=rc_v,self -e UCX_NET_DEVICES=rxe0:1
```

`raw/run.sh` is the sweep; `raw/*.json` are the per-run results (`thr_*` the thread sweep).
