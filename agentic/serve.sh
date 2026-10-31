#!/usr/bin/env bash
# serve.sh <moe|hybrid|custom> <vllm|lmcache|lmcache_lb|lb_native|vllm_fs|lmcache_fs> [fresh|keep]
# Start the benchmark vLLM server in one of the four cache configurations and wait until ready.
#   vllm        vLLM prefix cache only (no LMCache, no LifeboatCache)
#   lmcache     vLLM + LMCache MP server, 20 GB host-RAM L1, no LifeboatCache
#   lmcache_lb  vLLM + LMCache (20 GB L1) + LifeboatCache as LMCache's L2 (local NVMe)
#   lb_native   vLLM native offloading (20 GB CPU tier) + LifeboatCache tier (local NVMe), no LMCache
#   vllm_fs     baseline: vLLM native offloading (20 GB CPU tier) + vLLM's own fs tier (local NVMe)
#   lmcache_fs  baseline: vLLM + LMCache (20 GB L1) + LMCache's own fs L2 adapter (local NVMe)
# Host layout and sizes come from the environment:
#   EVAL_ROOT        dir with .venv, vllm-api-keys and matrix/ (local NVMe: the disk tiers live here)
#   MOE_DIR, HYBRID_DIR, DASH_TOKEN_FILE
#   CPU_GB (20)  host-RAM tier;  MAX_SEQS (64);  GPU_UTIL (0.90);  MAX_LEN (262144);  PORT (18080)
#   INSTANCE  one of several servers at once (own pid file, home and Lifeboat ports; GPU from
#             CUDA_VISIBLE_DEVICES);  KV_DIR  disk tier directory, shared by every instance given
#             the same one;  LB_SHARED=1  Lifeboat's storage.shared_lookup (needed when shared)
#   VLLM_BIN  the vllm executable (default .venv/bin/vllm): an existing install is used as is
#   custom model: MODEL_DIR, TP (1), KV_DTYPE (fp8), CHUNK (256), SERVE_EXTRA (extra vllm args),
#             NATIVE_BLOCK_TOKENS (native mode's offload block, default vLLM's)
set -u
LB=${LB_ROOT:-$HOME/lifeboat}
cd "${EVAL_ROOT:-$HOME/lifeboat-eval}"
CPU_GB=${CPU_GB:-20} PORT=${PORT:-18080}
I=${INSTANCE:-} DASH=$((9400 + 10 * ${INSTANCE:-0})) PROM=$((9402 + 10 * ${INSTANCE:-0}))
MODEL=$1 MODE=$2 KEEP=${3:-fresh}
M=$PWD/matrix
PIDF=$M/server$I.pid
mkdir -p "$M/logs"

# stop the previous server (its whole process group: API server, engine core, LMCache server)
if [ -f "$PIDF" ]; then
  P=$(cat "$PIDF")
  kill -TERM -- "-$P" 2>/dev/null
  for i in $(seq 1 60); do kill -0 "$P" 2>/dev/null || break; sleep 2; done
  kill -KILL -- "-$P" 2>/dev/null
  rm -f "$PIDF"
fi
for i in $(seq 1 60); do ss -ltn | grep -q -E ":($PORT|6555|$DASH|$PROM) " || break; sleep 2; done
for i in $(seq 1 90); do m=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$m" -lt 3000 ] && break; sleep 2; done
for f in /dev/shm/vllm_offload_*.mmap; do [ -e "$f" ] && [ -O "$f" ] && ! fuser "$f" >/dev/null 2>&1 && rm -f "$f"; done
[ "$MODE" = stop ] && exit 0

case $MODEL in
  moe)    DIR=${MOE_DIR:-$LB/models/Qwen-30b-a3}; CHUNK=256
          # Qwen3-30B-A3B is trained for 32K (128K with YaRN x4); x8 YaRN reaches 256K for this
          # performance test (output quality beyond 128K is not validated by the model vendor)
          # Triton MoE kernels: vLLM's default FlashInfer CUTLASS MoE kernel aborted the engine
          # ("GPU lacks the shared memory resources to run fused_moe kernel") under 16 users on
          # the RTX PRO 6000, so every regular-model configuration uses Triton
          EXTRA=(--moe-backend triton --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":8.0,"original_max_position_embeddings":32768},"max_position_embeddings":262144}')
          NBT="  native_block_tokens: 256" ;;
  hybrid) DIR=${HYBRID_DIR:-$LB/models/Qwen/Qwen3.8-27B}; CHUNK=1568; EXTRA=(); NBT="" ;;
  custom) DIR=${MODEL_DIR:?MODEL_DIR}; CHUNK=${CHUNK:-256}; read -r -a EXTRA <<< "${SERVE_EXTRA:-}"
          NBT=""; [ -n "${NATIVE_BLOCK_TOKENS:-}" ] && NBT="  native_block_tokens: $NATIVE_BLOCK_TOKENS" ;;
  *) echo "unknown model $MODEL"; exit 2 ;;
esac

H=$M/home-$MODEL-$MODE$I
KV=${KV_DIR:-$H/kv}
mkdir -p "$H"
[ "$KEEP" = fresh ] && rm -rf "$KV"
TF=${DASH_TOKEN_FILE:-$PWD/dash-token}
[ -s "$TF" ] || { python3 -c "import secrets;print(secrets.token_hex(24))" > "$TF"; chmod 600 "$TF"; }
TOKEN=$(cat "$TF")
[ -s vllm-api-keys ] || { python3 -c "import secrets;print(secrets.token_hex(24))" > vllm-api-keys; chmod 600 vllm-api-keys; }
VMODE=native; [ "$MODE" = lmcache_lb ] && VMODE=lmcache
# LB_POLICY / LB_CAPACITY_GB override the eviction policy (lru) and tier capacity (250 GB).
# LB_LADDER=none stores LMCache-path KV uncompressed (default ladder: fp8 on local NVMe), for a
# like-for-like comparison with LMCache's own fs adapter, which stores raw bytes
LADDER=""; [ -n "${LB_LADDER:-}" ] && LADDER="compression:
  ladder: {cpu: $LB_LADDER, local_nvme: $LB_LADDER, shared: $LB_LADDER, archive: $LB_LADDER}"
umask 077
cat > "$H/lifeboat.yaml" <<EOF
$LADDER
storage:
  write_mode: low_latency
  shared_lookup: $([ -n "${LB_SHARED:-}" ] && echo true || echo false)
  fallback_local:
    path: $KV
    capacity_gb: ${LB_CAPACITY_GB:-250}
eviction:
  l1_policy: ${LB_POLICY:-lru}
  admission:
    enabled: false
tenancy:
  enabled: false
telemetry:
  prometheus:
    port: $PROM
dashboard:
  port: $DASH
  auth_token: $TOKEN
engine:
  vllm_mode: $VMODE
  native_cpu_gb: $CPU_GB
$NBT
  lmcache_l1_size_gb: $CPU_GB
  lmcache_chunk_size: $CHUNK
EOF
umask 022

export HF_HOME=$PWD/hf FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_SERVER_DEV_MODE=1
export LIFEBOAT_CACHE_HOME=$H LIFEBOAT_CACHE_CONFIG=$H/lifeboat.yaml
KVT=()
# KV_LOAD_FAILURE_POLICY=recompute: a KV load that fails (LMCache "Some keys not found during
# retrieve") recomputes the request instead of failing it; vLLM's default "fail" took the engine
# down at 256 agents with LMCache DRAM + NVMe (2026-10-30)
KVLF=""; [ -n "${KV_LOAD_FAILURE_POLICY:-}" ] && KVLF=",\"kv_load_failure_policy\":\"$KV_LOAD_FAILURE_POLICY\""
case $MODE in
  vllm)       export LIFEBOAT_CACHE=0 ;;
  lmcache)    export LIFEBOAT_CACHE=0
              KVT=(--kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://localhost\",\"lmcache.mp.port\":6555,\"lmcache.mp.autostart\":true,\"lmcache.mp.autostart.server_args\":\"--l1-size-gb $CPU_GB --eviction-policy LRU --chunk-size $CHUNK\",\"lmcache.mp.autostart.wait_timeout\":180}${KVLF}}") ;;
  lmcache_lb|lb_native) export LIFEBOAT_CACHE=1 ;;
  # baselines: each engine's own disk tier, no LifeboatCache (they have no capacity limit;
  # matrix.py prunes the oldest files between cells to the same 250 GB Lifeboat runs with)
  vllm_fs)    export LIFEBOAT_CACHE=0
              BS=""; [ "$MODEL" = moe ] && BS=256
              KVT=(--kv-transfer-config "$(python3 -c '
import json, sys
extra = {"spec_name": "TieringOffloadingSpec", "cpu_bytes_to_use": int(sys.argv[3]) << 30,
         "secondary_tiers": [{"type": "fs", "root_dir": sys.argv[1] + "/fs"}]}
if sys.argv[2]:
    extra["block_size"] = int(sys.argv[2])
print(json.dumps({"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": extra}))
' "$KV" "$BS" "$CPU_GB")") ;;
  lmcache_fs) export LIFEBOAT_CACHE=0
              mkdir -p "$KV/lmcache"
              KVT=(--kv-transfer-config "$(python3 -c '
import json, shlex, sys
l2 = {"type": "fs", "base_path": sys.argv[1] + "/lmcache"}
if sys.argv[4] == "1":
    l2["use_odirect"] = True  # O_DIRECT for the NVMe layer
l2 = json.dumps(l2, separators=(",", ":"))
args = " ".join(shlex.quote(a) for a in ["--l1-size-gb", sys.argv[3], "--eviction-policy", "LRU", "--chunk-size", sys.argv[2], "--l2-adapter", l2])
kt = {"kv_connector": "LMCacheMPConnector", "kv_role": "kv_both", "kv_connector_extra_config": {
    "lmcache.mp.host": "tcp://localhost", "lmcache.mp.port": 6555, "lmcache.mp.autostart": True,
    "lmcache.mp.autostart.server_args": args, "lmcache.mp.autostart.wait_timeout": 180}}
if sys.argv[5]:
    kt["kv_load_failure_policy"] = sys.argv[5]
print(json.dumps(kt))
' "$KV" "$CHUNK" "$CPU_GB" "${LMCACHE_ODIRECT:-0}" "${KV_LOAD_FAILURE_POLICY:-}")") ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac

mapfile -t KEYS < <(grep -v '^\s*$' vllm-api-keys)
LOG=$M/logs/$MODEL-$MODE$I-$(date +%m%d-%H%M%S).log
T0=$(date +%s.%N)
setsid nohup "${VLLM_BIN:-.venv/bin/vllm}" serve "$DIR" --served-model-name bench \
  --host 127.0.0.1 --port $PORT --api-key "${KEYS[@]}" \
  --gpu-memory-utilization ${GPU_UTIL:-0.90} --max-model-len ${MAX_LEN:-262144} --kv-cache-dtype ${KV_DTYPE:-fp8} \
  --tensor-parallel-size ${TP:-1} \
  --max-num-seqs ${MAX_SEQS:-64} --enable-prefix-caching --enable-prompt-tokens-details \
  "${EXTRA[@]}" "${KVT[@]}" > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDF"
ln -sf "$LOG" "$M/logs/current$I.log"
for i in $(seq 1 450); do
  curl -sf -m 3 localhost:$PORT/health >/dev/null && break
  kill -0 "$(cat "$PIDF")" 2>/dev/null || { echo "EXITED"; tail -20 "$LOG"; exit 1; }
  sleep 2
done
curl -sf -m 3 localhost:$PORT/health >/dev/null || { echo "TIMEOUT"; tail -20 "$LOG"; exit 1; }
echo "ready model=$MODEL mode=$MODE ready_s=$(python3 -c "import time;print(round(time.time()-$T0,1))") log=$LOG"
