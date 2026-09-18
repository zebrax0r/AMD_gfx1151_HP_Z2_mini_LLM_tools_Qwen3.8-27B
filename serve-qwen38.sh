#!/usr/bin/env bash
# One-click Qwen3.8-27B server for AMD Strix Halo (gfx1151), via llama.cpp/HIP.
# See README.md and docs/TROUBLESHOOTING.md for context on every default here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_FILE="qwen38.env"
ENV_EXAMPLE="qwen38-env.example"

# ---------------------------------------------------------------- helpers --

log()  { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; exit 1; }

redact() {
  # Strip the live API key out of arbitrary text before it hits a log/terminal.
  if [[ -n "${API_KEY:-}" ]]; then
    sed "s#${API_KEY}#[REDACTED]#g"
  else
    cat
  fi
}

load_env() {
  if [[ ! -f "$ENV_FILE" ]]; then
    [[ -f "$ENV_EXAMPLE" ]] || die "$ENV_EXAMPLE not found next to this script."
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    log "Created $ENV_FILE from $ENV_EXAMPLE. Edit it for non-default settings, then rerun."
  fi
  # shellcheck disable=SC1090
  source "$ENV_FILE"

  LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-vendor/llama.cpp}"
  MODEL_DIR="${MODEL_DIR:-models}"
  LOG_DIR="${LOG_DIR:-logs}"
  SECRETS_DIR="${SECRETS_DIR:-.secrets}"
  GPU_TARGET="${GPU_TARGET:-gfx1151}"
  ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
  export ROCM_PATH
  # Built binaries link against libhipblas.so.3 etc. under $ROCM_PATH/lib,
  # but this system's ROCm packaging doesn't register that path with
  # ldconfig/ld.so.conf.d — without this, every llama-cli/llama-server
  # invocation fails with "cannot open shared object file". Confirmed by
  # hand: `LD_LIBRARY_PATH=/opt/rocm/lib llama-server --help` works,
  # unset it doesn't.
  export LD_LIBRARY_PATH="${ROCM_PATH}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  HOST="${HOST:-0.0.0.0}"
  PORT="${PORT:-8000}"
  # SERVER_HOST is the address a CLIENT should use to reach this server —
  # distinct from HOST above, which is the bind address `serve` listens on.
  # Leave as 127.0.0.1 when running this script on the same machine as the
  # server. Set to this box's LAN IP/hostname in a *client-only* copy of
  # qwen38.env (e.g. a laptop that only runs `wire-qwen-code`, never
  # `build`/`serve`) so `wire-qwen-code` and the `serve` banner print the
  # right address for that machine to use.
  SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
  CTX_SIZE="${CTX_SIZE:-73728}"
  # CLIENT_CTX_SIZE is what we tell qwen-code via generationConfig.
  # contextWindowSize — deliberately LOWER than CTX_SIZE. qwen-code's own
  # token counts are estimates (its debug log literally logs
  # `estimated=true`), and a single large step (e.g. one big tool result)
  # can jump past its compaction trigger before compaction runs. Confirmed
  # by hand: reporting the exact CTX_SIZE (65536) still overshot to 68046
  # and hard-failed. The gap between CLIENT_CTX_SIZE and CTX_SIZE is the
  # safety margin that absorbs that slop.
  CLIENT_CTX_SIZE="${CLIENT_CTX_SIZE:-57344}"
  # These three cap how much a SINGLE turn can grow the conversation, which
  # matters more than CLIENT_CTX_SIZE's margin does: confirmed by hand that
  # a single turn (many fanned-out tool calls) can add tens of thousands of
  # tokens in one shot — qwen-code's per-call truncateToolOutputThreshold
  # (25000 chars stock) still lets a *batch* of several truncated calls add
  # up to more than the CLIENT_CTX_SIZE margin before compaction ever runs.
  # sessionTokenLimit is a deterministic backstop that doesn't depend on
  # qwen-code's (proven-unreliable) token-count estimation at all — it
  # blocks sending the next message outright once the recorded prompt
  # count is already over the limit, rather than sending and hard-failing
  # server-side. NOT applied via context-shift server-side (deliberately
  # rejected — see README: recurrent/hybrid memory can't be partially
  # truncated, risking corruption, not just a performance hit).
  #
  # ORDERING MATTERS: QWEN_SESSION_TOKEN_LIMIT must be HIGHER than
  # CLIENT_CTX_SIZE, not lower. Confirmed by hand getting this wrong: an
  # initial 50000 (below CLIENT_CTX_SIZE=57344) made the hard block fire
  # *before* qwen-code's own proactive compaction ever got a chance to run
  # automatically — every session just hit a wall requiring manual
  # /compress or /clear instead of compacting quietly in the background.
  # The intended layering is: CLIENT_CTX_SIZE (~57K) triggers soft
  # automatic compaction first; QWEN_SESSION_TOKEN_LIMIT (~66K) is a true
  # last-resort hard stop, comfortably below the real CTX_SIZE (73728) so
  # it still catches a runaway single turn before that hits the server.
  QWEN_TOOL_OUTPUT_THRESHOLD="${QWEN_TOOL_OUTPUT_THRESHOLD:-8000}"
  QWEN_TOOL_OUTPUT_LINES="${QWEN_TOOL_OUTPUT_LINES:-300}"
  QWEN_SESSION_TOKEN_LIMIT="${QWEN_SESSION_TOKEN_LIMIT:-65536}"
  # Caps a single turn's generated tokens. Without this, qwen-code defaults
  # to the model's *declared* output limit — effectively unbounded here.
  # Confirmed by hand: a real turn generated 12,000+ tokens at a healthy,
  # stable ~13.5 tok/s (server logs show no stall/corruption) and was still
  # going when qwen-code's own 15-minute stream-lifetime cap
  # (QWEN_STREAM_MAX_LIFETIME_MS, default 900000ms) killed the connection —
  # discarding the entire in-flight response instead of truncating cleanly.
  # 10000 tokens / ~13.5 tok/s (low end of measured range) ≈ 12.3 min,
  # comfortable margin under 15 min; hitting it gives a clean
  # `finish_reason: length` you can just ask to continue from, rather than
  # losing the whole response to a timeout.
  QWEN_CODE_MAX_OUTPUT_TOKENS="${QWEN_CODE_MAX_OUTPUT_TOKENS:-10000}"
  UBATCH_SIZE="${UBATCH_SIZE:-8192}"
  BATCH_SIZE="${BATCH_SIZE:-8192}"
  PARALLEL="${PARALLEL:-1}"
  GPU_LAYERS="${GPU_LAYERS:-999}"
  FLASH_ATTN="${FLASH_ATTN:-auto}"
  # MTP speculative decoding, using the model's own "nextn" tensors as a
  # draft head (no separate draft model needed) — confirmed by hand: real
  # measured throughput went from ~7.4 tok/s to ~14-20 tok/s (roughly
  # 1.9-2.7x) on this exact model/quant/hardware, output verified correct
  # (coherent reasoning, correct final answers, clean `finish_reason:
  # "stop"`). Matches the flags already used by the pre-existing Ollama
  # deployment on this same box. Set SPEC_TYPE="" to disable.
  SPEC_TYPE="${SPEC_TYPE:-draft-mtp}"
  SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-4}"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b}"

  # bench subcommand config — see qwen38-env.example for full rationale
  # (FP8/BF8 ruled out; Q4_K_M-and-below hard-blocked given MTP
  # draft-acceptance-collapse risk). Fallbacks here so `bench` still works
  # safely even if qwen38.env predates this addition (git pull without a
  # manual re-sync, an established gap in this repo's config pattern).
  BENCH_QUANTS="${BENCH_QUANTS:-Q6_K,Q5_K_M}"
  BENCH_ALLOWED_QUANTS="${BENCH_ALLOWED_QUANTS:-Q8_0,Q6_K,Q6_K_L,Q5_K_M,Q5_K_L,Q5_1,Q5_0}"
  BENCH_Q6_K_EXPECT_BYTES="${BENCH_Q6_K_EXPECT_BYTES:-23860565728}"
  BENCH_Q5_K_M_EXPECT_BYTES="${BENCH_Q5_K_M_EXPECT_BYTES:-20923877088}"
  BENCH_GEN_TOKENS="${BENCH_GEN_TOKENS:-512}"
  BENCH_REPEATS="${BENCH_REPEATS:-3}"
  BENCH_LONG_PROMPT_MIN_CHARS="${BENCH_LONG_PROMPT_MIN_CHARS:-80000}"
  BENCH_MIN_SPEEDUP_PCT="${BENCH_MIN_SPEEDUP_PCT:-10}"
  BENCH_MAX_ACCEPTANCE_DROP_PP="${BENCH_MAX_ACCEPTANCE_DROP_PP:-5}"

  BUILD_DIR="$LLAMA_CPP_DIR/build"
  LLAMA_SERVER_BIN="$BUILD_DIR/bin/llama-server"
  LLAMA_CLI_BIN="$BUILD_DIR/bin/llama-cli"
  PID_FILE="$LOG_DIR/qwen38.pid"
  SERVER_LOG="$LOG_DIR/qwen38.log"
  STDOUT_LOG="$LOG_DIR/qwen38.stdout.log"
  BUILD_INFO="$LOG_DIR/build-info.txt"
  API_KEY_FILE="$SECRETS_DIR/api_key"

  if [[ -f "$API_KEY_FILE" ]]; then
    API_KEY="$(cat "$API_KEY_FILE")"
  fi
}

need_bin() { command -v "$1" >/dev/null 2>&1 || die "Required tool '$1' not found on PATH."; }

# ------------------------------------------------------------- subcommands --

cmd_init() {
  mkdir -p "$MODEL_DIR" "$LOG_DIR" "$SECRETS_DIR"
  chmod 700 "$SECRETS_DIR"

  if [[ -z "${API_KEY:-}" ]]; then
    need_bin openssl
    ( umask 077; openssl rand -hex 24 > "$API_KEY_FILE" )
    chmod 600 "$API_KEY_FILE"
    API_KEY="$(cat "$API_KEY_FILE")"
    log "Generated a new API key in $API_KEY_FILE (not printed; will be redacted in all logs)."
  else
    log "Using existing API key from $ENV_FILE / $API_KEY_FILE."
  fi
  log "init complete. Directories ready: $MODEL_DIR, $LOG_DIR, $SECRETS_DIR"
}

cmd_probe() {
  local blocking=0

  echo "== GPU =="
  # NOTE: capture output into a variable before grepping it, rather than
  # piping a live command straight into `grep -q`. `grep -q` exits as soon
  # as it finds a match, which can SIGPIPE a still-writing producer; under
  # `set -o pipefail` that shows up as pipeline failure even though the
  # match succeeded. Capturing first avoids the race entirely.
  local lspci_out
  lspci_out="$(lspci 2>/dev/null)"
  if grep -qi 'strix halo' <<<"$lspci_out"; then
    log "Strix Halo GPU detected via lspci."
  else
    warn "Could not confirm Strix Halo via lspci. This script targets gfx1151 specifically."
  fi

  echo "== ROCm =="
  local rocm_root="${ROCM_PATH:-/opt/rocm}"
  if [[ -x "$rocm_root/bin/rocminfo" ]] || command -v rocminfo >/dev/null 2>&1; then
    local rocminfo_bin rocminfo_out
    rocminfo_bin="$(command -v rocminfo || echo "$rocm_root/bin/rocminfo")"
    rocminfo_out="$("$rocminfo_bin" 2>/dev/null)"
    if grep -qi "$GPU_TARGET" <<<"$rocminfo_out"; then
      log "rocminfo reports $GPU_TARGET as an available agent."
    else
      warn "rocminfo did not list $GPU_TARGET. ROCm userspace may be an older/mismatched build."
    fi
  else
    warn "rocminfo not found at $rocm_root/bin or on PATH. ROCm userspace not installed / ROCM_PATH not set."
    blocking=1
  fi

  echo "== GTT / kernel memory tuning =="
  if grep -q 'amdgpu.gttsize=' /proc/cmdline 2>/dev/null; then
    log "amdgpu.gttsize is set: $(grep -o 'amdgpu\.gttsize=[0-9]*' /proc/cmdline)"
  else
    warn "amdgpu.gttsize not set in /proc/cmdline. Large-context serving may be capped well below the 96GB nominal budget. See README §2c."
  fi

  echo "== Disk / RAM =="
  df -h "$SCRIPT_DIR" | tail -n1
  free -h | head -n2

  echo "== Known upstream bugs baked into this script's defaults =="
  cat <<'EOF'
  - llama.cpp#28211: HIP/gfx1151 gives silently WRONG logits on prompts
    longer than n_ubatch. Mitigation: UBATCH_SIZE/BATCH_SIZE default 8192
    (not a fix, just a much higher ceiling before the bug bites).
  - llama.cpp#27623: decode throughput collapses ~25x past ~80K KV position
    on this hybrid Gated-DeltaNet architecture. Mitigation: CTX_SIZE
    defaults to 73728.
  - llama.cpp#20354: the Gated-DeltaNet fused kernel runs on GPU on gfx1151
    but performs no better than CPU fallback (RDNA register-pressure/tuning
    gaps). Base rate ~7-12 tokens/sec; MTP speculative decoding (on by
    default, SPEC_TYPE=draft-mtp) measured at ~14-20 tokens/sec on this
    exact model/quant/hardware. Not comparable to dedicated-HBM (MI210-
    class) numbers either way.
  - llama.cpp#24437: GGML_HIP_ROCWMMA_FATTN causes up to -41% prefill
    throughput on gfx1151 at 8K+ context, worsening with context length.
    This build compiles it OFF (a deliberate divergence from some
    community "known-good Strix Halo" recipes that set it ON).
  - lemonade-sdk#3160: progressive generation corruption under sustained/
    concurrent load on ROCm-nightly gfx1151, recovers only on reload.
    Mitigation: PARALLEL defaults to 1 (single-slot serving); use
    `restart` if output degrades, or `install-watchdog` for automated
    selftest-gated restarts.
EOF

  [[ "$blocking" -eq 0 ]] || die "probe found a blocking issue (see ROCm section above)."
  log "probe complete — no blocking issues."
}

cmd_build() {
  need_bin cmake
  need_bin git
  [[ -n "${ROCM_PATH:-}" ]] || export ROCM_PATH=/opt/rocm
  local hipconfig_bin="$ROCM_PATH/bin/hipconfig"
  [[ -x "$hipconfig_bin" ]] || command -v hipconfig >/dev/null 2>&1 \
    || die "hipconfig not found. Is ROCm installed and ROCM_PATH set correctly? See README §2."

  mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
  if [[ -d "$LLAMA_CPP_DIR/.git" ]]; then
    log "Updating existing llama.cpp checkout to latest origin/master..."
    git -C "$LLAMA_CPP_DIR" fetch origin master
    git -C "$LLAMA_CPP_DIR" reset --hard origin/master
  else
    log "Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
  fi

  local hip_compiler
  hip_compiler="$(command -v hipconfig >/dev/null 2>&1 && hipconfig -l || echo "$ROCM_PATH/llvm/bin")/clang"

  log "Configuring (GPU_TARGETS=$GPU_TARGET, ROCWMMA_FATTN=OFF per llama.cpp#24437, NO_VMM=ON)..."
  cmake -S "$LLAMA_CPP_DIR" -B "$BUILD_DIR" \
    -DGGML_HIP=ON \
    -DGPU_TARGETS="$GPU_TARGET" \
    -DGGML_HIP_ROCWMMA_FATTN=OFF \
    -DGGML_HIP_NO_VMM=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_COMPILER="$hip_compiler"

  log "Building (this can take a while)..."
  cmake --build "$BUILD_DIR" -j"$(nproc)"

  [[ -x "$LLAMA_SERVER_BIN" ]] || die "Build finished but $LLAMA_SERVER_BIN is missing."

  log "Verifying required llama-server flags are present in this build..."
  local help_text
  help_text="$("$LLAMA_SERVER_BIN" --help 2>&1)"
  local required_flags=(--ubatch-size --parallel --jinja --mmproj --api-key --ctx-size)
  local missing=()
  for f in "${required_flags[@]}"; do
    grep -q -- "$f" <<<"$help_text" || missing+=("$f")
  done
  if [[ "${#missing[@]}" -gt 0 ]]; then
    die "This llama-server build is missing required flag(s): ${missing[*]}. Rerun 'build' to pick up a newer master, or check upstream for renames."
  fi

  {
    echo "commit: $(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)"
    echo "date:   $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    "$LLAMA_SERVER_BIN" --version 2>&1 || true
  } > "$BUILD_INFO"

  log "Build OK. Info recorded in $BUILD_INFO."
}

cmd_update() { cmd_build; }

_hf_bin() {
  # Bootstrap a local venv for the `hf` CLI if it's not already on PATH.
  # This system (like most current distros) enforces PEP 668
  # externally-managed-environment, so a bare `pip install` is refused —
  # a throwaway venv avoids touching system Python entirely.
  if command -v hf >/dev/null 2>&1; then
    echo "hf"
    return
  fi
  local venv_dir=".venv-hf"
  if [[ ! -x "$venv_dir/bin/hf" ]]; then
    need_bin python3
    log "Bootstrapping a local venv for the 'hf' CLI ($venv_dir, gitignored)..." >&2
    python3 -m venv "$venv_dir" >&2
    "$venv_dir/bin/pip" install --quiet --upgrade pip >&2
    "$venv_dir/bin/pip" install --quiet "huggingface_hub[cli]" >&2
  fi
  echo "$venv_dir/bin/hf"
}

# _download_gguf <repo> <file> [expect_bytes]
# Shared by cmd_download (MODEL_FILE/MMPROJ_FILE) and cmd_bench (candidate
# quants) — disk preflight, hf download (resumable/idempotent — skips
# already-complete files), size-sanity warning. Extracted so there's only
# one place that knows how to fetch a GGUF file from this repo's model
# source, not one copy per caller.
_download_gguf() {
  local repo="$1" file="$2" expect_bytes="${3:-}"
  local hf_bin; hf_bin="$(_hf_bin)"
  mkdir -p "$MODEL_DIR"

  local avail_kb required_kb
  avail_kb="$(df -Pk "$MODEL_DIR" | awk 'NR==2{print $4}')"
  required_kb=$((35 * 1024 * 1024))
  if [[ "$avail_kb" -lt "$required_kb" ]]; then
    die "Need ~35GB free in $MODEL_DIR, have $((avail_kb/1024/1024))GB."
  fi

  export HF_XET_HIGH_PERFORMANCE=1  # current huggingface_hub uses Xet, not hf_transfer, for fast downloads
  log "Downloading $file from $repo..."
  "$hf_bin" download "$repo" "$file" --local-dir "$MODEL_DIR"

  local got
  got="$(stat -c%s "$MODEL_DIR/$file" 2>/dev/null || echo 0)"
  if [[ -n "$expect_bytes" ]]; then
    local diff=$(( got > expect_bytes ? got - expect_bytes : expect_bytes - got ))
    if [[ "$diff" -gt $((1024*1024*1024)) ]]; then
      warn "$file size ($got bytes) differs from expected (~$expect_bytes) by more than 1GB. Re-download may be needed."
    fi
  fi
}

cmd_download() {
  _download_gguf "$MODEL_REPO" "$MODEL_FILE" "${MODEL_FILE_EXPECT_BYTES:-}"
  _download_gguf "$MODEL_REPO" "$MMPROJ_FILE" "${MMPROJ_FILE_EXPECT_BYTES:-}"
  log "download complete."
}

cmd_check() {
  [[ -x "$LLAMA_CLI_BIN" ]] || die "$LLAMA_CLI_BIN not found. Run 'build' first."
  [[ -f "$MODEL_DIR/$MODEL_FILE" ]] || die "$MODEL_DIR/$MODEL_FILE not found. Run 'download' first."

  log "Running bounded smoke test (load + a few tokens, ~120s timeout)..."
  # -st (--single-turn): without it, llama-cli enters interactive
  # conversation mode and waits on stdin instead of exiting after one
  # response — confirmed by hand on this build (b1-4ff829e).
  local out
  if ! out="$(timeout 120 "$LLAMA_CLI_BIN" \
      -m "$MODEL_DIR/$MODEL_FILE" \
      -ngl "$GPU_LAYERS" \
      -st \
      -p "Say OK." -n 8 < /dev/null 2>&1)"; then
    echo "$out" | redact | tail -n 40
    die "check failed — see output above for HIP/arch errors."
  fi
  if grep -qiE 'error|hip.*fail|no kernel image' <<<"$out"; then
    echo "$out" | redact | tail -n 40
    die "check produced errors in output above."
  fi
  log "check OK — model loads and generates without visible HIP/arch errors."
}

_detect_flag() {
  # _detect_flag <flag> -> prints flag if present in llama-server --help, else nothing
  local flag="$1" help_text
  help_text="$("$LLAMA_SERVER_BIN" --help 2>&1)"
  grep -q -- "$flag" <<<"$help_text" && echo "$flag"
}

# _build_server_args <model_file> -> populates global SERVER_ARGS()
# Shared by cmd_serve and cmd_bench so there is exactly one place that
# knows how to build a llama-server invocation — bench can never
# accidentally test a quant with a different flag set (e.g. MTP
# speculative decoding silently off) than what production actually runs,
# which would invalidate the whole point of bench-testing for
# draft-acceptance collapse. Only --model varies by caller.
_build_server_args() {
  local model_file="$1"
  SERVER_ARGS=(
    --model "$MODEL_DIR/$model_file"
    --host "$HOST" --port "$PORT"
    --ctx-size "$CTX_SIZE"
    --ubatch-size "$UBATCH_SIZE"
    --batch-size "$BATCH_SIZE"
    --parallel "$PARALLEL"
    -ngl "$GPU_LAYERS"
    --api-key "$API_KEY"
    --alias "$SERVED_MODEL_NAME"
    --jinja
    --log-file "$SERVER_LOG"
  )
  [[ -f "$MODEL_DIR/$MMPROJ_FILE" ]] && SERVER_ARGS+=(--mmproj "$MODEL_DIR/$MMPROJ_FILE")

  # Feature-detect flags whose names/semantics have churned on a fast-moving
  # master branch, rather than hardcoding and risking a startup failure.
  if _detect_flag --flash-attn >/dev/null; then
    SERVER_ARGS+=(--flash-attn "$FLASH_ATTN")
  fi
  if _detect_flag -dio >/dev/null; then
    SERVER_ARGS+=(-dio)
  fi
  if [[ -n "$SPEC_TYPE" ]] && _detect_flag --spec-type >/dev/null; then
    SERVER_ARGS+=(--spec-type "$SPEC_TYPE" --spec-draft-n-max "$SPEC_DRAFT_N_MAX")
  fi
}

# _wait_healthy <pid> — shared by cmd_serve and cmd_bench.
_wait_healthy() {
  local pid="$1"
  # Health-check polling always targets loopback — it's this machine
  # checking its own just-launched process, regardless of what address
  # other machines should use to reach it (that's SERVER_HOST, below).
  local health_url="http://127.0.0.1:${PORT}"
  local tries=60
  until curl -sf "${health_url}/health" >/dev/null 2>&1; do
    tries=$((tries - 1))
    if [[ "$tries" -le 0 ]]; then
      warn "Server did not report healthy within timeout. Check $STDOUT_LOG / $SERVER_LOG."
      exit 1
    fi
    kill -0 "$pid" 2>/dev/null || die "Server process died during startup. Check $STDOUT_LOG."
    sleep 2
  done
}

cmd_serve() {
  [[ -x "$LLAMA_SERVER_BIN" ]] || die "$LLAMA_SERVER_BIN not found. Run 'build' first."
  [[ -f "$MODEL_DIR/$MODEL_FILE" ]] || die "$MODEL_DIR/$MODEL_FILE not found. Run 'download' first."
  [[ -n "${API_KEY:-}" ]] || die "No API key set. Run 'init' first."

  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    die "Already running (PID $(cat "$PID_FILE")). Use 'restart' or 'stop' first."
  fi

  mkdir -p "$LOG_DIR"

  [[ "$PARALLEL" -le 1 ]] || warn "PARALLEL=$PARALLEL (>1). This raises exposure to lemonade-sdk#3160 (progressive corruption under concurrent load). Consider 'install-watchdog'."
  [[ "$CTX_SIZE" -le 81920 ]] || warn "CTX_SIZE=$CTX_SIZE (>81920). llama.cpp#27623 causes ~25x decode slowdown past ~80K KV position on this architecture."

  _build_server_args "$MODEL_FILE"

  log "Starting llama-server..."
  setsid "$LLAMA_SERVER_BIN" "${SERVER_ARGS[@]}" >> "$STDOUT_LOG" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$PID_FILE"

  _wait_healthy "$pid"

  local display_url="http://${SERVER_HOST}:${PORT}"
  cat <<EOF

================================================================
 Qwen3.8-27B is up.
   Base URL:    ${display_url}/v1
   Model alias: ${SERVED_MODEL_NAME}
   Ctx size:    ${CTX_SIZE} (qwen-code told: ${CLIENT_CTX_SIZE})   Ubatch: ${UBATCH_SIZE}   Parallel: ${PARALLEL}
   Speculative: ${SPEC_TYPE:-off}$( [[ -n "$SPEC_TYPE" ]] && echo " (draft-n-max ${SPEC_DRAFT_N_MAX})" )
   Expect roughly ~14-20 tok/s with MTP speculative decoding on (measured;
   ~7-12 tok/s without it — llama.cpp#20354, gfx1151 hybrid GDN kernel ties
   CPU-fallback speed at the base rate). See README for details.
$( [[ "$SERVER_HOST" == "127.0.0.1" ]] && echo "   (SERVER_HOST is 127.0.0.1 — set it to this box's LAN IP in qwen38.env if other machines need to reach this server.)" )

 Smoke test:
   curl ${display_url}/v1/chat/completions \\
     -H "Authorization: Bearer \$(cat ${API_KEY_FILE})" \\
     -H 'Content-Type: application/json' \\
     -d '{"model":"${SERVED_MODEL_NAME}","messages":[{"role":"user","content":"Say hello in one sentence."}]}'
================================================================
EOF
}

cmd_status() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    local pid; pid="$(cat "$PID_FILE")"
    log "Running (PID $pid)."
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      log "/health: OK"
    else
      warn "/health: not responding"
    fi
    echo "-- last 15 lines of $SERVER_LOG --"
    tail -n 15 "$SERVER_LOG" 2>/dev/null | redact || true
  else
    log "Not running."
  fi
}

cmd_stop() {
  [[ -f "$PID_FILE" ]] || { log "No PID file — nothing to stop."; return 0; }
  local pid; pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    log "Stopping PID $pid (SIGTERM)..."
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 15); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      warn "Still alive after 15s, sending SIGKILL."
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$PID_FILE"
  log "Stopped."
}

cmd_restart() { cmd_stop; cmd_serve; }

# _bench_quant_file <label> -> echoes the GGUF filename for a bench label.
# Bartowski's repo (our MODEL_REPO) names sibling quants of this model
# consistently as Qwen3.8-27B-<QUANT>.gguf — same pattern as our own
# MODEL_FILE (Qwen3.8-27B-Q8_0.gguf) — so no per-quant lookup table is
# needed beyond this.
_bench_quant_file() { echo "Qwen3.8-27B-$1.gguf"; }

# _bench_quant_expect_bytes <label> -> echoes the confirmed expected byte
# size for a bench label, or empty if unknown (download proceeds, just
# without the sanity-check warning). Real sizes must be confirmed via HEAD
# request against the actual repo before adding a case here — see
# qwen38-env.example's BENCH_*_EXPECT_BYTES comment for how these two were
# obtained. Do not guess.
_bench_quant_expect_bytes() {
  case "$1" in
    Q6_K) echo "$BENCH_Q6_K_EXPECT_BYTES" ;;
    Q5_K_M) echo "$BENCH_Q5_K_M_EXPECT_BYTES" ;;
    *) echo "" ;;
  esac
}

# _bench_long_prompt -> prints a long prompt built by repeating
# bench/prompts/short.txt until it exceeds BENCH_LONG_PROMPT_MIN_CHARS,
# cached under logs/bench/ so repeated bench runs are reproducible without
# committing a large file to git.
_bench_long_prompt() {
  local cache="$LOG_DIR/bench/.longprompt-cache"
  if [[ ! -f "$cache" ]]; then
    mkdir -p "$(dirname "$cache")"
    local seed content=""
    seed="$(cat bench/prompts/short.txt)"
    while [[ "${#content}" -lt "$BENCH_LONG_PROMPT_MIN_CHARS" ]]; do
      content+="$seed"$'\n\n'
    done
    printf '%s' "$content" > "$cache"
  fi
  cat "$cache"
}

# _bench_request <prompt_content> -> prints the raw JSON response, or
# nothing (with a warning already emitted) on failure. temperature=0
# removes sampling variance from the comparison and lets output be
# spot-checked for coherence across quants essentially for free.
_bench_request() {
  local prompt_content="$1"
  local max_time=$(( BENCH_GEN_TOKENS / 3 + 60 ))  # pessimistic ~3 tok/s floor + fixed buffer, same spirit as cmd_check's timeout
  local payload
  payload="$(jq -n --arg content "$prompt_content" --argjson max_tokens "$BENCH_GEN_TOKENS" --arg model "$SERVED_MODEL_NAME" \
    '{model: $model, messages: [{role: "user", content: $content}], max_tokens: $max_tokens, temperature: 0, stream: false}')"
  curl -sf --max-time "$max_time" "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
    -d "$payload"
}

cmd_bench() {
  need_bin jq
  [[ -x "$LLAMA_SERVER_BIN" ]] || die "$LLAMA_SERVER_BIN not found. Run 'build' first."
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    die "A server is already running (PID $(cat "$PID_FILE")). bench needs exclusive control of the GPU/port to swap models safely — run 'stop' first, then rerun bench."
  fi
  [[ -f "$MODEL_DIR/$MODEL_FILE" ]] || die "$MODEL_DIR/$MODEL_FILE (the current baseline) not found. Run 'download' first."
  [[ -f "bench/prompts/short.txt" ]] || die "bench/prompts/short.txt not found — run bench from the repo root."

  local requested="${1:-$BENCH_QUANTS}"
  local -a requested_labels allowed_labels
  IFS=',' read -ra requested_labels <<< "$requested"
  IFS=',' read -ra allowed_labels <<< "$BENCH_ALLOWED_QUANTS"

  local label ok a
  for label in "${requested_labels[@]}"; do
    ok=0
    for a in "${allowed_labels[@]}"; do [[ "$label" == "$a" ]] && ok=1 && break; done
    [[ "$ok" -eq 1 ]] || die "Quant '$label' is not in BENCH_ALLOWED_QUANTS ($BENCH_ALLOWED_QUANTS). Q4_K_M and below are deliberately excluded: a community report on similar hardware (same MTP speculative-decoding approach) found more aggressive quantization caused 'draft-acceptance collapse' — net SLOWER, not faster, because the draft head disagrees with a noisier main model more often. Not independently reproduced on this box, but treated as a real constraint, not a guess to test past. See README."
  done

  # Baseline is whatever's actually configured right now — so a future
  # rerun after adopting a candidate automatically benchmarks the new
  # production quant against further candidates, not a hardcoded "Q8_0".
  local -a entry_labels=("${MODEL_FILE} (current)") entry_files=("$MODEL_FILE")
  local file
  for label in "${requested_labels[@]}"; do
    file="$(_bench_quant_file "$label")"
    [[ "$file" == "$MODEL_FILE" ]] && continue  # already the baseline, don't test twice
    entry_labels+=("$label")
    entry_files+=("$file")
  done

  mkdir -p "$LOG_DIR/bench"
  local ts results_file
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  results_file="$LOG_DIR/bench/bench-${ts}.jsonl"

  # Guarantee production is left running regardless of how bench exits
  # (clean finish, error, or Ctrl-C mid-run) — stop whatever bench spawned,
  # restore MODEL_FILE, restart on the original config. Deliberately NOT
  # `local`: the EXIT trap fires after cmd_bench itself has already
  # returned (it's a script-exit trap, not a function-exit trap), so a
  # `local` here goes out of scope before the trap runs — confirmed by
  # hand (an early version hit "cleanup_done: unbound variable" at exactly
  # this point). Must be a real global to still be readable when the trap
  # fires.
  BENCH_ORIG_MODEL_FILE="$MODEL_FILE"
  BENCH_CLEANUP_DONE=0
  _bench_cleanup() {
    [[ "$BENCH_CLEANUP_DONE" -eq 1 ]] && return
    BENCH_CLEANUP_DONE=1
    cmd_stop
    MODEL_FILE="$BENCH_ORIG_MODEL_FILE"
    log "bench: restoring production server ($BENCH_ORIG_MODEL_FILE)..."
    cmd_serve
  }
  trap _bench_cleanup EXIT INT TERM

  log "bench: ${#entry_labels[@]} entries, ${BENCH_REPEATS} reps x 2 prompts each — this will take a while (30-90+ minutes at this hardware's throughput). Results: $results_file"

  local i
  for i in "${!entry_labels[@]}"; do
    label="${entry_labels[$i]}"; file="${entry_files[$i]}"
    if [[ ! -f "$MODEL_DIR/$file" ]]; then
      _download_gguf "$MODEL_REPO" "$file" "$(_bench_quant_expect_bytes "$label")"
    fi

    MODEL_FILE="$file"
    _build_server_args "$MODEL_FILE"
    log "bench: [$label] starting llama-server..."
    setsid "$LLAMA_SERVER_BIN" "${SERVER_ARGS[@]}" >> "$STDOUT_LOG" 2>&1 < /dev/null &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    _wait_healthy "$pid"

    _bench_request "Say OK." >/dev/null 2>&1 || true  # discarded warm-up

    local prompt_label prompt_content rep resp finish_reason prompt_n prompt_ms predicted_n predicted_ms draft_n draft_n_accepted
    for prompt_label in short long; do
      prompt_content="$([[ "$prompt_label" == short ]] && cat bench/prompts/short.txt || _bench_long_prompt)"
      for rep in $(seq 1 "$BENCH_REPEATS"); do
        resp="$(_bench_request "$prompt_content")" || { warn "bench: [$label] $prompt_label rep $rep: request failed, skipping."; continue; }
        finish_reason="$(jq -r '.choices[0].finish_reason // "error"' <<<"$resp")"
        prompt_n="$(jq -r '.timings.prompt_n // 0' <<<"$resp")"
        prompt_ms="$(jq -r '.timings.prompt_ms // 0' <<<"$resp")"
        predicted_n="$(jq -r '.timings.predicted_n // 0' <<<"$resp")"
        predicted_ms="$(jq -r '.timings.predicted_ms // 0' <<<"$resp")"
        draft_n="$(jq -r '.timings.draft_n // 0' <<<"$resp")"
        draft_n_accepted="$(jq -r '.timings.draft_n_accepted // 0' <<<"$resp")"
        jq -nc --arg type request --arg label "$label" --arg file "$file" --arg prompt "$prompt_label" --argjson rep "$rep" \
          --argjson prompt_n "$prompt_n" --argjson prompt_ms "$prompt_ms" --argjson predicted_n "$predicted_n" --argjson predicted_ms "$predicted_ms" \
          --argjson draft_n "$draft_n" --argjson draft_n_accepted "$draft_n_accepted" --arg finish_reason "$finish_reason" \
          '{type:$type,label:$label,file:$file,prompt:$prompt,rep:$rep,prompt_n:$prompt_n,prompt_ms:$prompt_ms,predicted_n:$predicted_n,predicted_ms:$predicted_ms,draft_n:$draft_n,draft_n_accepted:$draft_n_accepted,finish_reason:$finish_reason}' \
          >> "$results_file"
        local gen_tps=0; [[ "$predicted_ms" != "0" ]] && gen_tps="$(jq -n --argjson n "$predicted_n" --argjson ms "$predicted_ms" '($n / ($ms/1000)) | round')"
        log "bench: [$label] $prompt_label rep $rep/$BENCH_REPEATS: gen ${gen_tps} tok/s, finish=${finish_reason}"
      done
    done

    cmd_stop
  done

  echo "=== bench results: $results_file ==="
  jq -s '
    map(select(.type=="request"))
    | group_by(.label + "|" + .prompt)
    | map({
        label: .[0].label, prompt: .[0].prompt, n: length,
        gen_tps_median: (map(if .predicted_ms>0 then (.predicted_n/(.predicted_ms/1000)) else 0 end) | sort | .[(length-1)/2|floor]),
        prompt_tps_median: (map(if .prompt_ms>0 then (.prompt_n/(.prompt_ms/1000)) else 0 end) | sort | .[(length-1)/2|floor]),
        accept_pct_median: ([.[] | select(.draft_n>0) | (100*.draft_n_accepted/.draft_n)] | if length>0 then (sort | .[(length-1)/2|floor]) else null end),
        stop_pct: (100 * (map(.finish_reason=="stop") | map(if . then 1 else 0 end) | add) / length),
        no_empty: (map(.predicted_n>0) | all)
      })
  ' "$results_file" | tee "$LOG_DIR/bench/bench-${ts}-summary.json"

  echo
  echo "Decision rule: candidate must beat baseline long-context gen tok/s by >= ${BENCH_MIN_SPEEDUP_PCT}%, and not drop draft-accept% by more than ${BENCH_MAX_ACCEPTANCE_DROP_PP} points vs baseline, with all responses finishing cleanly."
  # NOTE: finish_reason=="stop" is reported but NOT a pass/fail gate.
  # Confirmed by hand: this model reasons at length before answering (a
  # real production turn ran 12,000+ tokens without reaching a natural
  # stop — see the QWEN_CODE_MAX_OUTPUT_TOKENS fix/README), so at a fixed
  # BENCH_GEN_TOKENS budget, healthy responses routinely hit `length`
  # before finishing a thought — that's a budget artifact, not a quant
  # defect. Gating on all_stop would fail every candidate regardless of
  # quality. The real correctness signal is no_empty (every response
  # actually produced output) — a truly broken/garbled load tends to
  # error or return nothing, not merely truncate.
  local baseline_label="${entry_labels[0]}"
  jq -s --arg baseline "$baseline_label" --argjson min_speedup "$BENCH_MIN_SPEEDUP_PCT" --argjson max_drop "$BENCH_MAX_ACCEPTANCE_DROP_PP" '
    map(select(.type=="request"))
    | group_by(.label + "|" + .prompt)
    | map({
        label: .[0].label, prompt: .[0].prompt,
        gen_tps_median: (map(if .predicted_ms>0 then (.predicted_n/(.predicted_ms/1000)) else 0 end) | sort | .[(length-1)/2|floor]),
        accept_pct_median: ([.[] | select(.draft_n>0) | (100*.draft_n_accepted/.draft_n)] | if length>0 then (sort | .[(length-1)/2|floor]) else null end),
        stop_pct: (100 * (map(.finish_reason=="stop") | map(if . then 1 else 0 end) | add) / length),
        no_empty: (map(.predicted_n>0) | all)
      })
    | (map(select(.label==$baseline and .prompt=="long")) | .[0]) as $base
    | map(select(.label!=$baseline and .prompt=="long"))
    | map(. + {
        speedup_pct: (if $base.gen_tps_median>0 then (100*(.gen_tps_median-$base.gen_tps_median)/$base.gen_tps_median) else null end),
        accept_drop_pp: (if (.accept_pct_median!=null and $base.accept_pct_median!=null) then ($base.accept_pct_median - .accept_pct_median) else null end)
      })
    | map(. + { pass: (.no_empty and (.speedup_pct!=null and .speedup_pct>=$min_speedup) and ((.accept_drop_pp==null) or (.accept_drop_pp<=$max_drop))) })
  ' "$results_file"

  log "bench complete. Adoption is manual — see README 'Adopting a bench result'. Restoring production server now."
}

cmd_selftest() {
  local url="http://127.0.0.1:${PORT}"
  local resp
  resp="$(curl -sf "${url}/v1/chat/completions" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"${SERVED_MODEL_NAME}"'","messages":[{"role":"user","content":"Reply with exactly the word: PONG"}],"max_tokens":16}' \
    2>/dev/null)" || { warn "selftest: request failed."; return 1; }

  grep -qi 'pong' <<<"$resp" || { warn "selftest: response did not contain expected content: $(redact <<<"$resp")"; return 1; }
  log "selftest OK."
}

cmd_install_watchdog() {
  local interval_hours="${1:-2}"
  mkdir -p "$HOME/.config/systemd/user"
  local unit_dir="$HOME/.config/systemd/user"
  local script_path; script_path="$(readlink -f "${BASH_SOURCE[0]}")"

  cat > "$unit_dir/qwen38-watchdog.service" <<EOF
[Unit]
Description=Qwen3.8 selftest-gated watchdog (restarts only on failed selftest)

[Service]
Type=oneshot
WorkingDirectory=${SCRIPT_DIR}
ExecStart=/bin/bash -c '${script_path} selftest || ${script_path} restart'
EOF

  cat > "$unit_dir/qwen38-watchdog.timer" <<EOF
[Unit]
Description=Run qwen38-watchdog every ${interval_hours}h

[Timer]
OnBootSec=15min
OnUnitActiveSec=${interval_hours}h
Persistent=true

[Install]
WantedBy=timers.target
EOF

  systemctl --user daemon-reload
  systemctl --user enable --now qwen38-watchdog.timer
  log "Watchdog installed: selftest every ${interval_hours}h, restarts server only if selftest fails."
  log "This does NOT blindly restart on a timer — see README for why."
}

cmd_uninstall_watchdog() {
  systemctl --user disable --now qwen38-watchdog.timer 2>/dev/null || true
  rm -f "$HOME/.config/systemd/user/qwen38-watchdog.service" "$HOME/.config/systemd/user/qwen38-watchdog.timer"
  systemctl --user daemon-reload
  log "Watchdog uninstalled."
}

cmd_wire_qwen_code() {
  need_bin jq
  local target="${QWEN_ENV_TARGET:-user}"
  local qwen_dir
  if [[ "$target" == "project" ]]; then
    [[ -n "${QWEN_PROJECT_DIR:-}" ]] || die "QWEN_ENV_TARGET=project but QWEN_PROJECT_DIR is not set."
    qwen_dir="${QWEN_PROJECT_DIR}/.qwen"
  else
    qwen_dir="$HOME/.qwen"
  fi
  local env_path="$qwen_dir/.env"
  local settings_path="$qwen_dir/settings.json"
  local base_url="http://${SERVER_HOST}:${PORT}/v1"

  mkdir -p "$qwen_dir"
  if [[ -f "$env_path" ]]; then
    cp "$env_path" "${env_path}.bak.$(date +%s)"
    log "Backed up existing $env_path"
  fi
  cat > "$env_path" <<EOF
OPENAI_BASE_URL=${base_url}
OPENAI_API_KEY=${API_KEY}
OPENAI_MODEL=${SERVED_MODEL_NAME}
EOF
  chmod 600 "$env_path"
  log "Wrote $env_path"

  # qwen-code's settings.json 'security.auth' takes precedence over the
  # .env file above — writing only .env silently has no effect if
  # settings.json already names a different provider as default (as found
  # on this box, pointed at a pre-existing Ollama deployment on :11434).
  # Patch settings.json's default provider in place, non-destructively:
  # add/update our provider entry by id, keep any others (e.g. Ollama)
  # untouched in the list, just stop selecting them by default.
  local provider_id="qwen38-gfx1151"
  local env_key="QWEN38_GFX1151_API_KEY"
  local existing="{}"
  if [[ -f "$settings_path" ]]; then
    cp "$settings_path" "${settings_path}.bak.$(date +%s)"
    log "Backed up existing $settings_path"
    existing="$(cat "$settings_path")"
  fi

  [[ "$CLIENT_CTX_SIZE" -lt "$CTX_SIZE" ]] || warn "CLIENT_CTX_SIZE ($CLIENT_CTX_SIZE) is not lower than CTX_SIZE ($CTX_SIZE) — this removes qwen-code's safety margin and it will likely overshoot into a hard 400 again."
  [[ "$QWEN_SESSION_TOKEN_LIMIT" -gt "$CLIENT_CTX_SIZE" ]] || warn "QWEN_SESSION_TOKEN_LIMIT ($QWEN_SESSION_TOKEN_LIMIT) is not higher than CLIENT_CTX_SIZE ($CLIENT_CTX_SIZE) — the hard session-token block will fire before qwen-code's own proactive compaction gets a chance to run, so sessions will hit a wall requiring manual /compress or /clear instead of compacting automatically. Confirmed by hand — see README."
  [[ "$QWEN_SESSION_TOKEN_LIMIT" -lt "$CTX_SIZE" ]] || warn "QWEN_SESSION_TOKEN_LIMIT ($QWEN_SESSION_TOKEN_LIMIT) is not lower than CTX_SIZE ($CTX_SIZE) — it won't catch a runaway turn before the server's real hard limit does."

  # generationConfig.contextWindowSize tells qwen-code our ceiling.
  # Deliberately CLIENT_CTX_SIZE (lower than the real CTX_SIZE), not
  # CTX_SIZE itself — see the CLIENT_CTX_SIZE comment in load_env for why:
  # qwen-code's token counts are estimates, and reporting the exact real
  # limit still overshot into a hard 400 in practice. Without this field at
  # all, qwen-code defaults to assuming ~1,000,000 tokens (Qwen3.8's
  # advertised native/YaRN context) and paces its own auto-compaction
  # against that instead of what this server can actually serve. This
  # field is documented in qwen-code's own model-providers.md and is an
  # "impermeable layer" that fully replaces generationConfig for this
  # provider entry (per-field settings-level values are NOT inherited).
  # Timeouts are raised generously given this hardware's measured
  # multi-minute turn latency (see README "Expected performance").
  local updated
  updated="$(jq \
    --arg base_url "$base_url" \
    --arg env_key "$env_key" \
    --arg api_key "$API_KEY" \
    --arg provider_id "$provider_id" \
    --arg provider_name "Qwen3.8-27B (gfx1151 llama.cpp/HIP)" \
    --arg model_name "$SERVED_MODEL_NAME" \
    --argjson ctx_size "$CLIENT_CTX_SIZE" \
    --argjson tool_output_threshold "$QWEN_TOOL_OUTPUT_THRESHOLD" \
    --argjson tool_output_lines "$QWEN_TOOL_OUTPUT_LINES" \
    --argjson session_token_limit "$QWEN_SESSION_TOKEN_LIMIT" \
    --argjson max_output_tokens "$QWEN_CODE_MAX_OUTPUT_TOKENS" \
    '
    .env[$env_key] = $api_key
    | .modelProviders.openai = ((.modelProviders.openai // []) | map(select(.id != $provider_id)) + [{
        baseUrl: $base_url, envKey: $env_key, id: $provider_id, name: $provider_name,
        generationConfig: {
          contextWindowSize: $ctx_size,
          timeout: 300000,
          streamIdleTimeoutMs: 600000,
          maxRetries: 1,
          samplingParams: { max_tokens: $max_output_tokens }
        }
      }])
    | .security.auth = { baseUrl: $base_url, selectedType: "openai" }
    | .model.name = $model_name
    | .tools.truncateToolOutputThreshold = $tool_output_threshold
    | .tools.truncateToolOutputLines = $tool_output_lines
    | .model.sessionTokenLimit = $session_token_limit
    ' <<<"$existing")" || die "jq failed to update $settings_path — check it's valid JSON."

  printf '%s\n' "$updated" > "$settings_path"
  log "Updated $settings_path: default provider now this server ($base_url)."
  log "Also lowered tools.truncateToolOutputThreshold/Lines and set model.sessionTokenLimit (see README) — these are global settings, not just for this provider. Requires a fresh 'qwen' session (not just a retry) to take effect."

  cat <<EOF

Alternative (interactive): run 'qwen', then '/auth' -> Custom Provider, and
enter the same base URL / key / model shown above (redacted here).
EOF
}

usage() {
  cat <<'EOF'
Usage: ./serve-qwen38.sh <command>

  init               Create dirs, generate/reuse API key
  probe              Preflight: GPU, ROCm, GTT tuning, disk/RAM, known-bug summary
  build              Clone/update llama.cpp to latest master and build for gfx1151
  update             Alias for build
  download           Fetch GGUF weights + mmproj (needs ~35GB free)
  check              Bounded smoke-load test (no long-running server)
  serve              Launch llama-server in the background, wait for /health
  status             Show whether it's running, health, recent log lines
  stop               Stop the running server
  restart            stop + serve (also the fix for lemonade-sdk#3160 corruption)
  selftest           One deterministic request; exit nonzero on failure
  install-watchdog [hours]   Install a systemd --user timer: selftest, restart only on failure (default 2h)
  uninstall-watchdog         Remove the watchdog timer
  wire-qwen-code     Write ~/.qwen/.env (or project .qwen/.env) pointing at this server
  bench [quants]     Benchmark alternate GGUF quants (default: $BENCH_QUANTS) against
                      the current MODEL_FILE, using production's exact server flags.
                      Always restores the original server on exit. See README.
EOF
}

main() {
  load_env
  case "${1:-}" in
    init) cmd_init ;;
    probe) cmd_probe ;;
    build) cmd_build ;;
    update) cmd_update ;;
    download) cmd_download ;;
    check) cmd_check ;;
    serve) cmd_serve ;;
    status) cmd_status ;;
    stop) cmd_stop ;;
    restart) cmd_restart ;;
    selftest) cmd_selftest ;;
    install-watchdog) shift; cmd_install_watchdog "${1:-2}" ;;
    uninstall-watchdog) cmd_uninstall_watchdog ;;
    wire-qwen-code) cmd_wire_qwen_code ;;
    bench) shift; cmd_bench "${1:-}" ;;
    *) usage; exit 1 ;;
  esac
}

main "$@"
