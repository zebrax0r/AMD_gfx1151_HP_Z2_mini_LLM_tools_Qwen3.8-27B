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
  CTX_SIZE="${CTX_SIZE:-65536}"
  UBATCH_SIZE="${UBATCH_SIZE:-8192}"
  BATCH_SIZE="${BATCH_SIZE:-8192}"
  PARALLEL="${PARALLEL:-1}"
  GPU_LAYERS="${GPU_LAYERS:-999}"
  FLASH_ATTN="${FLASH_ATTN:-auto}"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b}"

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
    defaults to 32768.
  - llama.cpp#20354: the Gated-DeltaNet fused kernel runs on GPU on gfx1151
    but performs no better than CPU fallback (RDNA register-pressure/tuning
    gaps). Expect roughly ~12 tokens/sec real-world, not comparable to
    dedicated-HBM (MI210-class) numbers.
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

cmd_download() {
  local hf_bin; hf_bin="$(_hf_bin)"
  mkdir -p "$MODEL_DIR"

  local avail_kb required_kb
  avail_kb="$(df -Pk "$MODEL_DIR" | awk 'NR==2{print $4}')"
  required_kb=$((35 * 1024 * 1024))
  if [[ "$avail_kb" -lt "$required_kb" ]]; then
    die "Need ~35GB free in $MODEL_DIR, have $((avail_kb/1024/1024))GB."
  fi

  export HF_XET_HIGH_PERFORMANCE=1  # current huggingface_hub uses Xet, not hf_transfer, for fast downloads
  log "Downloading $MODEL_FILE from $MODEL_REPO..."
  "$hf_bin" download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$MODEL_DIR"
  log "Downloading $MMPROJ_FILE from $MODEL_REPO..."
  "$hf_bin" download "$MODEL_REPO" "$MMPROJ_FILE" --local-dir "$MODEL_DIR"

  local got
  got="$(stat -c%s "$MODEL_DIR/$MODEL_FILE" 2>/dev/null || echo 0)"
  if [[ -n "${MODEL_FILE_EXPECT_BYTES:-}" ]]; then
    local diff=$(( got > MODEL_FILE_EXPECT_BYTES ? got - MODEL_FILE_EXPECT_BYTES : MODEL_FILE_EXPECT_BYTES - got ))
    if [[ "$diff" -gt $((1024*1024*1024)) ]]; then
      warn "$MODEL_FILE size ($got bytes) differs from expected (~$MODEL_FILE_EXPECT_BYTES) by more than 1GB. Re-download may be needed."
    fi
  fi
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

  local args=(
    --model "$MODEL_DIR/$MODEL_FILE"
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
  [[ -f "$MODEL_DIR/$MMPROJ_FILE" ]] && args+=(--mmproj "$MODEL_DIR/$MMPROJ_FILE")

  # Feature-detect flags whose names/semantics have churned on a fast-moving
  # master branch, rather than hardcoding and risking a startup failure.
  if _detect_flag --flash-attn >/dev/null; then
    args+=(--flash-attn "$FLASH_ATTN")
  fi
  if _detect_flag -dio >/dev/null; then
    args+=(-dio)
  fi

  log "Starting llama-server..."
  setsid "$LLAMA_SERVER_BIN" "${args[@]}" >> "$STDOUT_LOG" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$PID_FILE"

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

  local display_url="http://${SERVER_HOST}:${PORT}"
  cat <<EOF

================================================================
 Qwen3.8-27B is up.
   Base URL:    ${display_url}/v1
   Model alias: ${SERVED_MODEL_NAME}
   Ctx size:    ${CTX_SIZE}   Ubatch: ${UBATCH_SIZE}   Parallel: ${PARALLEL}
   Expect roughly ~7-12 tok/s (llama.cpp#20354, gfx1151 hybrid GDN kernel
   ties CPU-fallback speed). See README for details.
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

  # generationConfig.contextWindowSize tells qwen-code our real ceiling.
  # Without it, qwen-code defaults to assuming ~1,000,000 tokens (Qwen3.8's
  # advertised native/YaRN context) and paces its own auto-compaction
  # against that instead of what this server can actually serve — meaning
  # it grows the conversation until it hits a hard 400 "exceeds context
  # size" error instead of compacting proactively. Confirmed by hand: this
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
    --argjson ctx_size "$CTX_SIZE" \
    '
    .env[$env_key] = $api_key
    | .modelProviders.openai = ((.modelProviders.openai // []) | map(select(.id != $provider_id)) + [{
        baseUrl: $base_url, envKey: $env_key, id: $provider_id, name: $provider_name,
        generationConfig: {
          contextWindowSize: $ctx_size,
          timeout: 300000,
          streamIdleTimeoutMs: 600000,
          maxRetries: 1
        }
      }])
    | .security.auth = { baseUrl: $base_url, selectedType: "openai" }
    | .model.name = $model_name
    ' <<<"$existing")" || die "jq failed to update $settings_path — check it's valid JSON."

  printf '%s\n' "$updated" > "$settings_path"
  log "Updated $settings_path: default provider now this server ($base_url)."

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
    *) usage; exit 1 ;;
  esac
}

main "$@"
