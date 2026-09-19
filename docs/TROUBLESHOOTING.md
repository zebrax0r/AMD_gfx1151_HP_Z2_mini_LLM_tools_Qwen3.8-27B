# Troubleshooting

## `error while loading shared libraries: libhipblas.so.3: cannot open shared object file`

Already handled by this script (`load_env` exports `LD_LIBRARY_PATH` to
include `$ROCM_PATH/lib`), documented here in case you invoke
`llama-server`/`llama-cli` directly outside the script. This machine's ROCm
channel (`stable.repo.amd.com`, `amdrocm-*10.0-gfx1151` packages) installs
its shared libraries under a versioned path
(`/opt/rocm/core-10.0/lib`, reached via the `/opt/rocm/lib` ->
`/etc/alternatives/rocm-lib` symlink) but does not register that path with
`ldconfig`/`ld.so.conf.d`, so any freshly-built ROCm/HIP binary fails to
start unless `LD_LIBRARY_PATH` (or an `ld.so.conf.d` entry + `ldconfig`) is
set explicitly. If you build other ROCm software outside this repo on this
box, you'll hit the same thing.

## `llama-server: error while loading shared libraries: libllama-server-impl.so: cannot open shared object file`

Already handled by this script (`load_env` exports `LD_LIBRARY_PATH` to
include this repo's own `vendor/llama.cpp/build/bin`), documented here in
case you invoke the binary directly or still hit this. Real incident,
2026-09-19: renaming this repo's directory
(`qwen3.8_amd_gfx1151` -> `laguna_s21_amd_gfx1151`) broke `llama-server`
outright with exactly this error, even though nothing about the actual
build changed. Cause: cmake bakes an **absolute** `RUNPATH` into every
binary/`.so` it produces at build time (confirmed via `readelf -d
llama-server`) — e.g. `/home/zebra/Downloads/qwen3.8_amd_gfx1151/vendor/
llama.cpp/build/bin`. Once that directory no longer exists, every binary
that links against `libllama-server-impl.so`, `libggml-hip.so`, etc.
fails to start, and a full rebuild (or manual `patchelf --set-rpath` on
every affected binary) would otherwise be the only fix. Since these
binaries use `RUNPATH` (not the older `RPATH`), the dynamic linker checks
`LD_LIBRARY_PATH` **before** `RUNPATH` — so exporting the correct current
`build/bin` path there overrides the stale baked-in one without a rebuild.
If you ever move/rename this repo's directory by hand outside of `git mv`
and hit a similar "cannot open shared object file" for one of this
repo's own built `.so` files (as opposed to a system ROCm library), this
is almost certainly why — check `readelf -d <binary> | grep -i path`
first before assuming a rebuild is required.

## Garbled / wrong output on a very long single prompt

Check `UBATCH_SIZE` in `laguna.env` (default 2048). This is the class of
bug described in
[llama.cpp#28211](https://github.com/ggml-org/llama.cpp/issues/28211): on
HIP/gfx1151, prompts longer than `n_ubatch` have been reported to get
silently wrong logits (no crash, no error — just bad output) on some
model/config combinations. Verified NOT reproduced on this exact
model/build at `UBATCH_SIZE=2048` up to 4088 tokens tested
(needle-in-haystack, exact marker retrieval) — nothing past that has been
directly verified. If you're pushing a much longer single prompt, raise
`UBATCH_SIZE`/`BATCH_SIZE` and re-test the same way (a unique marker
string buried in a long prompt, check it comes back exactly) rather than
assuming a larger value is automatically safe, or unsafe.

## Output quality degrades over a long session, a fresh restart fixes it

[lemonade-sdk#3160](https://github.com/lemonade-sdk/lemonade/issues/3160):
progressive generation corruption under sustained/concurrent load on
ROCm-nightly gfx1151. Run `./serve-laguna.sh restart`. If you routinely hit
this, install the selftest-gated watchdog:

```bash
./serve-laguna.sh install-watchdog 2   # selftest every 2h, restarts only on failure
```

This deliberately does **not** restart blindly on a timer — a
selftest-gated restart doesn't interrupt healthy sessions and doubles as a
diagnostic signal worth reporting upstream if it fires often.

## Build succeeds but `serve`/`build` reports a missing required flag

`build`'s verification step checks that `llama-server --help` exposes
`--ubatch-size`, `--parallel`, `--jinja`, `--mmproj`, `--api-key`,
`--ctx-size`. If one is missing, your checkout is likely too old or a flag
was renamed upstream. Rerun `./serve-laguna.sh update` to pull latest
master.

## `apt update` finds no ROCm package for this Ubuntu release

Ubuntu 26.04 ("resolute") is new enough that ROCm's apt repo may not
publish a `resolute` component yet. Use the `noble` component instead (see
README "System prep") — ROCm's userspace isn't kernel-ABI-tied the way the
driver is. If a needed fix is only in ROCm nightly, use a
[TheRock](https://github.com/ROCm/TheRock) nightly tarball and set
`ROCM_PATH` in `laguna.env`.

## Server hangs on model load

Confirm `-dio` made it into the assembled `llama-server` argv (visible in
`logs/laguna.stdout.log` at startup, or check `serve`'s flag-detection
logic against `llama-server --help`). Direct I/O is reported necessary for
models this large (~73GB across 3 shards) to avoid a load hang on some
configurations.

## `probe` warns that `amdgpu.gttsize` isn't set

You haven't applied the GRUB kernel parameter from README "System prep §2"
yet (or haven't rebooted since). Without it, GPU memory allocation is
capped well below the nominal 96GB budget, and large-context serving may
fail unexpectedly.

## `[API Error: 400 request (N tokens) exceeds the available context size (CTX_SIZE tokens)]`

This is a `qwen-code` client-side behavior, not specific to which model is
behind the server — worth understanding in full if you hit it:

1. **Naive fix (insufficient on its own)**: raise `CTX_SIZE` in
   `laguna.env` and `./serve-laguna.sh restart`.
2. **`qwen-code` needs to know the ceiling**: it doesn't know this server's
   real `CTX_SIZE` unless its provider entry in `settings.json` sets
   `generationConfig.contextWindowSize`. Without it, `qwen-code` assumes
   the model's advertised native context (Laguna's is 1,048,576) and won't
   proactively compact — it just keeps growing until the server
   hard-rejects the request.
3. **Telling it the *exact* ceiling still isn't enough**: `qwen-code`'s own
   token counts are estimates (`estimated=true` in its debug log), and a
   single large step (one big tool result, a large file read) can jump
   past its compaction trigger before compaction gets a chance to run.
   Confirmed directly on this repo's deployment history: reporting the
   exact real ceiling still overshot and hard-failed anyway.
4. **A margin between `CLIENT_CTX_SIZE` and `CTX_SIZE` helps but isn't
   sufficient on its own either**: confirmed directly — a 16,384-token
   margin was completely blown through in one turn. `qwen-code`'s
   proactive compaction does not reliably trigger before a large enough
   single addition.
5. **Cap how much a single turn can add, don't just hope compaction
   catches it in time.** `qwen-code`'s per-tool-call
   `tools.truncateToolOutputThreshold` (stock default 25,000 characters)
   still lets a *batch* of several fanned-out tool calls (e.g. reading
   many files at once) add up to far more than that in one turn.
   `wire-qwen-code` sets `tools.truncateToolOutputThreshold=8000` and
   `tools.truncateToolOutputLines=300` to bound this. These are global
   `settings.json` fields, not provider-specific, and **require a fresh
   `qwen` session to take effect** (not just Ctrl+Y retry).
6. **`model.sessionTokenLimit` ordering matters — it must sit *above*
   `CLIENT_CTX_SIZE`, not below it.** Confirmed directly by getting this
   wrong first: a value below `CLIENT_CTX_SIZE` meant the hard block fired
   *before* `qwen-code`'s own proactive compaction ever got a chance to
   run — every session just hit a wall (`Session token limit exceeded`)
   requiring manual `/compress` or `/clear`, instead of compacting quietly
   in the background as intended. The default here (114,688) sits between
   `CLIENT_CTX_SIZE` (98,304, soft trigger) and `CTX_SIZE` (131,072, real
   hard limit), so it functions as the true last-resort backstop it was
   meant to be.

If you see `Session token limit exceeded: N tokens > LIMIT limit` from
`qwen-code` itself (not a server 400), that's this backstop working as
designed, not a new bug — run `/compress` (preserves the session) or
`/clear` (starts fresh) as it suggests. **But also check whether the
backstop itself has room to move up first**: confirmed directly
2026-09-19, a real session hit this at 115,063 tokens against a
114,688 limit, while the real `CTX_SIZE` at the time was 131,072 —
16,009 tokens of already-safe server capacity were sitting unused because
`QWEN_SESSION_TOKEN_LIMIT` had simply never been revisited since it was
first set. `grep -E "^CTX_SIZE=|^QWEN_SESSION_TOKEN_LIMIT="
laguna.env`, and if there's real daylight between them, raise
`QWEN_SESSION_TOKEN_LIMIT` (and `CLIENT_CTX_SIZE` proportionally) toward
`CTX_SIZE` rather than immediately reaching for `/compress`/`/clear`.

Why not just enable llama-server's `--context-shift` ("infinite"
generation by discarding old context instead of hard-failing)? Its discard
mechanism needs to partially truncate the KV cache. That's fine for
Laguna's conventional attention, but this repo treats it as a
per-architecture risk to re-evaluate rather than a default to lean on —
some hybrid/recurrent architectures can't have their memory partially
truncated safely at all, and the client-side layered defenses above work
regardless of architecture, so there's no need to depend on that
assumption either way.

`./serve-laguna.sh wire-qwen-code` writes all of this automatically;
rerun it after changing any of `CTX_SIZE`, `CLIENT_CTX_SIZE`,
`QWEN_TOOL_OUTPUT_THRESHOLD`, `QWEN_TOOL_OUTPUT_LINES`, or
`QWEN_SESSION_TOKEN_LIMIT`, and on every other machine (laptops included)
that points `qwen-code` at this server.

## `[API Error: Stream exceeded its 900000ms upstream-wait cap after N chunks without completing...]`

This is `qwen-code`'s own client-side stream-lifetime cap
(`QWEN_STREAM_MAX_LIFETIME_MS`, default 900000ms = 15 min), not a server
error — it kills the connection and **discards the entire in-flight
response** if a single streamed reply runs longer than that, regardless of
whether generation is healthy.

First check the server side wasn't actually stuck: `tail logs/laguna.log`
for that time window — a healthy generation shows steady `n_gen`/`tg`
climbing at a consistent tok/s (confirmed directly: one real incident
generated 12,000+ tokens at a stable pace with no stall, so the throughput
alone doesn't tell you whether the *content* was useful progress or a
repetitive/stuck ramble — you'd need to check what `qwen-code` actually
displayed).

`wire-qwen-code` sets `QWEN_CODE_MAX_OUTPUT_TOKENS` (default 10,000, via
`generationConfig.samplingParams.max_tokens`) specifically to prevent
this — without it, `qwen-code` defaults to the model's declared output
limit (effectively unbounded here), so a long turn has nothing stopping it
short of this 15-minute wall. If you still hit this after `wire-qwen-code`,
either raise `QWEN_CODE_MAX_OUTPUT_TOKENS` further (if the task genuinely
needs more output per turn) or investigate whether the model is stuck
rambling rather than making real progress (check for repetitive content in
the transcript) — raising the timeout alone doesn't fix a runaway
generation, it just lets it run longer before losing the response anyway.

## `qwen-code` seems to hang / doesn't respond for a long time

Likely not a hang. `qwen-code`'s full agentic mode sends a large system/
tool-definition prompt (measured 8,000-20,000+ tokens on this setup) and
can make several sequential LLM round-trips per turn (tool calls,
reasoning), each only partially served from cache. A single interactive
request can genuinely take a while. Confirm it's actually working via
`./serve-laguna.sh status` or `tail -f logs/laguna.log` — you should see
`prompt processing` and `n_gen` lines advancing. Use `qwen --bare` for a
lighter/faster session if you don't need the full tool/skill registry.

## `wire-qwen-code` ran but `qwen` still talks to a different server

Check `~/.qwen/settings.json` — its `security.auth.baseUrl` takes
precedence over the `.env` file. `wire-qwen-code` patches both, but if
you've manually edited `settings.json` since, or have multiple provider
entries, confirm `security.auth` points at `http://127.0.0.1:$PORT/v1`.
This box has a separate, pre-existing Ollama deployment (port 11434,
unrelated to this repo) that was already wired into `settings.json` when
this repo was first set up — `wire-qwen-code` preserves that entry in
`modelProviders.openai` (so you can switch back by hand) but changes which
one is default.

## `wire-qwen-code` ran, no errors, but a config value (e.g. `sessionTokenLimit`) is still old

Two things to check, in order:

1. **Is `laguna.env` actually current?** `wire-qwen-code` only writes
   what's in `laguna.env` — if a value was never explicitly set there, it
   falls back to a default hardcoded inside `serve-laguna.sh` itself.
   Confirmed by hand on this repo's history: those internal fallback
   defaults can go stale after a config bump if `laguna.env` isn't
   updated to match — any machine whose `laguna.env` predates a given var
   (or never had it explicitly set) silently keeps the *old* number
   instead of the current one. `wire-qwen-code`'s own ordering-validation
   warnings are a useful tell here — if you see e.g.
   `QWEN_SESSION_TOKEN_LIMIT (65536) is not lower than CTX_SIZE (65536)`,
   both sides of that comparison resolving to the same stale number is a
   sign at least one of them is falling back to an old default rather
   than reading a real value. Fix: `grep -E
   "^CTX_SIZE=|^CLIENT_CTX_SIZE=|^QWEN_SESSION_TOKEN_LIMIT="
   laguna.env` and compare against current `laguna-env.example` — add or
   correct any missing/stale lines.
2. **A fresh `qwen` session, not a retry** — `settings.json` is only read
   at process startup (see the stream-lifetime-cap entry above for the
   full explanation of this same gotcha in a different context).

**On macOS**: if you hand-write a `sed -i '...'` fix and it silently
doesn't take effect, check whether you used Linux/GNU `sed -i` syntax —
macOS ships BSD `sed`, which requires a backup-suffix argument
(`sed -i '' '...'` or `sed -i.bak '...'`) and otherwise errors out or
behaves unexpectedly without one. `serve-laguna.sh` itself doesn't use
`sed` (it uses `jq` for all JSON/config manipulation, which is portable),
but any one-off manual fix command using `sed -i` needs the
macOS-compatible form.

## `connect ECONNREFUSED 127.0.0.1:8000` on a client-only machine (e.g. a laptop)

The client's `qwen` process is pointed at `localhost` instead of this
server's real LAN address — usually because `SERVER_HOST` was never set
(or got lost) in that machine's local `laguna.env` before running
`wire-qwen-code`. Confirmed as a real incident: the server was healthy the
whole time, so this is easy to misdiagnose as a server-side outage.

**Root cause**: `laguna.env` is gitignored on every machine — `git pull`
only updates the tracked `laguna-env.example` template, never a machine's
own real `laguna.env`. If a client clone's `laguna.env` predates
`SERVER_HOST` being set, or the file was regenerated fresh, `SERVER_HOST`
silently defaults to `127.0.0.1`.

Fix, on the client machine:

```bash
jq -r '.modelProviders.openai[] | select(.id=="laguna-gfx1151") | .baseUrl' ~/.qwen/settings.json   # confirm the actual bug
grep -n SERVER_HOST laguna.env                                                                       # confirm the cause
echo 'SERVER_HOST="<server-lan-ip>"' >> laguna.env   # or edit the existing line
./serve-laguna.sh wire-qwen-code
```

Then start a fresh `qwen` session. Also worth checking `curl
http://<server-lan-ip>:8000/health` from the client to confirm basic
network reachability if the above doesn't explain it.

## `SPEC_TYPE=""` in `laguna.env` doesn't actually disable speculative decoding

Historical, already fixed — worth knowing the mechanism if you're
extending this script and add a variable where "explicitly empty" needs
to mean something different from "not set": bash's `${VAR:-default}`
(colon-dash) treats an explicit empty string the *same* as unset, silently
falling back to the default anyway. An earlier version of `load_env()`
used exactly this pattern for `SPEC_TYPE` with a non-empty fallback
default, which meant an explicit `SPEC_TYPE=""` was being silently
overridden back to that default — caught only because the server's own
`Speculative:` startup banner line was inconsistent with what the env file
actually said. If you ever give a variable here a non-empty default and
need "explicitly empty" to stick, use `${VAR=default}` (bare `=`, no
colon) instead, which only fills in the default when the variable is
truly unset.

## Poolside's `llama.cpp` fork (branch `laguna`), DFlash speculative decoding hangs

`"dflash requires ctx_other to be set (this warning is normal during
memory fitting)"` followed by the process hanging (not crashing, not
progressing — `pgrep` shows it alive, GPU memory stays near-idle,
indefinitely). Reproduced with and without `-fa`. This is a real bug in
that fork's draft-model memory-measurement path for this exact
model/hardware combo, not a flag/config mistake — confirmed by reading
the fork's own source (`src/llama-context.cpp`, `src/models/dflash.cpp`):
`ctx_other` is how DFlash's draft model shares the target model's
embedding/output-head weights, and the server isn't recovering from the
expected-during-fitting exception the way the "this warning is normal"
message implies it should. Not adopted — the model runs fine on
**mainline** llama.cpp without speculative decoding (~26 tok/s, see
README "Expected performance"), so this wasn't worth chasing further. If
you want to retry: the target model itself works correctly on this same
fork with plain decoding (only the `-md`/`--spec-type draft-dflash`
combination is broken), so if the fork updates, it may be worth a quick
retest — `git -C vendor/llama.cpp-poolside pull` then rerun the same
command from this entry.

## Adding a new model: things to check first

- **Sharded GGUF models** (multiple `-0000N-of-0000M.gguf` files): set
  `MODEL_FILE` to the first shard, `MODEL_FILE_EXTRA_SHARDS` (comma-
  separated) to the rest — `download` fetches all of them, `serve` only
  needs the first shard's path (llama.cpp auto-detects siblings by
  naming convention). Get real sizes via `curl -sIL <resolve-url> | grep
  -i content-length` per shard — don't trust a "~XGB" figure from a
  webpage scrape, HF's tree UI is JS-rendered and unreliable to scrape.
- **`UBATCH_SIZE`/`BATCH_SIZE` are not free to leave at a previous
  model's value.** Compute-buffer memory scales with ubatch size roughly
  independent of context length — a large value tuned for one model's
  needs can cost several GB of GPU memory that a different model might
  not need to spend at all. Test whether the new model actually needs a
  large ubatch (a needle-in-haystack prompt longer than the smaller
  candidate value, checking for exact retrieval) before assuming a
  previous model's setting still applies.
- **Check the model's own architecture file in your local llama.cpp
  checkout** (`vendor/llama.cpp/src/models/<name>.cpp`) before assuming
  GPU support is mature: does it use shared helpers like `build_attn`/
  `build_moe_ffn` (mature, backend-complete, low risk) or introduce a
  novel custom op (higher risk of immature-kernel performance problems)?
- **A model's own `--jinja`/tool-calling/reasoning support isn't
  guaranteed**, but in practice every model tried on this box so far has
  needed `--jinja` — check the model card's own serving instructions
  before assuming, but it's a safe first guess.
- **mmproj is optional** — `MMPROJ_FILE=""` for a non-multimodal model;
  `_build_server_args`/`cmd_download` already handle this correctly
  (conditional on the file existing / the variable being non-empty).
- **The quant-benchmarking (`bench`) tooling from an earlier iteration of
  this repo does not exist for the current model.** It was hardcoded to a
  flat single-file quant-naming convention and was removed rather than
  left half-working when this repo's model changed to one shipped as
  sharded multi-file quants. If you want to benchmark alternate quants
  for the current model, that tooling needs to be rebuilt to handle
  per-quant shard sets, not just renamed.

## General: this whole stack is young

Mainline llama.cpp support for Laguna's architecture merged in mid-2026
and is under active bugfixing. Prefer `./serve-laguna.sh update` regularly
over pinning to an old build — but re-run `check` after every update, since
a "latest master" isn't guaranteed stable either.
