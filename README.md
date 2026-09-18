# LLM inference on AMD Strix Halo (gfx1151)

One-click deployment of a local LLM via **llama.cpp / HIP**, serving an
OpenAI-compatible endpoint for [qwen-code](https://github.com/QwenLM/qwen-code)
or any other OpenAI-compatible client.

**Current default: [Poolside Laguna S 2.1](https://huggingface.co/poolside/Laguna-S-2.1)**
(118B total / 8B active MoE, conventional GQA + sliding-window attention,
July 2026). Adopted 2026-09-19, replacing Qwen3.8-27B — see "Model history"
below for the full story and why. Qwen3.8-27B remains available as a
documented, working fallback (see "Rolling back to Qwen3.8-27B").

Target hardware: an AMD Strix Halo APU (GPU arch **gfx1151**, e.g. Ryzen AI
Max/Max+ 300 series), **96GB unified memory**, no dedicated VRAM.

This is the sibling of [AMD_MI210_Bunya_LLM_tools_Qwen3.8-27B](https://github.com/zebrax0r/AMD_MI210_Bunya_LLM_tools_Qwen3.8-27B),
rebuilt from scratch for a single-APU workstation instead of a SLURM/MI210
cluster. It deliberately does **not** use SGLang or vLLM — see "Why
llama.cpp" below.

## Model history

1. **Qwen3.8-27B** (original default) — a hybrid architecture with 48
   Gated-DeltaNet (linear-attention) layers + 16 full-attention layers.
   Worked, but this exotic architecture was the root cause of most pain
   documented in this repo's history: an immature GPU kernel on gfx1151
   (llama.cpp#20354, ~7-12 tok/s base rate, clawed back to ~14-23 tok/s
   only via MTP speculative decoding), a decode-collapse risk at long
   context (llama.cpp#27623), a silent-wrong-logits risk past `n_ubatch`
   (llama.cpp#28211), and `qwen-code` client-side runaway-generation
   incidents compounding the slowness. Quant was tuned (Q8_0 → Q5_K_M via
   the `bench` subcommand) and context/output-limit safety nets were
   built up considerably before the underlying architecture problem was
   ever questioned.
2. **Laguna S 2.1** (current default) — prompted by the user directly
   asking whether Qwen3.8 was simply the wrong model for the hardware.
   Research found the entire current Qwen line (3.5→3.8, including the
   Qwen4 preview) has doubled down on the same hybrid-attention family,
   so a newer Qwen wouldn't help; DeepSeek/GLM's large models use a
   different but similarly-immature mechanism (DSA). Laguna S 2.1 was the
   standout candidate that both avoids exotic-architecture risk *and*
   fits this hardware's memory budget: confirmed by reading its
   `laguna.cpp` in this repo's own llama.cpp checkout that it's built
   entirely from the same shared `build_attn`/`build_moe_ffn` helpers
   dozens of other well-supported models use — no novel op like
   Gated-DeltaNet needed. Measured directly on this exact box: **~26
   tok/s sustained, with NO speculative decoding at all** — already
   faster than Qwen3.8-27B's best result, which needed MTP to get there.
   Poolside's own DFlash speculative decoding was tried and hit a real,
   reproducible bug in their fork (see "Known bugs" below) — not
   adopted, but the un-accelerated baseline was already a clear win.

## Why llama.cpp (not SGLang, not vLLM)

- **SGLang's official AMD support is CDNA-only** (gfx942/gfx950 —
  MI300/MI350-class datacenter cards). There is no gfx1151/RDNA support,
  official or otherwise, worth relying on. The MI355X sibling repo in this
  same family confirms this scoping explicitly.
- **vLLM's official Qwen3.8-27B recipe lists zero AMD hardware** (NVIDIA
  Blackwell and Huawei Ascend only). Unofficial community patches exist for
  a predecessor model on gfx1151 but ran at ~4.2 tokens/sec — worse than
  llama.cpp's already-degraded number on this hardware.
- **llama.cpp** is the only stack with both genuine upstream support for
  these models' architectures and a real gfx1151 backend. Both Qwen3.8-27B
  and Laguna S 2.1 load and serve correctly on mainline llama.cpp; the one
  fork used in this repo's history (Poolside's, for DFlash) hit a real bug
  and was not adopted — see below.

## Known bugs (this exact hardware, by model)

This repo's defaults exist specifically to mitigate these. They are **not
fixed**, just worked around — expect to rebuild against llama.cpp master
periodically as fixes land upstream.

**Laguna S 2.1 (current default):**

| Issue | Symptom | This repo's mitigation |
|---|---|---|
| Poolside `llama.cpp` fork (branch `laguna`), DFlash speculative decoding | Reproducibly hangs during draft-model memory measurement — `"dflash requires ctx_other to be set"` followed by a hang, not a clean failure. Tested with and without `-fa`, same result. A real bug in that fork, not a flag/config issue on our end | Not adopted. `SPEC_TYPE=""` — no speculative decoding for this model. The ~26 tok/s baseline (no spec decoding) already beats Qwen3.8-27B's best result, so this wasn't a blocker |

No other known issues for Laguna S 2.1 on this hardware as of adoption —
its conventional GQA + sliding-window architecture avoids the entire
class of exotic-kernel problems below, which are all specific to
Qwen3.8-27B's hybrid Gated-DeltaNet design.

**Qwen3.8-27B (previous default, kept as documented fallback):**

| Issue | Symptom | This repo's mitigation |
|---|---|---|
| [llama.cpp#28211](https://github.com/ggml-org/llama.cpp/issues/28211) | HIP/gfx1151: prompts longer than `n_ubatch` get **silently wrong logits** (no crash) | `UBATCH_SIZE`/`BATCH_SIZE` set to 8192 (vs stock 512) when running Qwen — raises the ceiling, does not fix the bug. **Laguna does not need this** (verified directly — see "Expected performance") and defaults to 2048 instead, which matters for memory headroom |
| [llama.cpp#27623](https://github.com/ggml-org/llama.cpp/issues/27623) | Upstream-reported ~25x decode collapse past ~80K KV tokens (other hardware/quants; issue still open) | Retested directly on Qwen3.8-27B 2026-09-18 — NOT reproduced (sustained ~12-15 tok/s at ~97K context). `CTX_SIZE` raised to 131072 accordingly; `serve` warns above 163840 (untested territory, not "known bad") |
| [llama.cpp#20354](https://github.com/ggml-org/llama.cpp/issues/20354) | Gated-DeltaNet fused kernel runs on GPU on gfx1151 but performs no better than CPU fallback | Base rate is a performance ceiling, not fixable directly (~7-12 tok/s). Largely clawed back by MTP speculative decoding instead (`SPEC_TYPE=draft-mtp`) — measured **~14-20 tok/s**, ~1.9-2.7x, using the model's own draft head |
| [llama.cpp#24437](https://github.com/ggml-org/llama.cpp/issues/24437) | `GGML_HIP_ROCWMMA_FATTN=ON` causes up to -41% prefill throughput on gfx1151 at 8K+ context, worsening with context length | Build compiles this flag **OFF** for both models — a deliberate divergence from some "known-good Strix Halo" community recipes that set it ON |
| [lemonade-sdk#3160](https://github.com/lemonade-sdk/lemonade/issues/3160) | Progressive generation corruption under sustained/concurrent load on ROCm-nightly gfx1151, recovers only on full reload | `PARALLEL=1` (single-slot serving) by default for both models; use `restart` if output degrades, or `install-watchdog` for automated selftest-gated restarts |

## Expected performance

### Laguna S 2.1 (current default)

Measured directly on this exact box, Q4_K_M, **no speculative decoding**
(DFlash hits a real bug in Poolside's fork — see "Known bugs"): **~26
tok/s sustained** across three separate generations (26.85 / 26.25 / 25.86
tok/s), no degradation over long (3000-token) generations. Prompt
processing similarly strong. Output verified correct: math, tool-calling
(`finish_reason: "tool_calls"` with well-formed arguments), code
generation, and a needle-in-haystack retrieval test at 4088 tokens of
context (exact match, confirming no silent corruption at the ubatch
boundary — see below).

This is already **faster than Qwen3.8-27B's best result**, which needed
MTP speculative decoding just to reach 14-23 tok/s. Capacity was never
the bottleneck on this hardware (96GB unified memory) — kernel maturity
for a given model's architecture is, and Laguna's conventional
attention avoids the whole class of problems Qwen3.8's hybrid
architecture ran into here.

**`UBATCH_SIZE=2048` for Laguna, not Qwen's 8192.** Compute-buffer memory
scales with ubatch size roughly independent of context length — at
`CTX_SIZE=131072`, ubatch=8192 used 81.4GB/82GB GTT (446MB free,
dangerously tight), while ubatch=2048 used only 77.2GB (4.7GB free) for
the *same* context. Verified this doesn't reintroduce Qwen's
silently-wrong-logits risk (llama.cpp#28211) for Laguna specifically: a
4088-token prompt (well past 2048) correctly retrieved an exact marker
string in a needle-in-haystack test, no corruption.

### Qwen3.8-27B (previous default, if rolled back)

Base rate ~7-12 tok/s (llama.cpp#20354 — the GPU kernel path doesn't beat
CPU-fallback speed due to RDNA register-pressure/tuning gaps); measured
~7.4 tok/s at Q8_0 without speculative decoding. With `SPEC_TYPE=draft-mtp`
(the model's own MTP "nextn" tensors as a draft head, no separate draft
model needed): **~14-20 tok/s**, roughly 1.9-2.7x. Default quant was tuned
from Q8_0 to Q5_K_M via `bench` (+17.7% over Q8_0 baseline) — see
"Benchmarking alternate quants". Full history and numbers in this
section's git history if rolling back and want the details.

### If you're using `qwen-code` (applies to either model)

Its full agentic mode sends a large system/tool-definition prompt
(measured ~8,000-20,000+ tokens on this setup, depending on loaded
skills/tools) and can make several sequential LLM round-trips per user
turn (tool calls, reasoning steps), each reprocessing a large chunk of
that context with only partial cache reuse. Both models tested here also
reason at length before answering (verbose `reasoning_content`, often
consuming the whole token budget before reaching final `content`) — this
is a model-behavior trait, not a bug, but it means real interactive
requests take longer than raw tok/s alone suggests. Watch
`./serve-qwen38.sh status` or tail `logs/qwen38.log` to confirm it's
actively generating rather than stalled. `qwen --bare` skips
auto-discovery/tool loading for a lighter, faster session if you don't
need the full agentic toolset.

**`wire-qwen-code` caps a single turn's output** at
`QWEN_CODE_MAX_OUTPUT_TOKENS` (default 10,000) via
`generationConfig.samplingParams.max_tokens`. Without this, `qwen-code`
defaults to the model's *declared* output limit — effectively unbounded.
Confirmed directly (on Qwen3.8-27B, but the underlying `qwen-code`
behavior is model-agnostic): a real turn generated 12,000+ tokens at a
healthy, stable pace (no stall or corruption in the server logs) and was
still going when `qwen-code`'s own 15-minute stream-lifetime cap
(`QWEN_STREAM_MAX_LIFETIME_MS`, default 900000ms) killed the connection,
discarding the entire in-flight response. Hitting the token cap instead
gives a clean `finish_reason: length` you can ask it to continue from,
rather than losing the whole response to a timeout.

## Benchmarking alternate quants

**Note**: this section and the `bench` subcommand's `BENCH_QUANTS`/
`BENCH_ALLOWED_QUANTS`/`BENCH_*_EXPECT_BYTES` config were built and tuned
against Qwen3.8-27B's quant lineup (bartowski's `Q6_K`/`Q5_K_M`-style
naming). The `bench` mechanism itself is model-agnostic — it always tests
against whatever `MODEL_FILE` is currently configured — but those specific
config values would need updating to bench alternate Laguna S 2.1 quants
(different repo, different quant names: `UD-Q3_K_M`/`UD-Q4_K_M`/etc. via
unsloth). Not yet done for Laguna as of adoption.

**Why not FP8/BF8?** Ruled out, not implemented. gfx1151 (RDNA3.5) has
zero native FP8/BF8 matrix-core acceleration — that's a CDNA3/CDNA4/RDNA4
feature only, confirmed via AMD's own ROCm precision-support docs.
Separately, llama.cpp has no mainline GGUF FP8 quant type to load even if
the hardware supported it (PR #10055 has been in unmerged draft status
since Dec 2024). And even in the hypothetical where both existed, FP8 and
our current Q8_0 are both ~8 bits/weight, so there'd be no
memory-bandwidth win either. A smaller integer k-quant is the only real
lever on this hardware.

**`./serve-qwen38.sh bench [quants]`** benchmarks candidate GGUF quants
(default: `BENCH_QUANTS`, currently `Q6_K,Q5_K_M`) against whatever
`MODEL_FILE` is *currently* configured (the baseline — so after adopting a
new default, a future `bench` run automatically benchmarks that as the new
baseline, not a hardcoded `Q8_0`). It uses `_build_server_args` — the exact
same function `serve` itself calls — so every candidate is tested with
production's real flags, MTP speculative decoding included. This matters:
a community report on similar hardware (same MTP-speculative-decoding
approach) found more aggressive quantization (Q4_K_M) made things net
*slower*, not faster, because the smaller/noisier quant's outputs
diverged too far from what the MTP draft head expects
("draft-acceptance collapse") — not independently reproduced on this
exact box, but treated as a real constraint: `BENCH_ALLOWED_QUANTS` is a
hard allowlist, and `bench` refuses (doesn't just warn about) anything
outside it, currently Q6_K/Q6_K_L/Q5_K_M/Q5_K_L/Q5_1/Q5_0.

`bench` downloads any missing candidate files, then sequentially (never
concurrently — `PARALLEL=1` is already a hard constraint here, see
lemonade-sdk#3160) serves each quant, fires a short realistic prompt
(`bench/prompts/short.txt`) and a long-context prompt (built at runtime by
repeating the short prompt until it exceeds `BENCH_LONG_PROMPT_MIN_CHARS`,
matching real measured `qwen-code` prompt sizes) `BENCH_REPEATS` times
each at `temperature=0`, and records `timings.prompt_n/predicted_n/
draft_n/draft_n_accepted` per request to
`logs/bench/bench-<timestamp>.jsonl`. **A full run is 30-90+ minutes** at
this hardware's throughput (3 entries × 2 prompts × 3 repeats by default)
— it runs in the foreground so you can watch progress.

A `trap` guarantees the original production server is restored on exit no
matter what — clean finish, error, or Ctrl-C mid-run (confirmed by hand:
an early version of this had a real bug here — a `local` variable read by
the trap handler went out of scope by the time the trap actually fired,
since `trap ... EXIT` fires at script exit, after `cmd_bench` itself has
already returned, not at the end of the function; fixed by making those
globals instead).

**Decision rule** (per candidate, within the same run only — never against
historical numbers, to avoid conflating a real quant effect with
thermal/driver/build drift): candidate's long-context generation tok/s
must beat baseline's by ≥ `BENCH_MIN_SPEEDUP_PCT` (default 10%), MTP
draft-acceptance ratio must not drop more than `BENCH_MAX_ACCEPTANCE_DROP_PP`
points (default 5) vs. baseline, and every response must have produced
real output (`no_empty`). **`finish_reason` is reported but not a
pass/fail gate** — confirmed by hand: this model reasons at length before
answering (a real production turn ran 12,000+ tokens without reaching a
natural stop, see above), so at a fixed token budget, healthy responses
routinely hit `length` before finishing a thought. Gating on that would
fail every candidate regardless of quality. `bench` prints a recommendation
but does **not** touch `qwen38.env` or auto-switch — adoption is always a
manual, deliberate step (below). **"No candidate passes" is a valid,
useful outcome** — it would mean this box is kernel-overhead-bound rather
than memory-bandwidth-bound, which is itself worth knowing.

### Adopting a bench result

1. Hand-edit `qwen38.env`: `MODEL_FILE` and `MODEL_FILE_EXPECT_BYTES` to
   the winning quant's filename/size (see `qwen38-env.example`'s
   `BENCH_*_EXPECT_BYTES` comments for where those numbers came from).
2. Keep the old `Q8_0` file on disk — don't delete it. Disk isn't a
   constraint (3.5TB free on this box), and it makes rollback a two-line
   edit instead of a multi-minute re-download.
3. `./serve-qwen38.sh restart`, then `selftest`, then a few real hands-on
   `qwen-code` sessions before fully trusting it.
4. Update `qwen38-env.example`'s defaults and this README's "Expected
   performance" section with the real measured numbers (dated, "confirmed
   by hand" style, matching the rest of this doc).
5. Commit, push.

**Rolling back**: revert `qwen38.env`'s two `MODEL_FILE*` lines to the
Q8_0 values, `restart`, `selftest` — no re-download needed since the old
file was deliberately kept.

## Quickstart

```bash
./serve-qwen38.sh init          # dirs + API key
./serve-qwen38.sh probe         # GPU/ROCm/GTT preflight — read the warnings
./serve-qwen38.sh build         # clone+build llama.cpp from latest master
./serve-qwen38.sh download      # fetch ~73.1GB Laguna S 2.1 Q4_K_M (3 shards, no mmproj)
./serve-qwen38.sh check         # bounded smoke-load test
./serve-qwen38.sh serve         # launch + wait for /health, prints connection banner
./serve-qwen38.sh wire-qwen-code   # point qwen-code CLI at this server
```

## Subcommands

See `./serve-qwen38.sh` with no args, or `docs/TROUBLESHOOTING.md`.

## Configuration

Copy `qwen38-env.example` to `qwen38.env` (done automatically on first run)
and edit. Every default is documented inline with which bug it mitigates.

## System prep (manual — do before `build`)

### 1. ROCm

This machine already has a working ROCm install via AMD's dedicated
Strix-Halo/workstation apt channel, confirmed present:

```
X-Repo-Id: amdrocm-stable
Types: deb
URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2604/
Suites: stable
Components: main
```
(`/etc/apt/sources.list.d/amdrocm-stable.sources`, keyring at
`/etc/apt/keyrings/amdrocm.gpg`). It installs gfx1151-specific packages
(`amdrocm-core10.0-gfx1151`, `amdrocm-blas10.0-gfx1151`, etc., apt-versioned
`10.0.0-4`) directly targeting `ubuntu2604` — i.e. Ubuntu 26.04 is a first-
class target on this channel, unlike the general `repo.radeon.com` apt repo
used by older guides. `hipconfig --version` reports the underlying HIP
runtime as `7.15.x` — consistent with ROCm 7.14+ having native gfx1151
support (no `HSA_OVERRIDE_GFX_VERSION` needed). `rocminfo` confirms the
`gfx1151` agent (`AMD Radeon 8060S Graphics`, 40 CUs) is visible.

If you're setting this up on a fresh machine and this repo isn't already
configured, check whether AMD's install docs point you at
`stable.repo.amd.com` for your distro before falling back to
`repo.radeon.com`'s general apt repo (which may lack a component for a very
new Ubuntu release — pin to `noble` in that case) or a
[TheRock](https://github.com/ROCm/TheRock) nightly tarball extracted to a
side-by-side prefix with `ROCM_PATH` set in `qwen38.env` accordingly.

**Do not run `amdgpu-install` / install `amdgpu-dkms`.** The `amdgpu`
kernel module is already loaded (in-tree, mainline). Installing the DKMS
driver package would fight with it — install ROCm's userspace packages
only, as was already done here.

### 2. GTT / kernel memory tuning

**Applied on this box 2026-09-19** (was documented but deferred for weeks
before that — worth doing early, not after you're already blocked on it
by a model that needs the headroom). Strix Halo has no fixed VRAM
partition — the GPU claims system RAM dynamically via GTT. Without
tuning, you'll be capped well below the nominal 96GB budget: this box
sat at a 47GB GTT ceiling (the driver's unconfigured ~50%-of-RAM default)
for the entire Qwen3.8-27B phase of this repo's history, only becoming a
real blocker once Laguna S 2.1's 73GB weights needed more room.

```bash
sudoedit /etc/default/grub
# Add to GRUB_CMDLINE_LINUX_DEFAULT (keep existing params):
#   amdgpu.gttsize=81920 ttm.pages_limit=20971520
sudo update-grub
sudo reboot
```

This reserves ~80GiB of your ~91GiB actual RAM for GPU/GTT use, leaving
~11GiB for the OS and host-side staging buffers. `amdgpu.gttsize` is in
MiB; `ttm.pages_limit` is `GiB_reserved × 262144` (4KiB pages). Recompute
both together if you change the reservation. `./serve-qwen38.sh probe`
checks whether this has been applied.

## qwen-code CLI wiring

```bash
./serve-qwen38.sh wire-qwen-code
```

Writes `~/.qwen/.env` (or a project-local `.qwen/.env` if you set
`QWEN_ENV_TARGET=project` / `QWEN_PROJECT_DIR` in `qwen38.env`) with
`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`. Alternatively, run
`qwen` and use the interactive `/auth` → Custom Provider flow with the same
three values (see the command's output for exact values, redacted here).

**Important**: `qwen-code`'s `~/.qwen/settings.json` (`security.auth`) takes
precedence over `.env` if it already names a provider — writing `.env`
alone can silently have no effect. `wire-qwen-code` patches
`settings.json` too: it backs up the existing file, adds/updates a provider
entry keyed by a stable id (`qwen38-gfx1151`) without deleting any other
providers already configured (e.g. a separate Ollama-based setup — this
box already had one, on port 11434, discovered while building this repo),
and sets that entry as the default (`security.auth`). Safe to rerun.

It also writes `generationConfig.contextWindowSize: $CLIENT_CTX_SIZE` into
that provider entry — note `CLIENT_CTX_SIZE`, not `CTX_SIZE`. Without this
field at all, `qwen-code` assumes the model's *advertised* context
(Qwen3.8's ~1,000,000-token YaRN-extended native context) rather than what
this server is actually configured to serve, and paces its own
auto-compaction against that wrong number — meaning it keeps growing the
conversation until it hits a hard `400 ... exceeds the available context
size` error instead of compacting proactively. Confirmed directly: this
recurred at both 32768 and 65536. Reporting the *exact* real `CTX_SIZE`
still isn't enough either — also confirmed directly: at `CTX_SIZE=65536`
reported exactly, a real session still overshot to 68046 tokens and
hard-failed, because `qwen-code`'s own token counts are estimates and a
single large step can jump past its compaction trigger before compaction
runs. `CLIENT_CTX_SIZE` is deliberately lower than `CTX_SIZE` (currently
98304 vs. 131072) — the gap is the safety margin.

**Even that margin isn't fully reliable** — confirmed directly a second
time: a 16,384-token margin was completely consumed in one turn (74602
tokens sent, 874 over the real hard `CTX_SIZE`). `qwen-code`'s proactive
compaction doesn't reliably trigger before a single large-enough addition.
So `wire-qwen-code` also sets three more (global, not provider-specific)
`settings.json` fields: `tools.truncateToolOutputThreshold` and
`tools.truncateToolOutputLines` (lowered from qwen-code's stock 25,000
chars / 1,000 lines to 8,000 / 300 — this bounds how much a single tool
call can add; the stock default still let a *batch* of several fanned-out
tool calls add up to more than the whole `CLIENT_CTX_SIZE` margin), and
`model.sessionTokenLimit` (114,688 — a deterministic backstop that blocks
sending the next message outright once the recorded prompt is already over
budget, independent of token-count estimation entirely). These require a
fresh `qwen` session to take effect, not just a Ctrl+Y retry.

**Ordering matters for `sessionTokenLimit`** — it must sit *above*
`CLIENT_CTX_SIZE`, not below it. Confirmed by getting this wrong first: an
initial value of 50,000 (below the then-`CLIENT_CTX_SIZE` of 57,344) fired
the hard block *before* proactive compaction ever got a chance to run, so
every session hit a wall requiring manual `/compress`/`/clear` instead of
compacting quietly. Current value (114,688) sits between `CLIENT_CTX_SIZE`
(98,304, soft compaction trigger) and the real `CTX_SIZE` (131,072, hard
server limit), restoring it as a true last resort. If you see
`Session token limit exceeded` from `qwen-code` itself, that's this
backstop working as intended, not a bug — `/compress` or `/clear` as it
suggests.

### The 80K "cliff" was retested and didn't reproduce

All the numbers above were originally scaled around staying safely under
llama.cpp#27623's reported ~80K-token decode-collapse cliff. On
2026-09-18, that assumption was directly retested on this exact
build/quant/config rather than taken on faith: a real ~97,176-token prompt
was sent (comfortably past the reported collapse point), and decode speed
was measured over two separate requests (53 and then 400 generated
tokens, the second reusing a 99.9%-cached prefill). Result: a sustained
**~12-15 tok/s**, right in line with normal short-context throughput — no
collapse, and the output was coherent and on-topic (spot-checked by hand,
not just timed). GPU memory was checked too: ~39GB of the 47GB GTT pool in
use with a 140,032-token ctx-size allocated, comfortable headroom.

**This does not mean #27623 is fixed** — it's still open upstream, was
reported on different hardware/quants, and this was one test scenario
(one prompt shape, Q5_K_M, MTP speculative decoding on). Nothing above
~97K has actually been verified on this box, and the model's native
context tops out at 262,144 — there's real headroom beyond 131072 that
just hasn't been tested yet. `CTX_SIZE` was raised from 73728 to 131072
(with `CLIENT_CTX_SIZE`/`QWEN_SESSION_TOKEN_LIMIT` scaled up
proportionally, preserving the same layered-defense ratios), which is a
large, real improvement for long coding sessions, but treat it as
"verified up to ~97K with margin," not "safe all the way to native
context." Re-verify after any `./serve-qwen38.sh update` picks up a new
llama.cpp build — a kernel change could reintroduce the collapse just as
easily as it could push the safe ceiling higher.

We deliberately did **not** reach for llama-server's `--context-shift`
here, even though it's designed for exactly this (discard old context
instead of hard-failing). Its discard mechanism needs to partially
truncate the KV cache, but this model's Gated-DeltaNet recurrent-state
layers can't be partially truncated the way normal attention KV cache
can — real-world reports describe resulting position-accounting desync on
hybrid/recurrent architectures, not just a performance hit. The risk is
silent corruption, so the fix has to live client-side instead.

If you change `CTX_SIZE`, `CLIENT_CTX_SIZE`, `QWEN_TOOL_OUTPUT_THRESHOLD`,
`QWEN_TOOL_OUTPUT_LINES`, or `QWEN_SESSION_TOKEN_LIMIT`, rerun
`wire-qwen-code` (on every machine running `qwen-code` against this
server, laptops included) so the client's config stays in sync.

## Rolling back to Qwen3.8-27B

Laguna S 2.1 is the current default, but Qwen3.8-27B remains fully
supported as a documented fallback — its weights are still on disk
(`models/Qwen3.8-27B-Q5_K_M.gguf` + mmproj), nothing was deleted.

1. Stop the server: `./serve-qwen38.sh stop`
2. Edit `qwen38.env`:
   - Comment out the active Laguna `MODEL_REPO`/`MODEL_FILE`/
     `MODEL_FILE_EXTRA_SHARDS`/`MMPROJ_FILE`/`MODEL_FILE_EXPECT_BYTES`
     lines in the "Model source" section.
   - Uncomment the "Previous model (Qwen3.8-27B)" block right below them.
   - Set `SPEC_TYPE="draft-mtp"` (Laguna's `SPEC_TYPE=""` has no draft
     head for Qwen to use).
   - Set `UBATCH_SIZE="8192"` and `BATCH_SIZE="8192"` (Laguna's `2048`
     does **not** carry the same llama.cpp#28211 silent-corruption
     protection for Qwen's architecture — this is not optional).
   - Optionally lower `CTX_SIZE`/`CLIENT_CTX_SIZE`/`QWEN_SESSION_TOKEN_LIMIT`
     back toward their Qwen-era values if memory is tight, though 131072
     was independently verified safe for Qwen3.8-27B too (see "Expected
     performance").
3. `./serve-qwen38.sh serve`, then `selftest`.
4. `./serve-qwen38.sh wire-qwen-code` (on every machine pointed at this
   server, laptops included) — `SERVED_MODEL_NAME` needs to switch back
   to `qwen3.8-27b` too, which is also in `qwen38.env`.
5. Fully quit and restart any open `qwen` sessions (config changes here
   only take effect on a fresh session, not a retry).

No re-download needed either direction — both models' weights coexist on
disk. Disk isn't a constraint on this box (3.5TB free as of this repo's
last disk check).

## Client-only setup (a laptop or other machine that doesn't run the server)

This repo is meant to be cloned on both the machine that runs the server
and any client machine that just runs `qwen-code` against it — that keeps
`CTX_SIZE`/`SERVER_HOST`/etc. in sync via `git pull` instead of manually
re-copying config by hand.

On the client machine:

```bash
git clone <this-repo-url>
cd <repo-dir>
cp qwen38-env.example qwen38.env
```

Edit `qwen38.env`: set `SERVER_HOST` to the server machine's LAN IP or
hostname (leave everything else — `PORT`, `CTX_SIZE`, etc. — matching the
server's actual config; `git pull` keeps the *example* defaults in sync,
but your local `qwen38.env` won't auto-update, so re-diff it against
`qwen38-env.example` after a pull if the server's settings changed).

Then fetch a copy of the API key — **don't paste it through a chat session
with an AI assistant to relay it**, that happened twice while building this
repo and both times the key had to be rotated as a result:

```bash
mkdir -p .secrets
ssh <user>@<server-host> cat <path-to-repo-on-server>/.secrets/api_key > .secrets/api_key
chmod 600 .secrets/api_key
```

Do **not** run `./serve-qwen38.sh init` on the client — that generates a
*new*, different key, which won't match what the server actually accepts.
Only `wire-qwen-code` needs to run here:

```bash
./serve-qwen38.sh wire-qwen-code
```

No `build`/`download` needed on the client — `wire-qwen-code` only reads
config and the key file, it doesn't touch the model or the llama.cpp
checkout.

## Troubleshooting

See `docs/TROUBLESHOOTING.md`.
