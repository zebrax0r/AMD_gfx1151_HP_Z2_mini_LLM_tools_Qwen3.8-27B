# Qwen3.8-27B on AMD Strix Halo (gfx1151)

One-click deployment of **Qwen/Qwen3.8-27B** (Alibaba's Aug-2026 hybrid
Gated-DeltaNet + full-attention dense VLM) via **llama.cpp / HIP**, serving
an OpenAI-compatible endpoint for [qwen-code](https://github.com/QwenLM/qwen-code)
or any other OpenAI-compatible client.

Target hardware: an AMD Strix Halo APU (GPU arch **gfx1151**, e.g. Ryzen AI
Max/Max+ 300 series), **96GB unified memory**, no dedicated VRAM.

This is the sibling of [AMD_MI210_Bunya_LLM_tools_Qwen3.8-27B](https://github.com/zebrax0r/AMD_MI210_Bunya_LLM_tools_Qwen3.8-27B),
rebuilt from scratch for a single-APU workstation instead of a SLURM/MI210
cluster. It deliberately does **not** use SGLang or vLLM — see "Why
llama.cpp" below.

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
  this model's hybrid architecture and a real (if imperfect) gfx1151
  backend.

## Known upstream bugs (this exact hardware + model combo)

This repo's defaults exist specifically to mitigate these. They are **not
fixed**, just worked around — expect to rebuild against llama.cpp master
periodically as fixes land upstream.

| Issue | Symptom | This repo's mitigation |
|---|---|---|
| [llama.cpp#28211](https://github.com/ggml-org/llama.cpp/issues/28211) | HIP/gfx1151: prompts longer than `n_ubatch` get **silently wrong logits** (no crash) | `UBATCH_SIZE`/`BATCH_SIZE` default to 8192 (vs stock 512) — raises the ceiling, does not fix the bug |
| [llama.cpp#27623](https://github.com/ggml-org/llama.cpp/issues/27623) | Decode throughput collapses ~25x once KV position exceeds ~80K tokens | `CTX_SIZE` defaults to 73728 (raised from 32768, then 65536 — both too tight for real `qwen-code` sessions, see below); `serve` warns if you raise it above 81920 |
| [llama.cpp#20354](https://github.com/ggml-org/llama.cpp/issues/20354) | Gated-DeltaNet fused kernel runs on GPU on gfx1151 but performs no better than CPU fallback | Base rate is a performance ceiling, not fixable directly (~7-12 tok/s). Largely clawed back by MTP speculative decoding instead (`SPEC_TYPE=draft-mtp`, on by default) — measured **~14-20 tok/s**, ~1.9-2.7x, using the model's own draft head |
| [llama.cpp#24437](https://github.com/ggml-org/llama.cpp/issues/24437) | `GGML_HIP_ROCWMMA_FATTN=ON` causes up to -41% prefill throughput on gfx1151 at 8K+ context, worsening with context length | Build compiles this flag **OFF** — a deliberate divergence from some "known-good Strix Halo" community recipes that set it ON |
| [lemonade-sdk#3160](https://github.com/lemonade-sdk/lemonade/issues/3160) | Progressive generation corruption under sustained/concurrent load on ROCm-nightly gfx1151, recovers only on full reload | `PARALLEL=1` (single-slot serving) by default; use `restart` if output degrades, or `install-watchdog` for automated selftest-gated restarts |

## Expected performance

Base rate is in the ballpark of **~7-12 tokens/sec** on this hybrid
architecture on gfx1151 (llama.cpp#20354 — the GPU kernel path doesn't beat
CPU-fallback speed due to RDNA register-pressure/tuning gaps) — measured on
this exact machine at Q8_0: prompt processing ~200-300 tok/s, generation a
very consistent ~7.4 tok/s without speculative decoding.

**With `SPEC_TYPE=draft-mtp` (on by default)**, measured **~14-20 tok/s** —
roughly 1.9-2.7x — using the model's own MTP "nextn" tensors as a draft
head, no separate draft model needed. Output verified correct (coherent
reasoning, correct final answers, clean `finish_reason: "stop"`) across
multiple test prompts. This matches the flags already used by a
pre-existing Ollama deployment found on this same box. Set `SPEC_TYPE=""`
in `qwen38.env` to disable and fall back to plain decoding if you ever
suspect it's causing an issue.

Either way, this is not comparable to dedicated-HBM datacenter cards
(MI210/MI300-class). Capacity (96GB unified memory) is not the bottleneck
here — kernel maturity on this specific GPU architecture is.

**If you're using `qwen-code` specifically**: its full agentic mode sends a
large system/tool-definition prompt (measured ~8,000-20,000+ tokens on this
setup, depending on loaded skills/tools) and can make several sequential
LLM round-trips per user turn (tool calls, reasoning steps), each
reprocessing a large chunk of that context with only partial cache reuse.
Even with MTP's speedup, a single interactive request can realistically
take **several minutes** end-to-end — this was confirmed directly (a
trivial "reply with one word" prompt took multiple sequential
~40-80s round-trips before finishing). This is expected behavior on this
hardware, not a hang — watch `./serve-qwen38.sh status` or tail
`logs/qwen38.log` to confirm it's actively generating. `qwen --bare` skips
auto-discovery/tool loading for a lighter, faster session if you don't need
the full agentic toolset.

**`wire-qwen-code` also caps a single turn's output** at
`QWEN_CODE_MAX_OUTPUT_TOKENS` (default 10,000) via
`generationConfig.samplingParams.max_tokens`. Without this, `qwen-code`
defaults to the model's *declared* output limit — effectively unbounded.
Confirmed directly: a real turn generated 12,000+ tokens at a healthy,
stable ~13.5 tok/s (no stall or corruption in the server logs) and was
still going when `qwen-code`'s own 15-minute stream-lifetime cap
(`QWEN_STREAM_MAX_LIFETIME_MS`, default 900000ms) killed the connection,
discarding the entire in-flight response. Hitting the 10,000-token cap
instead gives a clean `finish_reason: length` you can ask it to continue
from, rather than losing the whole response to a timeout.

## Benchmarking alternate quants

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
./serve-qwen38.sh download      # fetch ~29.1GB Q8_0 GGUF + mmproj
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

Strix Halo has no fixed VRAM partition — the GPU claims system RAM
dynamically via GTT. Without tuning, you'll be capped well below the
nominal 96GB budget.

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
57344 vs. 73728) — the gap is the safety margin.

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
`model.sessionTokenLimit` (65,536 — a deterministic backstop that blocks
sending the next message outright once the recorded prompt is already over
budget, independent of token-count estimation entirely). These require a
fresh `qwen` session to take effect, not just a Ctrl+Y retry.

**Ordering matters for `sessionTokenLimit`** — it must sit *above*
`CLIENT_CTX_SIZE` (57344), not below it. Confirmed by getting this wrong
first: an initial value of 50,000 fired the hard block *before* proactive
compaction ever got a chance to run, so every session hit a wall requiring
manual `/compress`/`/clear` instead of compacting quietly. 65,536 sits
between `CLIENT_CTX_SIZE` (soft compaction trigger) and the real `CTX_SIZE`
(73,728, hard server limit), restoring it as a true last resort. If you see
`Session token limit exceeded` from `qwen-code` itself, that's this
backstop working as intended, not a bug — `/compress` or `/clear` as it
suggests.

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
