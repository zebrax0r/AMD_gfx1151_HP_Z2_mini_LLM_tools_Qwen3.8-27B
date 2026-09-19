# Laguna S 2.1 inference on AMD Strix Halo (gfx1151)

One-click deployment of [Poolside Laguna S 2.1](https://huggingface.co/poolside/Laguna-S-2.1)
via **llama.cpp / HIP**, serving an OpenAI-compatible endpoint for
[qwen-code](https://github.com/QwenLM/qwen-code) or any other
OpenAI-compatible client. (qwen-code is the name of the CLI harness this
repo wires up by default — it's a general-purpose, model-agnostic coding
agent, unrelated to which model is actually serving requests. See
"Client harness" below.)

**Laguna S 2.1**: 118B total / 8B active MoE, conventional GQA +
sliding-window attention, July 2026. Adopted 2026-09-19 after an earlier
model on this hardware (a hybrid-attention architecture) ran into a class
of immature-GPU-kernel problems that Laguna's more conventional attention
mechanism avoids entirely — see "Known bugs" below for what's actually
been hit on this exact box.

Target hardware: an AMD Strix Halo APU (GPU arch **gfx1151**, e.g. Ryzen AI
Max/Max+ 300 series), **96GB unified memory**, no dedicated VRAM.

This is the sibling of [AMD_MI210_Bunya_LLM_tools_Qwen3.8-27B](https://github.com/zebrax0r/AMD_MI210_Bunya_LLM_tools_Qwen3.8-27B),
rebuilt from scratch for a single-APU workstation instead of a SLURM/MI210
cluster. It deliberately does **not** use SGLang or vLLM — see "Why
llama.cpp" below.

## Why llama.cpp (not SGLang, not vLLM)

- **SGLang's official AMD support is CDNA-only** (gfx942/gfx950 —
  MI300/MI350-class datacenter cards). RDNA3.5/gfx1151 enablement exists
  only as an in-progress upstream PR (kernel build support) plus
  unofficial community forks — not something to build production on yet.
- **vLLM doesn't officially support gfx1151 either.** Running it here
  would mean building against ROCm *nightly* plus manual patches (e.g.
  missing `amdsmi` support), with only community toolboxes filling the
  gap. Even where it works, vLLM's real advantage — continuous batching
  across many concurrent requests — doesn't help a single-user,
  single-stream interactive workload like this one; it's built to win at
  high-concurrency serving, not single-session latency.
- **Ollama** (a separate, pre-existing deployment already present on this
  box, port 11434) vendors its own bundled llama.cpp, which lags upstream
  — on Strix Halo specifically it's missing Wave32 flash-attention and
  graphics-queue improvements present in a current standalone llama.cpp
  build, independently measured at ~56% slower on this hardware class.
  Using it here would be a downgrade from a hand-tuned direct build.
- **llama.cpp**, built directly from source with HIP/ROCm flags tuned for
  this exact GPU, is the only stack with both genuine upstream support for
  Laguna's architecture and a mature gfx1151 backend. One fork is used
  elsewhere in this repo's history (Poolside's own, for DFlash speculative
  decoding) and hit a real bug — not adopted, mainline is what's actually
  in production. See "Known bugs" below.

## Known bugs (this exact hardware)

This repo's defaults exist specifically to mitigate these. They are **not
fixed**, just worked around — expect to rebuild against llama.cpp master
periodically as fixes land upstream.

| Issue | Symptom | This repo's mitigation |
|---|---|---|
| Poolside `llama.cpp` fork (branch `laguna`), DFlash speculative decoding | Reproducibly hangs during draft-model memory measurement — `"dflash requires ctx_other to be set"` followed by a hang, not a clean failure. Tested with and without `-fa`, same result. A real bug in that fork's draft-model memory-measurement path, not a flag/config issue on our end (confirmed by reading the fork's own source) | Not adopted. `SPEC_TYPE=""` — no speculative decoding currently. The ~26 tok/s baseline (no spec decoding) is already a strong result, so this isn't a blocker — see "Expected performance" |
| [llama.cpp#28211](https://github.com/ggml-org/llama.cpp/issues/28211) | Upstream report: HIP/gfx1151 can give silently wrong logits on prompts longer than `n_ubatch` (no crash, just bad output) | Verified NOT reproduced on this exact model/build at `UBATCH_SIZE=2048` up to 4088 tokens tested (needle-in-haystack, exact retrieval). If you push well past that, re-verify with the same kind of test before trusting the output |
| [llama.cpp#24437](https://github.com/ggml-org/llama.cpp/issues/24437) | `GGML_HIP_ROCWMMA_FATTN=ON` causes up to -41% prefill throughput on gfx1151 at 8K+ context, worsening with context length | Build compiles this flag **OFF** — a deliberate divergence from some "known-good Strix Halo" community recipes that set it ON |
| [lemonade-sdk#3160](https://github.com/lemonade-sdk/lemonade/issues/3160) | Progressive generation corruption under sustained/concurrent load on ROCm-nightly gfx1151, recovers only on full reload | `PARALLEL=1` (single-slot serving) by default; use `restart` if output degrades, or `install-watchdog` for automated selftest-gated restarts |

## Expected performance

Measured directly on this exact box, Q4_K_M, **no speculative decoding**
(DFlash hits a real bug in Poolside's fork — see above): **~26 tok/s
sustained** across three separate generations (26.85 / 26.25 / 25.86
tok/s), no degradation over long (3000-token) generations. Prompt
processing similarly strong. Output verified correct: math, tool-calling
(`finish_reason: "tool_calls"` with well-formed arguments), code
generation, and a needle-in-haystack retrieval test at 4088 tokens of
context (exact match, confirming no silent corruption at the ubatch
boundary — see "Known bugs" above).

**GPU utilization, confirmed via `amd-smi` during real generation
(2026-09-19)**: 96-100% `GFX_ACTIVITY`, GFX clock averaging 2829MHz
against a 2900MHz ceiling (97.6% of max), edge/GFX temperature 66°C,
zero throttle residency (`PROCHOT`/`SPL`/`SPPT`/`THM_GFX`/`THM_SOC` all
0). Socket power sits around 63-66W — genuinely low, not a sign of
underutilization: the GPU is maxed on clock, fully busy, cool, and
unthrottled the entire time. `GFX_ACTIVITY` is a busy signal (compute
units not idle), not an intensity signal — Laguna activates only 8B of
its 118B total params per token and runs on mature, shared llama.cpp
attention/FFN kernels, so it does genuinely less work per token than a
model with a larger active-parameter count or novel custom kernels would.
There is no free performance being left on the table here: clocks are
already at their ceiling, temperatures are nowhere near thermal
slowdown, and no BIOS/overdrive tweak was found (researched directly —
this box's BIOS has no exposed CPU/GPU clock controls, and Strix Halo's
`amdgpu` driver reports most overdrive/power-cap controls as
"not supported on the given system" even with `ppfeaturemask` unlocked).
The one real, working memory-tuning lever for this hardware class — GTT
allocation boot params — is already applied (see "System prep" below).

**`UBATCH_SIZE=2048`.** GPU compute-buffer memory scales with ubatch size
roughly independent of context length — at `CTX_SIZE=131072` (an earlier,
lower value than the current default), ubatch=8192 used 81.4GB/82GB GTT
(446MB free, dangerously tight), while ubatch=2048 used only 77.2GB
(4.7GB free) for the *same* context. Verified this doesn't risk
silently-wrong-logits behavior (llama.cpp#28211): a 4088-token prompt
(well past 2048) correctly retrieved an exact marker string in a
needle-in-haystack test, no corruption.

**`CTX_SIZE=163840` — found empirically, not assumed.** Directly tested
on this box 2026-09-19: 262144 (2x an earlier 131072 baseline) hard OOMs
during KV-cache allocation; 196608 loads but leaves only ~1.4GB of GTT
free (too tight by this repo's own standard); 163840 loads with a
comfortable ~2.8GB free and was verified correct with a real ~130K-token
needle-in-haystack prompt (exact marker retrieval, `finish_reason: stop`,
14m18s end-to-end at this hardware's prefill speed for a prompt that
long). Laguna's advertised native context (1,048,576) is real capability
the *model* has — it is not evidence this *hardware's GPU memory* can
serve anywhere near that much of it; 163840 is the actual tested ceiling
here, a little over 15% of the advertised figure. `CLIENT_CTX_SIZE`
(122,880) and `QWEN_SESSION_TOKEN_LIMIT` (143,360) are scaled from this
new ceiling using the same ratios as before.

### If you're using `qwen-code`

Its full agentic mode sends a large system/tool-definition prompt
(measured ~8,000-20,000+ tokens on this setup, depending on loaded
skills/tools) and can make several sequential LLM round-trips per user
turn (tool calls, reasoning steps), each reprocessing a large chunk of
that context with only partial cache reuse. The model also reasons at
length before answering (verbose `reasoning_content`, often consuming a
large chunk of the token budget before reaching final `content`) — this
is a model-behavior trait, not a bug, but it means real interactive
requests take longer than raw tok/s alone suggests. Watch
`./serve-laguna.sh status` or tail `logs/laguna.log` to confirm it's
actively generating rather than stalled. `qwen --bare` skips
auto-discovery/tool loading for a lighter, faster session if you don't
need the full agentic toolset.

**`wire-qwen-code` caps a single turn's output** at
`QWEN_CODE_MAX_OUTPUT_TOKENS` (default 10,000) via
`generationConfig.samplingParams.max_tokens`. Without this, `qwen-code`
defaults to the model's *declared* output limit — effectively unbounded.
Confirmed directly: a real turn generated 12,000+ tokens at a healthy,
stable pace (no stall or corruption in the server logs) and was still
going when `qwen-code`'s own 15-minute stream-lifetime cap
(`QWEN_STREAM_MAX_LIFETIME_MS`, default 900000ms) killed the connection,
discarding the entire in-flight response. Hitting the token cap instead
gives a clean `finish_reason: length` you can ask it to continue from,
rather than losing the whole response to a timeout.

### Client harness: qwen-code vs. alternatives

`qwen-code` (QwenLM's fork of Gemini CLI) is what this repo wires up by
default via `wire-qwen-code`, and it works correctly with Laguna —
tool-calling, reasoning/content split, and code generation all confirmed
via `--jinja`. But it's explicitly positioned as the "native" agent for
Qwen-family models specifically, not model-agnostic by design. An
independent comparison running the same 12 coding tasks through local
CLI agents found qwen-code completing 10/12 (one partial, one failed)
vs. Aider 12/12 and OpenCode 11/12 — directionally consistent with
qwen-code carrying assumptions that don't generalize as cleanly to a
non-Qwen backend. **OpenCode** (provider-agnostic, 75+ backends including
any OpenAI-compatible endpoint) is worth trying as a side-by-side
alternative against this same server — no server-side changes needed,
just point it at `http://<SERVER_HOST>:8000/v1`. Not yet set up in this
repo; noted here as a live follow-up, not a recommendation to switch yet.

## Quickstart

```bash
./serve-laguna.sh init          # dirs + API key
./serve-laguna.sh probe         # GPU/ROCm/GTT preflight — read the warnings
./serve-laguna.sh build         # clone+build llama.cpp from latest master
./serve-laguna.sh download      # fetch ~73.1GB Q4_K_M (3 shards, no mmproj)
./serve-laguna.sh check         # bounded smoke-load test
./serve-laguna.sh serve         # launch + wait for /health, prints connection banner
./serve-laguna.sh wire-qwen-code   # point qwen-code CLI at this server
```

## Subcommands

See `./serve-laguna.sh` with no args, or `docs/TROUBLESHOOTING.md`.

## Configuration

Copy `laguna-env.example` to `laguna.env` (done automatically on first run)
and edit. Every default is documented inline with which bug or measurement
it's based on.

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
side-by-side prefix with `ROCM_PATH` set in `laguna.env` accordingly.

**Do not run `amdgpu-install` / install `amdgpu-dkms`.** The `amdgpu`
kernel module is already loaded (in-tree, mainline). Installing the DKMS
driver package would fight with it — install ROCm's userspace packages
only, as was already done here.

### 2. GTT / kernel memory tuning

**Applied on this box 2026-09-19.** Strix Halo has no fixed VRAM
partition — the GPU claims system RAM dynamically via GTT. Without
tuning, you'll be capped well below the nominal 96GB budget (the
driver's unconfigured default is roughly 50% of RAM).

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
both together if you change the reservation. `./serve-laguna.sh probe`
checks whether this has been applied.

This is the real, working memory-tuning lever on this hardware class —
by contrast, direct research into GPU clock/power overclocking on
gfx1151 and this specific machine's BIOS turned up nothing usable: the
BIOS has no exposed CPU/GPU clock/power settings (only third-party,
unofficial "BIOS mod" routes exist, not attempted here), and even with
the `amdgpu.ppfeaturemask` OverDrive-unlock boot param set, this chip's
driver reports `power_cap`/clock-overdrive controls as unsupported. See
"Expected performance" above for the live telemetry confirming there's
no throttling or clock headroom being left unused anyway.

## qwen-code CLI wiring

```bash
./serve-laguna.sh wire-qwen-code
```

Writes `~/.qwen/.env` (or a project-local `.qwen/.env` if you set
`QWEN_ENV_TARGET=project` / `QWEN_PROJECT_DIR` in `laguna.env`) with
`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`. Alternatively, run
`qwen` and use the interactive `/auth` → Custom Provider flow with the same
three values (see the command's output for exact values, redacted here).

**Important**: `qwen-code`'s `~/.qwen/settings.json` (`security.auth`) takes
precedence over `.env` if it already names a provider — writing `.env`
alone can silently have no effect. `wire-qwen-code` patches
`settings.json` too: it backs up the existing file, adds/updates a provider
entry keyed by a stable id (`laguna-gfx1151`) without deleting any other
providers already configured (e.g. a separate Ollama-based setup — this
box already had one, on port 11434, discovered while building this repo),
and sets that entry as the default (`security.auth`). It also removes a
legacy `qwen38-gfx1151` provider entry this repo used before it was
renamed. Safe to rerun.

It also writes `generationConfig.contextWindowSize: $CLIENT_CTX_SIZE` into
that provider entry — note `CLIENT_CTX_SIZE`, not `CTX_SIZE`. Without this
field at all, `qwen-code` assumes the model's *advertised* native context
(Laguna's is 1,048,576) rather than what this server is actually
configured to serve, and paces its own auto-compaction against that wrong
number — meaning it keeps growing the conversation until it hits a hard
`400 ... exceeds the available context size` error instead of compacting
proactively. Reporting the *exact* real `CTX_SIZE` isn't enough either —
confirmed directly on this repo's earlier deployment: `qwen-code`'s own
token counts are estimates, and a single large step can jump past its
compaction trigger before compaction runs. `CLIENT_CTX_SIZE` is
deliberately lower than `CTX_SIZE` (currently 122880 vs. 163840) — the gap
is the safety margin.

**Even that margin isn't fully reliable on its own** — confirmed directly:
a 16,384-token margin was completely consumed in one turn. `qwen-code`'s
proactive compaction doesn't reliably trigger before a single
large-enough addition. So `wire-qwen-code` also sets three more (global,
not provider-specific) `settings.json` fields: `tools.truncateToolOutputThreshold`
and `tools.truncateToolOutputLines` (lowered from qwen-code's stock 25,000
chars / 1,000 lines to 8,000 / 300 — this bounds how much a single tool
call can add; the stock default still let a *batch* of several fanned-out
tool calls add up to more than the whole `CLIENT_CTX_SIZE` margin), and
`model.sessionTokenLimit` (143,360 — a deterministic backstop that blocks
sending the next message outright once the recorded prompt is already over
budget, independent of token-count estimation entirely). These require a
fresh `qwen` session to take effect, not just a Ctrl+Y retry.

**Ordering matters for `sessionTokenLimit`** — it must sit *above*
`CLIENT_CTX_SIZE`, not below it. Confirmed by getting this wrong first on
an earlier deployment: setting it below `CLIENT_CTX_SIZE` fired the hard
block *before* proactive compaction ever got a chance to run, so every
session hit a wall requiring manual `/compress`/`/clear` instead of
compacting quietly. Current value (143,360) sits between `CLIENT_CTX_SIZE`
(122,880, soft compaction trigger) and the real `CTX_SIZE` (163,840, hard
server limit), functioning as a true last resort. If you see
`Session token limit exceeded` from `qwen-code` itself, that's this
backstop working as intended, not a bug — `/compress` or `/clear` as it
suggests.

**This backstop is a value to revisit, not a fact to accept.** Confirmed
directly 2026-09-19: a real session hit `Session token limit exceeded:
115,063 tokens > 114,688 limit` — a genuinely large session, not a bug —
but the real `CTX_SIZE` at the time was 131,072, meaning 16,009 tokens of
already-safe server capacity were sitting unused behind an artificial
backstop that had simply never been revisited since it was first set. If
you hit this error, before just starting a new session, check whether
`QWEN_SESSION_TOKEN_LIMIT` actually has room to move up toward the real
`CTX_SIZE` — it often does.

We deliberately did **not** reach for llama-server's `--context-shift`
here, even though it's designed for exactly this (discard old context
instead of hard-failing). Its discard mechanism needs to partially
truncate the KV cache — safe for Laguna's conventional attention, but this
repo treats that as a case-by-case risk to re-evaluate per architecture
rather than a default to lean on, since some hybrid/recurrent
architectures elsewhere can't have their memory partially truncated
safely at all. The client-side layered defenses above don't depend on
that assumption either way.

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
cp laguna-env.example laguna.env
```

Edit `laguna.env`: set `SERVER_HOST` to the server machine's LAN IP or
hostname (leave everything else — `PORT`, `CTX_SIZE`, etc. — matching the
server's actual config). **`git pull` only updates the tracked
`laguna-env.example` template — it does NOT touch your local, gitignored
`laguna.env`.** After every pull, diff your `laguna.env` against the
current `laguna-env.example` and manually carry over anything that
changed (`SERVER_HOST` is the one value that should differ between the
two — everything else should match). Forgetting this is a real, confirmed
failure mode: it silently leaves `SERVER_HOST` unset/stale, `wire-qwen-code`
then defaults the base URL to `127.0.0.1`, and `qwen-code` fails with
`connect ECONNREFUSED 127.0.0.1:8000` even though the actual server is
healthy — easy to misdiagnose as a server-side problem when it's purely a
client-config gap.

Then fetch a copy of the API key — **don't paste it through a chat session
with an AI assistant to relay it**, that happened twice while building this
repo and both times the key had to be rotated as a result:

```bash
mkdir -p .secrets
ssh <user>@<server-host> cat <path-to-repo-on-server>/.secrets/api_key > .secrets/api_key
chmod 600 .secrets/api_key
```

Do **not** run `./serve-laguna.sh init` on the client — that generates a
*new*, different key, which won't match what the server actually accepts.
Only `wire-qwen-code` needs to run here:

```bash
./serve-laguna.sh wire-qwen-code
```

No `build`/`download` needed on the client — `wire-qwen-code` only reads
config and the key file, it doesn't touch the model or the llama.cpp
checkout.

## Troubleshooting

See `docs/TROUBLESHOOTING.md`.
