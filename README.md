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
| [llama.cpp#20354](https://github.com/ggml-org/llama.cpp/issues/20354) | Gated-DeltaNet fused kernel runs on GPU on gfx1151 but performs no better than CPU fallback | None possible — this is a performance ceiling, not a correctness bug. Expect **~12 tokens/sec**, not MI210-class numbers |
| [llama.cpp#24437](https://github.com/ggml-org/llama.cpp/issues/24437) | `GGML_HIP_ROCWMMA_FATTN=ON` causes up to -41% prefill throughput on gfx1151 at 8K+ context, worsening with context length | Build compiles this flag **OFF** — a deliberate divergence from some "known-good Strix Halo" community recipes that set it ON |
| [lemonade-sdk#3160](https://github.com/lemonade-sdk/lemonade/issues/3160) | Progressive generation corruption under sustained/concurrent load on ROCm-nightly gfx1151, recovers only on full reload | `PARALLEL=1` (single-slot serving) by default; use `restart` if output degrades, or `install-watchdog` for automated selftest-gated restarts |

## Expected performance

Realistically in the ballpark of **~7-12 tokens/sec** on this hybrid
architecture on gfx1151 (llama.cpp#20354 — the GPU kernel path doesn't beat
CPU-fallback speed due to RDNA register-pressure/tuning gaps). Measured on
this exact machine at Q8_0: prompt processing ~200-300 tok/s, generation a
very consistent **~7.4 tok/s**. This is not comparable to dedicated-HBM
datacenter cards (MI210/MI300-class). Capacity (96GB unified memory) is not
the bottleneck here — kernel maturity on this specific GPU architecture is.

**If you're using `qwen-code` specifically**: its full agentic mode sends a
large system/tool-definition prompt (measured ~8,000-20,000+ tokens on this
setup, depending on loaded skills/tools) and can make several sequential
LLM round-trips per user turn (tool calls, reasoning steps), each
reprocessing a large chunk of that context with only partial cache reuse.
Combined with ~7.4 tok/s generation, a single interactive request can
realistically take **several minutes** end-to-end — this was confirmed
directly (a trivial "reply with one word" prompt took multiple sequential
~40-80s round-trips before finishing). This is expected behavior on this
hardware, not a hang — watch `./serve-qwen38.sh status` or tail
`logs/qwen38.log` to confirm it's actively generating. `qwen --bare` skips
auto-discovery/tool loading for a lighter, faster session if you don't need
the full agentic toolset.

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
57344 vs. 73728) — the gap is the safety margin. If you change either
value, rerun `wire-qwen-code` (on every machine running `qwen-code`
against this server, laptops included) so the client's number stays in
sync.

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
