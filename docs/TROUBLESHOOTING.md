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

## Garbled / wrong output on longer prompts

Check `UBATCH_SIZE` in `qwen38.env`. This is
[llama.cpp#28211](https://github.com/ggml-org/llama.cpp/issues/28211):
on HIP/gfx1151, prompts longer than `n_ubatch` get silently wrong logits
(no crash, no error — just bad output). The default here (8192) covers most
real prompts but is not a fix; if you're pushing very long single prompts,
raise `UBATCH_SIZE`/`BATCH_SIZE` further and re-test, or chunk the prompt.

## Throughput falls off a cliff on long conversations

[llama.cpp#27623](https://github.com/ggml-org/llama.cpp/issues/27623):
decode throughput collapses ~25x once KV position exceeds ~80K tokens on
this hybrid Gated-DeltaNet architecture. Lower `CTX_SIZE`, or start a fresh
conversation before hitting that range. `serve` warns if `CTX_SIZE` is set
above 81920.

## Output quality degrades over a long session, a fresh restart fixes it

[lemonade-sdk#3160](https://github.com/lemonade-sdk/lemonade/issues/3160):
progressive generation corruption under sustained/concurrent load on
ROCm-nightly gfx1151. Run `./serve-qwen38.sh restart`. If you routinely hit
this, install the selftest-gated watchdog:

```bash
./serve-qwen38.sh install-watchdog 2   # selftest every 2h, restarts only on failure
```

This deliberately does **not** restart blindly on a timer — a
selftest-gated restart doesn't interrupt healthy sessions and doubles as a
diagnostic signal worth reporting upstream if it fires often.

## Build succeeds but `serve`/`build` reports a missing required flag

`build`'s verification step checks that `llama-server --help` exposes
`--ubatch-size`, `--parallel`, `--jinja`, `--mmproj`, `--api-key`,
`--ctx-size`. If one is missing, your checkout is likely too old or a flag
was renamed upstream. Rerun `./serve-qwen38.sh update` to pull latest
master. bartowski's GGUF explicitly requires llama.cpp build b10896+.

## `apt update` finds no ROCm package for this Ubuntu release

Ubuntu 26.04 ("resolute") is new enough that ROCm's apt repo may not
publish a `resolute` component yet. Use the `noble` component instead (see
README "System prep") — ROCm's userspace isn't kernel-ABI-tied the way the
driver is. If a needed fix is only in ROCm nightly, use a
[TheRock](https://github.com/ROCm/TheRock) nightly tarball and set
`ROCM_PATH` in `qwen38.env`.

## Server hangs on model load

Confirm `-dio` made it into the assembled `llama-server` argv (visible in
`logs/qwen38.stdout.log` at startup, or check `serve`'s flag-detection
logic against `llama-server --help`). Direct I/O is reported necessary for
models this large (~29GB) to avoid a load hang on some configurations.

## `probe` warns that `amdgpu.gttsize` isn't set

You haven't applied the GRUB kernel parameter from README "System prep §2"
yet (or haven't rebooted since). Without it, GPU memory allocation is
capped well below the nominal 96GB budget, and large-context serving may
fail unexpectedly.

## `[API Error: 400 request (N tokens) exceeds the available context size (CTX_SIZE tokens)]`

Three layers to this, in the order they were actually discovered:

1. **Naive fix (insufficient on its own)**: raise `CTX_SIZE` in `qwen38.env` and `./serve-qwen38.sh restart`. Stay well clear of ~80K (llama.cpp#27623 — see the known-bugs table).
2. **`qwen-code` needs to know the ceiling**: it doesn't know this server's real `CTX_SIZE` unless its provider entry in `settings.json` sets `generationConfig.contextWindowSize`. Without it, `qwen-code` assumes the model's advertised ~1,000,000-token context and won't proactively compact — it just keeps growing until the server hard-rejects the request.
3. **Telling it the *exact* ceiling still isn't enough**: confirmed directly — reporting `contextWindowSize` equal to the real `CTX_SIZE` (65536) still overshot to 68046 tokens and hard-failed anyway. `qwen-code`'s own token counts are estimates (`estimated=true` in its debug log), and a single large step (one big tool result, a large file read) can jump past its compaction trigger before compaction gets a chance to run.

The actual fix is **`CLIENT_CTX_SIZE`**, a separate, deliberately-smaller number than `CTX_SIZE`: the server enforces `CTX_SIZE` (currently 73728), but `wire-qwen-code` tells `qwen-code` a lower `CLIENT_CTX_SIZE` (currently 57344) via `contextWindowSize`. The gap between them is the safety margin that absorbs the estimation slop. `./serve-qwen38.sh wire-qwen-code` writes this automatically; rerun it after changing either value, and on every other machine (laptops included) that points `qwen-code` at this server.

If it recurs even with this margin in place, that's a signal the session's genuine history has grown past a comfortable working set for this hardware — starting a fresh `qwen` session is more sustainable than continuing to raise the ceiling toward the decode-collapse threshold.

## `qwen-code` seems to hang / doesn't respond for a long time

Likely not a hang. `qwen-code`'s full agentic mode sends a large system/
tool-definition prompt (measured 8,000-20,000+ tokens on this setup) and
can make several sequential LLM round-trips per turn (tool calls,
reasoning), each only partially served from cache. At ~7-12 tok/s
generation, a single interactive request can genuinely take several
minutes. Confirm it's actually working via `./serve-qwen38.sh status` or
`tail -f logs/qwen38.log` — you should see `prompt processing` and `n_gen`
lines advancing. Use `qwen --bare` for a lighter/faster session if you
don't need the full tool/skill registry.

## `wire-qwen-code` ran but `qwen` still talks to a different server

Check `~/.qwen/settings.json` — its `security.auth.baseUrl` takes
precedence over the `.env` file. `wire-qwen-code` patches both, but if
you've manually edited `settings.json` since, or have multiple provider
entries, confirm `security.auth` points at `http://127.0.0.1:$PORT/v1`.
This box had a separate, pre-existing Ollama-based Qwen3.8 deployment
(port 11434) already wired in when this repo was first set up —
`wire-qwen-code` preserves that entry in `modelProviders.openai` (so you
can switch back by hand) but changes which one is default.

## General: this whole stack is young

llama.cpp's support for Qwen3.8's hybrid Gated-DeltaNet architecture only
fully landed in the last few weeks before this repo was written, and is
under active weekly bugfixing. Prefer `./serve-qwen38.sh update` regularly
over pinning to an old build — but re-run `check` after every update, since
a "latest master" isn't guaranteed stable either.
