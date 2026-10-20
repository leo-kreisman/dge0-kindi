# Edge0 + Kindi-inspired cache integration

Experimental native **Mac** runtime patch, 2026-10-20. Target: an 18GB unified-memory
Mac with company applications running, reserving at most 8 GiB for inference.
This is inference engineering with existing Edge0 weights. No model training.

## Status

Implemented and portable-tested; **not yet validated with Metal or model weights**.
No tokens/s improvement, coding-quality result, or measured 8 GiB peak is claimed.
The Linux workstation has run only small CPU tests and compilation checks. No
existing model service, desktop driver, boot configuration, or model file changed.

The integration targets Edge0's native Mac engine at revision
`85b2a03b2a05f66a6526ff904e820dfb3509a7d9`, not the older Python checkout under
`edge0/`. The older Python service resets all prompt state at every request;
the native engine already has exact-token prefix indexing and snapshot hooks.
The portable deliverable
is **edge0-kindi.patch**, with the application and verification scripts beside it.
Kindi is a design reference; this patch does not copy its CUDA engine or replace
the MoE model with Bonsai.

## Changes

* One thread-safe **2 GiB managed payload pool** admits expert bundles and prefix
  snapshots before their allocations. In-flight bundles hold reservations. Cache
  eviction does not release a reservation while another consumer holds the object.
  Optional snapshots are skipped if they do not fit; required expert admission
  failure stops the request rather than silently dropping an expert.
* Prefix snapshots also retain a 512 MiB per-cache bound. Admission checks include
  tensor bytes, routing predictions, logits, token-index bytes and metadata allowance.
  Disabled/oversize entries are rejected **before copying** live model state.
* Process resident memory and physical footprint are checked at layer boundaries,
  before hot-layer preparation, and before snapshot/restore. At 7 GiB the engine
  drops evictable caches, clears the MLX allocator cache and stops if pressure
  remains. Target is 8 GiB, leaving 1 GiB for transient growth. MLX allocator cache
  is limited to 128 MiB; its memory guideline is 6 GiB.
* Intermediate snapshots use existing batched-prefill boundaries and every 1024
  actually evaluated generation tokens. This permits shared-prefix reuse when
  prompt tails diverge and preserves multiple sessions in the existing LRU.
  Set `prefix_cache.checkpoint_every` to 0 to disable intermediate snapshots.
* A prefill-intermediate snapshot has no logits. It may resume only a **strictly
  longer** prompt. Exact-full-prompt hits require saved logits.
* Non-final prefill chunks evaluate the backbone without an unused vocabulary
  head. Final-prompt logits are still computed normally.
* Completion snapshots index **evaluated tokens**, not emitted tokens. At a token
  cap the last emitted token has not necessarily entered the model. String stops
  can also trim displayed history without rewinding recurrent state. Both cases
  previously produced incorrect snapshot labels.
* Replacing an existing prefix with a larger snapshot now evicts to its byte
  limit. Invalid index-bucket references across eviction are avoided.
* Expert I/O workers drain before member destruction; failed speculative futures
  are cleaned up. Forward exceptions are returned through the C ABI.
* `e0_stats` exposes joint-budget live/peak/rejection counters and explicitly
  reports `os_enforced: false`. Builds default to one job (`BUILD_JOBS=1`).

## Memory limitations — read before interpreting a result

The managed pool accounts for owned expert bundles and saved snapshots. It does
**not** replace every MLX allocation with a reserving allocator: active KV state,
intermediate graphs, stacked/full-layer weights, file-backed pages, Python/Rust
and driver overhead are observed through the process guard instead. Graphs can
retain underlying array storage after a Bundle handle is released. Consequently,
the managed-payload count is not total live GPU memory.

The 7 GiB guard and a 7.5 GiB parent RSS watchdog provide headroom, but sampling
cannot guarantee an instantaneous 8 GiB maximum. MLX's `set_memory_limit` is itself
a guideline, not an OS hard limit. **Do not advertise this patch as proven to run
under 8 GiB until target-Mac telemetry confirms it across the intended workload.**
Model loading precedes layer checks; the external watchdog covers that interval.
The guard may reject workloads that upstream runs using more memory/page cache.
That is a failed fit test, not a speed result.

## Apply in an isolated checkout

Use the supplied revision; do not patch an unrelated release or active install.
Example, from the folder containing this README:

```bash
git clone https://github.com/Edge0-AI/Edge0.git edge0-kindi
git -C edge0-kindi checkout --detach 85b2a03b2a05f66a6526ff904e820dfb3509a7d9
python3 apply.py edge0-kindi
bash test.sh edge0-kindi
```

The application script checks the revision and patch applicability first, is
idempotent, and starts no model or service. Roll back only these changes with
`git -C edge0-kindi apply --reverse /absolute/path/to/edge0-kindi.patch`.

On the Mac, build one job at a time:

```bash
CARGO_BUILD_JOBS=1 make -C edge0-kindi/macos engine BUILD_JOBS=1
```

Use the target Mac's actual compatible checkpoint and adapters. Do not alter
model architecture identifiers based on old conversation labels.

## Guarded Mac comparison

```bash
python3 verify_mac.py \
  --dylib edge0-kindi/macos/target/engine/libedge0_engine_native.dylib \
  --model-dir /absolute/path/to/Edge0-35B-A3B-preview \
  --tier edge0-35b \
  --output edge0-kindi-mac-result
```

Python 3.9+ is required. This script refuses Linux. It starts one worker at a time,
uses four short generations with about 2200 shared prompt tokens, kills its own
worker at sampled 7.5 GiB RSS or a 300-second per-mode timeout, and writes logs,
output token IDs, per-request latency, cache telemetry and a summary. A timeout
is adjustable with `--timeout`; it is not a reason to remove the memory watchdog.
No OS page-cache flush, package install, or external API is used.

The baseline is the **patched engine with prefix caching disabled**. This isolates
cache effects; it is not a before/after measurement of all changes. For output
parity against unmodified Edge0, also provide `--reference-dylib /path/to/original.dylib`.
All modes run sequentially and retain the normal OS file cache, so SSD cold-start
performance is not established. Fixture prompts are synthetic code comments:
this is an integration/latency check, **not a coding benchmark**.

For tokens/s measurement, add `--max-new-tokens 64 --timeout 600` to the
command above. This runs the same four cases in each mode, sequentially. The
console prints decode tokens/s, time to first token (TTFT), total request time
and generated token count. JSON also records every token's elapsed time and
end-to-end tokens/s. Decode speed is `(N - 1) / (last_token_time - first_token_time)`;
it excludes prompt processing and terminal cleanup. End-to-end speed includes
both. Early EOS is respected; fewer than 32 generated tokens is flagged as a
short sample. A 64-token cap does not guarantee a 64-token answer.

Prefix reuse primarily targets TTFT and total request time. It need not increase
decode tokens/s. Compare the same case across modes and preserve the output
parity and memory checks. Model loading is outside per-request timings; OS file
caches are not cleared. The modes run in a fixed order, so repeated trials with
controlled cache state are needed before claiming a performance improvement.

The Linux host cannot execute this Apple Metal integration. Its earlier CPU-only
Python Edge0 measurements are a different backend, not measurements of this
patch or predictions for the Mac. `python3 test_timing.py` checks only the timing
arithmetic using synthetic timestamps; it does not produce inference speeds.

A pass requires nonempty identical greedy output tokens, a repeated-prompt hit,
a shared-prefix hit after changing the tail, and a miss after changing the prefix.
Observed process memory must stay within 8 GiB and managed peak within its pool.
Reference parity is additionally required when an original library is supplied.
If parity or memory checks fail, preserve the report and leave the installed
working model unchanged. The smoke suite does not cover stochastic distributions,
all context lengths, or all prerouter states. Long-session coding evaluation is
still necessary even after this check passes.

## Still pending

* Actual target-Mac compile/link, inference parity and memory measurements.
* Persistent SSD **prompt snapshots across restarts**. Existing SSD expert
  streaming remains in place. This patch retains prefix snapshots only within
  the worker lifetime. Disk serialization needs model/adapter/runtime identity,
  corruption checks and bounded restore; it should follow verified state parity.
* Tuning the expert-versus-prefix allocation with real agent traces. The 2 GiB
  managed pool / 512 MiB prefix cap are starting limits, not measured optima.
* Long-context quality, multi-turn patch/tests, real workspace latency and peak
  total memory with company applications active.
* Any Metal kernel replacement, ternary model conversion, approximate output
  shortlist or NCM architecture work. None is required by this first integration.

## Sources

* https://github.com/Edge0-AI/Edge0/tree/85b2a03b2a05f66a6526ff904e820dfb3509a7d9/macos
* https://github.com/ahtx/kindi/blob/main/kindi/cache.py
* https://github.com/ahtx/kindi/blob/main/docs/HOW-IT-WORKS.md
* `macos/third_party/mlx/mlx/memory.h` in the pinned Edge0 tree documents the
  advisory nature of MLX memory limits.

## This distribution

This repository contains an experimental patch and verification tools, **not model weights or a prebuilt Mac binary**.
Download the portable ZIP and its SHA-256 checksum from [Releases](https://github.com/leo-kreisman/dge0-kindi/releases/tag/edge0-kindi-mac-pilot-v0.1.0).
Compatible weights are published separately at [Edge0-35B-A3B-preview](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) and [Edge0-8B-A1B-preview](https://huggingface.co/Edge0/Edge0-8B-A1B-preview). Use the complete compatible checkpoint, including its adapters and tokenizer.
See LICENSE and NOTICE for source attribution.
