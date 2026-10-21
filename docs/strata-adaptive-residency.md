# Strata-inspired adaptive expert residency

Implementation record, 2026-10-20. This extends the experimental Edge0 + Kindi patch for the 18GB Mac with approximately 8GB available to inference. It uses existing weights and adds no training. The existing GiB-based guards remain unchanged; see the [memory-units limitation](research-and-results-20261020.md#target-and-decision).

## Investigation and selected change

Inspected Strata revision `9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4`, especially [expert-cache implementation](https://github.com/Niko1221/Strata/blob/9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4/src/core/expert_cache.cpp), its [profile tests](https://github.com/Niko1221/Strata/blob/9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4/tests/core/expert_profile_save_test.cpp), and [technical documentation](https://github.com/Niko1221/Strata/blob/9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4/docs/DETAILS.md).

Strata ranks experts using residency and usage history, saves ranked profiles, and offers adaptive/background tier changes. Its current [Mac documentation](https://github.com/Niko1221/Strata/blob/9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4/docs/MAC.md) also describes contiguous expert copies and SSD streaming. These observations motivated the policy here; no Strata kernels or source implementations were copied.

Edge0 already had prefetching, incremental staging and some hot-expert counts, but its shared bundle cache used LRU eviction. The incremental change is a demand-frequency eviction policy plus restart hints inside that same cache. We did not add background model loads: existing workers and admission rules remain responsible for loading.

## Behavior

The new `expert_residency.hpp` provides the actual generic cache used by `ExpertIo`, so portable tests exercise the eviction implementation used by the engine.

- **Default remains LRU.** Set `expert_io.adaptive_cache` to true to enable the experiment.
- Observe the selected experts consumed in each single-token decode, including consumed Edge0 prerouter selections. Do not count predicted future loads, cache hits, or prefill batches as new demand observations.
- Discount old scores with a half-life of 64 decode observations **per layer**. Layer-local clocks prevent later layers from appearing hotter merely because they execute later. Ties use LRU order.
- Keep the existing 64-slot shared cache and joint payload budget. Under slot or byte pressure, evict the lowest-scoring cached bundle. A newly loaded, rarely used expert can be discarded from the cache while the current request still owns it.
- Cache admission/eviction never changes selected experts, weights or arithmetic. Active consumers retain their shared ownership and budget leases. The policy does not control every other staged or hot-layer allocation.
- Record observations, adaptive evictions, enabled state and loaded profile entries in `e0_stats` under `expert_residency`.

No extra GPU readback or synchronization is introduced. Batched decode records usage after its existing selection/assembly work; CPU-reference paths record the already available selected IDs. Profile statistics are bounded by the model's expert layout (at most 65,536 entries), with a 1 MiB file cap and bounded identity metadata. This small host metadata is outside the managed payload counter and remains subject to process-level telemetry.

## Profiles and restart behavior

Optional engine configuration:

```json
{
  "expert_io": {
    "adaptive_cache": true,
    "profile_path": "/absolute/path/outside-model/edge0-expert-profile.json"
  }
}
```

The parent directory must already exist. A missing profile starts fresh. An incompatible or malformed existing profile rejects engine creation instead of silently importing it. Enabling a profile without adaptive mode is rejected.

Profiles contain layer/expert scores and model metadata, not prompts, generated text, weights or KV/recurrent state. Identity includes the canonical model directory, regular-file names/sizes/mtime values (including adapters and configuration), the layer layout and a runtime format identifier. This is metadata validation, **not a cryptographic content check**; deliberately changed files with preserved metadata can evade it. That would affect cache hints, not supply model weights. Moving or touching the model may require removing the old hint file and starting fresh.

Save happens only on graceful engine destruction after workers drain, and only after new observations. A unique temporary file with mode 0600 is written, fsynced and renamed atomically. Save failure is reported to stderr and does not throw from the destructor. A killed process can lose its newest observations; concurrent engines use last-completed-save semantics rather than merging histories. Destination symlinks and paths inside the model directory are rejected.

Loading a profile **does not preload model tensors**. The first request still pays required SSD loads, but its cache retains experts using the saved rankings. No persistent prompt snapshots or instant warm startup are claimed.

## Run the Mac comparisons

Use a fresh pinned upstream checkout and apply the current patch, following [walkthrough.md](../walkthrough.md). The v0.2 release ZIP predates this change; get the current Git checkout for these commands.

```bash
python3 verify_mac.py   --dylib edge0-kindi/macos/target/engine/libedge0_engine_native.dylib   --model-dir ./model-download/Edge0-35B-A3B-preview   --adaptive-cache --max-new-tokens 64 --timeout 600   --output edge0-kindi-adaptive-result
```

This adds an `adaptive` arm to baseline/io/cached comparisons and reverses arm order on the second trial. If `--packed-experts-dir` is supplied, compare adaptive against packed; otherwise compare it against cached. Both pairs have matching batching, prompt-cache and storage settings.

For persistence, run a separate experiment with `--profile-restart`. Each trial gets a new profile path: a fresh adaptive worker saves its usage, then a new `profiled` worker loads it. The dependent pair always runs in that order, so the restarted result has an OS-cache/order advantage and is **not** a clean cold-start speed comparison. No model tensors or prompt state are retained across those workers. The normal memory watchdog remains active.

The checker requires identical nonempty greedy tokens, expected prompt-cache hits/misses, memory telemetry, positive adaptive observations, zero imported entries in the fresh arm and positive imported entries in the restart arm. Include `--reference-dylib` to compare against the unmodified pinned upstream engine. A separate longer coding workload is still needed to evaluate useful speed and quality.

## Validation and limits

Portable checks cover default LRU behavior, frequency retention, adaptation after a workload shift, layer-local decay, concurrent reads/eviction/observations, live budget leases, profile round trips, no eager preload, no writes without new observations, identity/layout/version rejection, malformed/duplicate/out-of-range scores, oversized/truncated files, and private output permissions. Six Python timing/comparison tests include rejecting absent adaptive/restart telemetry.

On one deterministic synthetic trace with recurring hot experts, intervening scans and a workload shift, misses were **1,200 for LRU versus 992 for adaptive** (17.3% fewer). This is a policy fixture, not an inference benchmark or SSD measurement. It does not imply 17.3% faster decoding. Portable sanitizer tests, native syntax checks and fresh-checkout patch validation are recorded in [residency-validation.json](../residency-validation.json).

Still unverified: Metal build/link, model-token parity on the Mac, whole-process 8GB fit, end-to-end latency improvement, and coding-task outcomes. No background inference job was launched on the development workstation. The new policy stays opt-in until these checks justify making it the default.
