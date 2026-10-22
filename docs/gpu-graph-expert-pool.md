# GPU-decided expert slots with an SSD miss path

Implemented 2026-10-21. This is an **opt-in experiment**, not a measured Mac speedup. No training, new quantization, model conversion, driver change or background service is involved.

## What was transferred from the research

The [llama.cpp GPU-resident LRU proposal](https://github.com/ggml-org/llama.cpp/issues/29949) and [implementation c81901a57](https://github.com/ggml-org/llama.cpp/commit/c81901a57) put expert-cache planning and RAM-to-VRAM fills into GPU graph operations. The author reports 9.7 → 37.2 tokens/s against no pool, and 20.1 → 31.2 against an existing cache at a similar budget on a particular Radeon setup. Those are author-reported measurements using pinned host RAM, not SSD reads under an 8GB limit. They are not projections for this project.

Our adaptation uses an original small Metal decision kernel with MLX graph operations. It does **not** import the CUDA/HIP implementation. The target remains the 18GB Apple Silicon Mac with work applications running and an 8GB inference allowance. Existing guards use GiB and do not prove a strict decimal 8GB maximum.

## Implemented execution path

For the 35B batched decoder's permanently routed layers (outside the existing predicted-consumer range), the optional path is:

```text
GPU router IDs
  → Metal: validate IDs/state, protect this step's experts, choose LRU slots
  → small mailbox to CPU: status, missing IDs and destination slots
      ├─ no misses: reuse the existing nine quantized weight arrays
      └─ misses: existing SSD/prefetch loader → MLX stack/scatter fill
  → GPU gather_qmm using the kernel's slot mapping
  → original routing-weight mixing and shared expert
```

The predicted-consumer path already has GPU slot lookup and asynchronous staging; it stays on that path. A predicted consumer temporarily using its live router also keeps its original fallback. Prefill and the 8B engine do not use the new pool.

The GPU selects victims and remaps routes; the CPU does not recompute LRU. All experts selected in a step are protected before choosing victims, including hits that appear later in routing order. Duplicate selections share one slot and one fill. Invalid IDs or corrupt state stop the operation before any fill address is used.

The kernel stores a bounded MRU-to-LRU permutation, avoiding a timestamp counter that can overflow. Slot state is separate for each participating layer. Its policy is LRU; the Strata-inspired adaptive cache remains the loader's separate, optional policy. The benchmark's `graph` arm uses the ordinary loader policy so an adaptive-policy change is not mixed into this comparison.

Routing IDs and scores needed for telemetry and adaptive observations are copied in one batch at the token boundary. Routing order, scores, quantized bytes and expert selection are preserved by design. Full-model numerical parity must still be checked on Metal.

## What this does not remove

**Every participating layer still has one mailbox synchronization, including all-hit steps.** Ordinary Metal compute kernels cannot issue our OS `pread` calls, and the host must learn which SSD requests are needed. This is not a completely host-free graph and not GPU-direct SSD access.

A miss has an additional materialization boundary before publishing the new pool state. Fill uses functional MLX stack/scatter operations and may copy a whole slot tensor. Consequently it can be slower when misses are frequent; it is not an in-place zero-copy fill. The expected opportunity is reuse on hits, avoiding repeated host remapping and weight-stack construction. The tiny single-thread decision kernel itself also has a cost.

Prefix caching and packed expert sidecars remain separate optimizations. This change neither removes compulsory SSD traffic nor adds speculative decoding/MTP. There is no claim of 3.8×, 55%, or any other measured speed gain here.

## Ownership and memory limits

- Default: **disabled**. Set `expert_io.graph_pool_slots` to a nonzero slot count in the engine configuration. The guarded CLI defaults to eight when `--graph-pool` is selected.
- Slots must fit the model's top-k and be at most 32. The 35B CLI permits 4–32; a syntactically valid count may still fail memory admission.
- A **256 MiB pool reservation cap** is shared across participating layers. The same reservations are also charged to the existing **2 GiB joint payload budget**, not added outside it.
- Each new weight generation reserves twice its full pool payload plus metadata allowance: one pool for the result and one for transient fill workspace. Old generations and source bundles remain separately charged while alive.
- Pools allocate on first demand, not by loading every expert at startup. No full model or host expert pool is pinned in RAM.
- Fills publish new immutable array generations. Pending matrix multiplications retain old generation handles and their leases until the decode graph and deferred routing have completed. No explicit host `memcpy` overwrites a live pool.
- Pressure cleanup can clear cached pool entries, but it cannot reclaim generations still leased by a pending decode. Failed admission stops the request; it does not increase the cap or silently bypass the feature.
- Existing 7 GiB engine pressure checks and the 7.5 GiB parent RSS watchdog remain. The existing engine pressure callback continues to run at every decode layer.

MLX/driver overhead, active model state, file cache and other allocations still require process telemetry. Managed reservations alone do not prove the total memory fit. This experiment competes with other caches within the existing budget; its benchmark is a whole-path comparison, not an isolation of synchronization overhead alone.

## Build and test before loading a model

Use the current distribution checkout and a **fresh pinned upstream checkout**, following [the walkthrough](../walkthrough.md). Do not apply this full patch on top of an older pilot patch. Build the native engine first, with one job.

The portable suite includes `test_graph_lru.py`. It extracts the exact Metal integer kernel body and compiles it as C++ under ASan/UBSan. It compares 42,000 decisions with an independent expert-level LRU reference and checks protected hits, duplicates, invalid inputs/state, slot-to-payload mapping and reservation lifetimes. This does not compile Metal or execute MLX.

After building on the Mac, run the small real-Metal probe from this distribution's root:

```bash
clang++ -std=c++20 -O1 \
  -Iedge0-kindi/macos/third_party/mlx \
  -Iedge0-kindi/macos/crates/engine-native/src \
  test_graph_pool_mac.cpp \
  -Ledge0-kindi/macos/target/engine \
  -Wl,-rpath,"$PWD/edge0-kindi/macos/target/engine" \
  -lmlx -o /tmp/edge0-graph-pool-probe
/tmp/edge0-graph-pool-probe
```

Keep `libmlx.dylib` and `mlx.metallib` together in `macos/target/engine`. This probe exercises Metal kernel compilation, planning, fill/gather, a pending reader of an old array, all-hit behavior and invalid-ID rejection using tiny synthetic arrays. It does not load the model. Stop if it fails; preserve the error rather than skipping it.

## Guarded model comparison

After the existing smoke test and the Metal probe pass:

```bash
EDGE0_EXEC=batched python3 verify_mac.py \
  --dylib edge0-kindi/macos/target/engine/libedge0_engine_native.dylib \
  --model-dir ./model-download/Edge0-35B-A3B-preview \
  --tier edge0-35b --graph-pool --graph-pool-slots 8 \
  --trials 1 --max-new-tokens 4 --timeout 600 \
  --output edge0-kindi-graph-result-smoke
```

If that passes, measure longer output with a fresh results directory:

```bash
EDGE0_EXEC=batched python3 verify_mac.py \
  --dylib edge0-kindi/macos/target/engine/libedge0_engine_native.dylib \
  --model-dir ./model-download/Edge0-35B-A3B-preview \
  --tier edge0-35b --graph-pool --graph-pool-slots 8 \
  --trials 2 --max-new-tokens 64 --timeout 600 \
  --output edge0-kindi-graph-result-speed
```

Optional `--packed-experts-dir ./packed-experts` enables the existing sidecars in both the `packed` and `graph` arms. Compare `graph` against `packed` then; without sidecars compare against `cached`. `--adaptive-cache` adds a separate adaptive arm; it does not change the graph arm's LRU policy. Saved usage profiles do not persist the graph pool or its weight buffers.

The runner requires nonempty greedy token parity, prefix reuse checks, measured memory checks, pool activity/fills, no pool admission failures and pool reservations within 256 MiB. A smoke pass need not include an all-hit step. **Before concluding that hit reuse improved speed, inspect `hit_steps` and compare longer identical workloads.** Early EOS can make a nominal 64-token run too short.

If the pool fails admission, retain the failed report and try four slots in a new directory; do not raise the cap. Four slots may thrash and be slower. Disable the experiment if it cannot fit or improve the intended workload. Removing `--graph-pool` restores the existing path; no installed service is changed.

## Measurements to retain

`e0_stats.graph_expert_pool` reports:

| Field | Interpretation |
| --- | --- |
| `enabled`, `slots` | Requested experiment and per-layer capacity. |
| `decisions`, `requested` | Kernel decisions and requested route entries. |
| `fills`, `fill_batches` | Missing experts filled and batches materialized. These need not be physical SSD reads: prefetch/OS cache may supply them. |
| `hit_steps` | Decisions whose complete selected set was already resident. |
| `mailbox_syncs` | Required small GPU-to-host handoffs. |
| `decision_wait_ns` | Wall time around decision execution and mailbox wait; includes preceding queued graph work, **not pure kernel time**. |
| `fill_wait_ns` | Host loading plus graph fill materialization time. |
| `reserved_live_bytes`, `reserved_peak_bytes`, `reserved_limit_bytes` | Conservative pool reservations, including workspace allowance. |
| `budget_failures` | Rejected pool admission attempts. |

Counters in `stats` are cumulative per engine. Each result row also includes `graph_pool_delta` for request-local counter differences. Retain complete result directories, build commit, hardware/OS, memory pressure, tokens, TTFT, decode tokens/s and complete request time. Do not treat a resident-cache microbenchmark as SSD or coding-agent performance.

## Validation status

See [graph-pool-validation.json](../graph-pool-validation.json) and [the portable test log](../graph-pool-tests.log). Linux checks cover the extracted decision body, portable regression tests, Python comparison checks, C++ syntax and clean patch application. **Real Metal execution, full-model parity, target-Mac speed and the total 8GB fit remain unverified.** The small Metal probe and the guarded comparison above are provided to obtain that evidence on the receiving Mac. The CUDA/HIP and Windows backends are not ported by this change.
