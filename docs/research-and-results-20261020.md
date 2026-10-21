# SSD MoE inference: research, implementation and measured status

Recorded 2026-10-20. This document is the handoff record for the Edge0 + Kindi-inspired experiment. Start with [the Mac walkthrough](../walkthrough.md) for reproducible commands and [README](../README.md) for implementation details.

## Target and decision

The deployment machine is an **18GB unified-memory Mac**, with company applications already using approximately **6.5–8GB** and at most about **9GB free**. The requested inference ceiling is **8GB total**, including runtime and context, while keeping those applications usable. The 64GB RAM / 32GB VRAM Linux workstation is a development host, not the target or a performance proxy.

The current implementation expresses its thresholds in **GiB**: 8 GiB is 8,589,934,592 bytes, approximately 8.59 decimal GB. This is a documented units discrepancy with a literal 8,000,000,000-byte ceiling; the current test must not be called proof of a strict decimal 8GB limit. A strict decimal-budget release needs adjusted thresholds and corresponding validation.

Use the existing **Edge0-35B-A3B-preview INT4 checkpoint**, SSD expert streaming and bounded caches. No new training, architecture conversion, reduced expert count or re-quantization is part of this integration. “Kindi-inspired” refers to cache/state-management ideas; this is not Kindi's CUDA engine ported to Metal and does not use its Bonsai weights.

The selected engineering direction is contiguous expert storage, fewer unnecessary waits and duplicate loads, reusable prompt state, and shared memory admission. These can address separate sources of latency. They do not establish that a 35B model fits this budget at a useful context length.

## Related implementations and large reported improvements

These are **author-reported measurements**, researched on 2026-10-20. Models, quantization, prompts, hardware, memory accounting and baselines differ. Speedup factors cannot be multiplied or treated as a ranking against Edge0.

| Source | Reported comparison | Relevance and constraint |
| --- | --- | --- |
| [OpenBMB LES](https://github.com/OpenBMB/llama.cpp-omni/blob/main/docs/les/lookahead-expert-scheduling.md) | MiniCPM-V 4.7 35B-A3B on an M4 iPad Pro with 16GB: SSD baseline 2.97 tokens/s; LES 9.81; with mixed quantization 14.92; with DSpark 17.39. Approximately 3.3× and 5.9× versus that baseline. | The closest design reference for lookahead scheduling and storage layout. The later stages also change quantization or add speculative decoding. This breakdown does not demonstrate an enforced 8GB total process limit. |
| [NunSpark](https://github.com/sharma-open-source/Nunspark) | Qwen3-30B-A3B on a 16GB M4 Mac: roughly 0.5 tokens/s naive, 6.9 live and 7.8–8.0 in probes. | Roughly 14–16× over its naive baseline, with a 10GB resident-weight budget plus engine/context overhead. Its 8GB weight-budget result is 6.40 tokens/s, not an 8GB total-memory result. Persistent buffers and compression avoidance are useful ideas. |
| [Slipstream results](https://github.com/Schero94/slipstream/blob/main/bench/RESULTS.md) | A qualified external-SSD Qwen test reports first visible tool latency of 23.36 → 4.73 seconds with reuse, about 4.9×, without swap growth. | Prompt reuse improves time to first useful output. This is not a decode-throughput improvement or proof of 8GB fit; the hardware has more memory. |
| [streamlx](https://github.com/srcterm/streamlx) | On a 32GB M4 Air, Qwen3.6-35B-A3B: 8 GiB expert pool gives about 11GB resident and 7.2 tokens/s; 16 GiB gives about 19GB resident and 15.9 tokens/s, 19.0 warm; all-resident 33.3. | Demonstrates the cache-size tradeoff and expert-major layout. Its smallest listed expert pool already exceeds our total budget once overhead is counted. |
| [SSD MoE](https://github.com/RasoulNik/ssdmoe) | Qwen3.5-35B on a 16GB M4 Air: about 7–8 tokens/s fresh and 10–12 warm. | Explicit reads and persistent prompt caching are relevant. Default K=4 versus original K=8 changes computation; low active-allocator memory is not proof of low total memory including page cache. |
| [FlashMoE paper](https://arxiv.org/html/2601.17063v1) | Up to 2.6× over compared offloading systems; improved cache hit rates. | Uses an RTX 5070 Ti with 16GB VRAM and a host with artificially constrained DRAM, not 8GB unified memory. Its learned cache policy requires training data, outside this project's no-training scope. |
| [Apple: LLM in a Flash](https://machinelearning.apple.com/research/efficient-large-language) | Reports 4–5× CPU and 20–25× GPU acceleration over naive loading, for models exceeding available DRAM. | Supports locality/windowing and contiguous layout as design ideas. These are not measured gains for this modern Qwen/Edge0 runtime. |
| [TUFF](https://github.com/rexmhall09/TUFF) | Lists SSD-streamed Qwen3.6-35B-A3B Q4 and an 8GB Mac minimum. | Worth investigating as a deployment candidate; its feature comparison is not a controlled engine benchmark. No measured advantage over Edge0 is established here. |

NunSpark also reports a useful negative result: speculative decoding can lose when the union of experts needed by draft verification increases storage traffic faster than accepted tokens increase. We therefore do not assume DSpark/MTP can simply be added for a guaranteed gain. See [NunSpark's discussion](https://github.com/sharma-open-source/Nunspark).

The strongest transferable lessons are to measure SSD traffic, batch materialization, preserve exact prompt prefixes, and count **all** memory. They do not justify promising another project's headline speed on this Mac.

## Implemented design

The complete patch is [edge0-kindi.patch](../edge0-kindi.patch), applied to upstream Edge0 revision `85b2a03b2a05f66a6526ff904e820dfb3509a7d9`.

| Component | Implemented behavior | Remaining limit |
| --- | --- | --- |
| Expert layout | Optional per-layer sidecars place each expert's nine original tensor regions contiguously; sizes, timestamps, offsets, lengths and CRC32 are checked. | CPU staging plus existing MLX constructors; no direct-to-Metal allocator. Requires extra SSD space and complete sidecars. |
| Materialization | Batch MLX evaluation, deduplicate requests, consume existing prefetched/in-flight bundles. | Metal execution and numerical parity are not yet tested. |
| Speculative prefetch | Admission uses free managed capacity; a failed speculative reservation cannot evict demand data and is retried as demand if needed. | Prediction does not eliminate mandatory reads on a miss. |
| Memory admission | Experts and saved prompt states share a 2 GiB managed payload pool. References keep reservations live; experts are evicted before prompt snapshots. | This pool is not the complete process/GPU memory footprint. |
| Prefix reuse | Exact evaluated-token checkpoints at prefill boundaries and every 1024 evaluated generation tokens; snapshots capped at 512 MiB. | In-process only. Persistent SSD snapshots across restarts are not implemented. |
| State correctness | Snapshot labels exclude unevaluated emitted tokens; snapshots without logits cannot answer exact-full-prompt hits; cache replacement obeys limits. | Long-session, all-state and stochastic equivalence still need testing. |
| Prefill overlap | Optional resident-expert GPU execution during host cold-expert loading, with evaluation every eight cold experts. | Disabled by default; no measured Mac speed gain. |
| Observability | Cache/budget/I/O counters, per-token timing, process telemetry and a guarded sequential comparison runner. | Requested bytes are not physical SSD bytes; sampled memory can miss transients. |

Other fixes include worker-pool teardown ordering, future cleanup, C ABI exception reporting, overflow-safe mapped bounds and reapplying MLX limits after model construction. Intermediate prefill skips an unused vocabulary head while final logits remain computed normally.

### Memory controls

| Setting | Current value |
| --- | --- |
| Managed experts + saved prompt-state payloads | 2 GiB total |
| Saved prompt-state sublimit | 512 MiB |
| MLX advisory memory guideline | 6 GiB |
| MLX allocator cache | 128 MiB |
| Engine pressure threshold | 7 GiB |
| External worker RSS termination threshold | 7.5 GiB, sampled |
| Reported comparison target | 8 GiB, not an OS-enforced cap |

These limits overlap; they are not independent allowances to add together. Active KV/recurrent state, graphs, temporary arrays, driver/runtime overhead, mapped pages and file-cache pressure require process and system measurements. Reference lifetimes in the graph can outlive cache handles. The worker watchdog also covers loading before engine-layer checks begin. A memory rejection is a failed fit, not a successful slow run.

## Local evidence, including the negative result

Artifacts: [validation summary](../validation-summary.json), [sanitizer log](../sanitizers.log), [raw layer-0 I/O results](../layer0-io.json). The summary's file hashes describe the published v0.2 validation snapshot; documentation has since been updated. Use [SHA256SUMS.files](../SHA256SUMS.files) for the current checkout.

Completed on Linux:

- Portable native cache/stream tests with AddressSanitizer and UndefinedBehaviorSanitizer.
- Six cross-language packed-reader cases: byte equality, corruption, truncation, stale source, invalid layout and overwrite refusal.
- Five timing/comparison tests, including rejection of missing cases, token mismatches, missing memory telemetry and unused packed paths.
- C++ syntax checks for `expertio.cpp`, `abiface.cpp`, `net35.cpp`, and `net8b.cpp` against pinned headers.
- Fresh-checkout patch application, repeat-application idempotence and whitespace checks.
- Real checkpoint layer-0 packing and sampled byte parity.

The real weights were `Edge0/Edge0-35B-A3B-preview`, revision `3fe15cbf2bd5bbcdfd611035e1ac88971ff1cdab`. Layer 0 contains 256 experts of 1,769,472 bytes each in this layout. Seven sampled experts matched all nine source regions exactly. Packing ran under a 512 MiB Linux cgroup, swap disabled, one CPU quota, reduced I/O priority and 15 MiB/s throttling.

| Diagnostic | Source-region pread | Packed pread |
| --- | ---: | ---: |
| Experts per trial | 7 | 7 |
| Bytes requested per trial | 12,386,304 | 12,386,304 |
| Read calls per trial | 63 | 7 |
| Observed elapsed range, four trials | 0.0569–0.0704 s | 0.0673–0.0721 s |

**Fewer read calls did not produce a warm Linux speed improvement.** Both arms include CRC checking and sanitizers. OS cache was uncontrolled; source reads open regions separately while the packed reader retains a file descriptor. The actual upstream path uses mmap, so this diagnostic is not an upstream inference comparison. It establishes byte fidelity and request consolidation only.

Not yet demonstrated: Metal compile/link, target-Mac output parity, tokens/s improvement, coding benchmark gain, or complete inference fitting the requested budget. Earlier CPU-only Python Edge0 results, other models' context-window successes and vendor iPad numbers cannot fill those gaps.

## Context and agent latency

There is no validated maximum context window for this build under the target budget. The smoke prompt is about 2200 tokens. Test 16K and larger only after shorter contexts pass; a runner argument allowing 128K does not prove the model fits or stays accurate there.

Keep stable system instructions and tool schemas at the beginning of requests, preserve the actual rendered history, and observe server cache counters. Changes to early prompt text, model chat templates or retained reasoning can invalidate reuse. Hybrid recurrent state also needs compatible checkpoints; ordinary KV-cache assumptions are insufficient. Prompt compaction may be necessary but can require new prefill.

Record separately: load time, time to first token, decode tokens/s, complete answer time, new versus cached prompt tokens, logical I/O, observed memory/physical footprint, system swap growth, and task correctness. Prefix caching mainly reduces repeated prefill. It does not inherently increase decode speed. CodeGraph/retrieval can reduce input size, but it is a separate agent-layer experiment and is not implemented by this patch.

## Release and next acceptance gates

The [v0.2 runtime release](https://github.com/leo-kreisman/dge0-kindi/releases/tag/edge0-kindi-mac-pilot-v0.2.0) contains the patch, scripts and evidence, not a compiled Mac engine. Runtime commit: `9abbb629e1e3c9afd66c64087c47f30978d685eb`.

The [35B split model release](https://github.com/leo-kreisman/dge0-kindi/releases/tag/model-edge0-35b-a3b-int4-v1) is published with **39 uploaded assets**. It preserves the native checkpoint, including auxiliary files; it is not a GGUF conversion. The downloader checks part and reconstructed-file SHA-256 hashes. The uploader's publication receipt and live GitHub asset state were checked on 2026-10-20. This documentation update did not download all assets again or alter either release archive.

Next gates, in order:

1. Build on Apple Silicon and run the guarded short test with company applications active.
2. Compare greedy token output to an unmodified pinned upstream library, then compare baseline/io/cached/packed arms.
3. Resolve the decimal-GB versus GiB limit and validate actual process/system memory, including loading and longer sessions.
4. Measure longer generations and realistic prompts, both with working prefix reuse and deliberate invalidation; keep failed/timeout cases in the report.
5. Run real coding tasks with patch/test outcomes, then increase context gradually. Publish quality and total latency together.
6. Only after these pass, tune cache allocation or add persistent disk prompt-state serialization with model/adapter/runtime identity and corruption checks.

The implementation is ready for that Mac evaluation. It is not yet evidence that this combination is faster than Edge0 or comfortable for daily coding within 8GB.
