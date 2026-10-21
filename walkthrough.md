# Mac walkthrough: Edge0 + Kindi-inspired SSD/cache pilot

Start with [START_HERE.md](START_HERE.md) for Mac/PC downloads and platform limitations.

Use this with the [research and measured-results record](docs/research-and-results-20261020.md). Target: an 18GB Mac with existing work applications, approximately 8GB available to inference. Current guard thresholds use GiB, and full target-memory compliance remains unverified. No training is involved.

## 1. Obtain the runtime and model

Run on the destination Apple Silicon Mac, with Python 3.9+, curl, Git and the upstream Mac build prerequisites available:

```bash
git clone https://github.com/leo-kreisman/dge0-kindi.git
cd dge0-kindi
shasum -a 256 -c SHA256SUMS.files
python3 download_model.py --directory ./model-download
```

This downloads and verifies the [split 35B release](https://github.com/leo-kreisman/dge0-kindi/releases/tag/model-edge0-35b-a3b-int4-v1), then reconstructs `model-download/Edge0-35B-A3B-preview`. Keep the complete checkpoint directory, including adapters and tokenizer. The native INT4 checkpoint is not GGUF. The command starts no inference.

Allow SSD space for downloaded parts **and** reconstructed files; optional packed sidecars need approximately another copy of expert weights. The downloader keeps its parts. Interrupted curl downloads can resume by rerunning the command; corrupt existing files are rejected rather than silently overwritten. If assembly was forcibly terminated, inspect a remaining `.assembling` file before retrying; the downloader deliberately refuses to overwrite one.

The [v0.2 ZIP](https://github.com/leo-kreisman/dge0-kindi/releases/tag/edge0-kindi-mac-pilot-v0.2.0) is an older runtime snapshot. Use the current Git checkout for adaptive residency and restart profiles; the v0.2 ZIP does not include them.

## 2. Build an isolated engine

The pinned upstream Mac guide specifies **macOS 14+, Apple Silicon M3 or later**.
Earlier Apple chips have not been validated for this distribution. You need:

- Xcode Command Line Tools and a working Metal compiler (`xcrun -f metal`).
- CMake 3.24+, Make, and a C++20 compiler.
- Rust 1.88 via rustup, including Cargo for the native tokenizer static library.
- Python 3.9+, Git, curl and ripgrep (`rg`).

Check before building:

```bash
sw_vers
uname -m
xcode-select -p
xcrun -f metal
cmake --version
rustup --version
cargo --version
python3 --version
rg --version
```

If the command-line tools are missing, use `xcode-select --install`. If `metal`
is unavailable afterward, install the compatible Apple Metal/Xcode toolchain
before continuing. Install CMake and ripgrep through your usual package manager
(for Homebrew users, `brew install cmake ripgrep`). Install Rust through its
standard rustup installer if absent, then `rustup toolchain install 1.88.0`.
Node/JavaScript dependencies are for the upstream app; they are not needed by
this native-library benchmark build. The build may download dependencies.


```bash
git clone https://github.com/Edge0-AI/Edge0.git edge0-kindi
git -C edge0-kindi checkout --detach 85b2a03b2a05f66a6526ff904e820dfb3509a7d9
python3 apply.py edge0-kindi
CARGO_BUILD_JOBS=1 make -C edge0-kindi/macos engine BUILD_JOBS=1
NLOHMANN_INCLUDE="$PWD/edge0-kindi/macos/work/engine-build/_deps/json-src/single_include" bash test.sh edge0-kindi
python3 test_timing.py
```

Build prerequisites are described in the [pinned upstream Mac tree](https://github.com/Edge0-AI/Edge0/tree/85b2a03b2a05f66a6526ff904e820dfb3509a7d9/macos). Native tests need `nlohmann/json.hpp`; use the built dependency or set `NLOHMANN_INCLUDE` to its include root. No prebuilt Mac library is supplied.

For an upstream parity reference, make a **second** checkout at the same revision, leave it unpatched and build its engine. Pass its library as `--reference-dylib /absolute/path/to/reference/libedge0_engine_native.dylib` in subsequent commands. Without that flag, comparisons only establish agreement among patched modes.

## 3. Run the guarded smoke test

```bash
python3 verify_mac.py \
  --dylib edge0-kindi/macos/target/engine/libedge0_engine_native.dylib \
  --model-dir ./model-download/Edge0-35B-A3B-preview \
  --tier edge0-35b --trials 1 \
  --output edge0-kindi-mac-result-smoke
```

The script refuses Linux and runs one worker at a time. Default output is four tokens per request, with a roughly 2200-token synthetic prompt. It exercises first request, exact repetition, changed tail and changed prefix. This checks integration, not coding skill or stable throughput.

The engine responds to sampled pressure at 7 GiB; the parent terminates its worker at sampled 7.5 GiB RSS. These controls cannot guarantee an instantaneous limit or prevent every system-memory spike. Keep other applications in their normal state, observe Activity Monitor memory pressure and swap, and stop the experiment if the desktop becomes unresponsive. Do not increase limits to turn a failed fit into a pass.

## 4. Measure answer speed

After smoke validation succeeds:

```bash
python3 verify_mac.py \
  --dylib edge0-kindi/macos/target/engine/libedge0_engine_native.dylib \
  --model-dir ./model-download/Edge0-35B-A3B-preview \
  --tier edge0-35b --max-new-tokens 64 --timeout 600 \
  --output edge0-kindi-mac-result-speed
```

Two trials run by default, reversing mode order on the second. The timeout covers a worker's mode run, including its four requests. Normal OS file caching is retained; model loading is outside per-request timing.

| Mode | Prefix reuse | Batched materialization | Packed sidecars |
| --- | --- | --- | --- |
| baseline | Off | Off | Off |
| io | Off | On | Off |
| cached | On | On | Off |
| packed, when requested | On | On | On |

Every mode contains the shared-budget and staging fixes. Baseline is therefore not unmodified upstream. A separate reference library is needed for upstream parity.

Read TTFT, decode tokens/s and complete request time together. Decode is `(N - 1) / (last-token time - first-token time)`; end-to-end tokens/s includes prompt processing. Fewer than 32 emitted tokens is flagged as a short sample. Cache reuse must show hits where expected, tokens must match, and memory telemetry must be present. Save failures as well as successes.

## 5. Test optional contiguous expert reads

After model reconstruction on this Mac:

```bash
python3 pack_experts.py \
  --model-dir ./model-download/Edge0-35B-A3B-preview \
  --output ./packed-experts --rate-mib 15
python3 verify_mac.py \
  --dylib edge0-kindi/macos/target/engine/libedge0_engine_native.dylib \
  --model-dir ./model-download/Edge0-35B-A3B-preview \
  --packed-experts-dir ./packed-experts \
  --tier edge0-35b --max-new-tokens 64 --timeout 600 \
  --output edge0-kindi-mac-result-packed
```

This preserves the original weights. It requires complete sidecars for all layers, refuses overwrites and retains a 5 GiB free-space reserve. `--layers 0` creates a limited diagnostic pack, not a usable full-runtime pack. Source timestamps are part of validation, so build sidecars after copying/reconstructing the checkpoint and rebuild if shards are touched or replaced.

Run a separate comparison with `--overlap-prefill` only after the default path passes. Overlap is off by default and has no demonstrated speed advantage yet. Packed loading likewise showed no gain in the Linux warm I/O diagnostic.

## 6. Decide whether to adopt it

Start longer-context testing at `--prompt-tokens 16384` only after short tests fit. There is no validated 128K context claim under this budget. Next run actual coding tasks and compare correct patches, test results, total time, cache reuse and system pressure. Do not infer professional coding quality from the synthetic timing fixture.

Keep the current working model installation available until the pilot passes. This walkthrough does not install a background server or configure OMP/OpenCode. Agent integration follows verified runtime behavior, and persistent prompt snapshots across process restarts remain future work.

## Optional adaptive residency and restart profiles

After the existing checks pass, add `--adaptive-cache` to the speed command to
compare frequency-based expert retention against LRU. Run `--profile-restart`
separately to validate saved usage hints in a fresh process. Use the current Git
checkout and a fresh patched upstream checkout; the v0.2 ZIP predates these options.
See [the detailed experiment record](docs/strata-adaptive-residency.md).

## Copyable adaptive and restart commands

After the smoke test passes, run:

```bash
python3 verify_mac.py \
  --dylib edge0-kindi/macos/target/engine/libedge0_engine_native.dylib \
  --model-dir ./model-download/Edge0-35B-A3B-preview \
  --adaptive-cache --max-new-tokens 64 --timeout 600 \
  --output edge0-kindi-adaptive-result
```

Then validate persistence separately:

```bash
python3 verify_mac.py \
  --dylib edge0-kindi/macos/target/engine/libedge0_engine_native.dylib \
  --model-dir ./model-download/Edge0-35B-A3B-preview \
  --profile-restart --max-new-tokens 64 --timeout 600 \
  --output edge0-kindi-profile-restart-result
```

Add `--packed-experts-dir ./packed-experts` to either only after creating a
complete valid pack. `--profile-restart` implies adaptive mode and manages fresh
profile files inside each trial directory. It saves statistics, then launches a
new worker to read them. Since that dependent pair runs in order and retains
normal OS file caching, its restarted timing is not a cold-SSD comparison.

Inspect `summary.json` in each output directory. A passing report is the first
gate, not proof of all-context correctness or daily coding usability. Do not
replace the existing working model/server until actual target-Mac testing passes.
