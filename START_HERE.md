# Start here: download on a Mac or PC

Updated 2026-10-21. This repository contains the Edge0 + Kindi-inspired native Mac patch, Strata-inspired adaptive expert caching, tests, and a downloader for the split Edge0 35B INT4 checkpoint. Use this Git checkout for the latest code. The older **v0.2 runtime ZIP does not contain adaptive residency or saved usage profiles**.

## Choose your platform

| Machine | What this repository supports | Next step |
| --- | --- | --- |
| Apple Silicon Mac | Build the patched MLX/Metal engine and run guarded comparisons. Target-Mac inference remains unverified. The pinned upstream guide specifies macOS 14+ and M3 or later. | Download below, then follow [the Mac walkthrough](walkthrough.md). |
| Linux PC | Download/verify/reconstruct the model; apply the patch in a source checkout; run portable C++/Python tests. | Follow the Linux testing section below. The Metal inference runner deliberately refuses Linux. |
| Windows PC | Download/verify/reconstruct the model with Python and curl. WSL can run the Linux portable tests. | Follow the PowerShell commands below. Native Windows inference uses a separate upstream backend; this patch is not ported to it. |
| Intel Mac | Download/verify the files, but this Metal/MLX inference build is not a supported target. | Transfer the verified checkpoint to the target Apple Silicon Mac. |

The deployment target remains an **18GB Mac**, with company apps using approximately **6.5–8GB**, and an **8GB inference budget**. The development PC's 64GB RAM and 32GB VRAM are not the deployment allowance. No training is needed or requested. Current guard values use **GiB**, so their 8 GiB target is approximately 8.59 decimal GB; strict decimal 8GB compliance has not been established.

## Download on macOS or Linux

Requirements: Git, Python 3.9+ and curl on PATH. Run in a directory where you want to keep the project:

```bash
git clone https://github.com/leo-kreisman/dge0-kindi.git
cd dge0-kindi
git rev-parse HEAD
```

Verify the checkout files on Mac:

```bash
shasum -a 256 -c SHA256SUMS.files
```

On Linux use:

```bash
sha256sum -c SHA256SUMS.files
```

Then download and reconstruct the model:

```bash
python3 download_model.py --directory ./model-download
```

The model directory will be `model-download/Edge0-35B-A3B-preview`. The downloader verifies each part and each reconstructed file using the release manifest. It starts no model or server. The checkpoint contains **19,716,799,341 bytes** (about 19.72GB / 18.36GiB); retaining both downloaded parts and assembled files needs about **39.44GB**, plus build files and temporary/free-space headroom. Optional packed expert sidecars need additional space roughly equal to expert weights and retain a 5 GiB free-space reserve.

## Download on Windows (PowerShell)

Install Git, Python 3.9+ with the `py` launcher, and ensure `curl.exe` is on PATH. Keep LF line endings so the checkout checksums match:

```powershell
git -c core.autocrlf=false clone https://github.com/leo-kreisman/dge0-kindi.git
Set-Location dge0-kindi
git config core.autocrlf false
git rev-parse HEAD
py -3 --version
curl.exe --version
```

Verify the checkout before downloading:

```powershell
foreach ($line in Get-Content SHA256SUMS.files) {
    $entry = $line -split '  ', 2
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $entry[1] -ErrorAction Stop).Hash.ToLowerInvariant()
    if ($actual -ne $entry[0]) { throw "Checksum mismatch: $($entry[1])" }
}
Write-Host 'Checkout checksums passed'
py -3 download_model.py --directory .\model-download
```

If Python is installed without `py`, use `python` after checking `python --version`. Python invokes the curl executable directly; PowerShell's `curl` alias is not used. Model downloads from Windows have not been exercised by this project; report any platform-specific error instead of converting or modifying the checkpoint to bypass it.

## Resume or transfer to an offline Mac

Rerun the same download command to resume curl transfers and skip files already verified. Do not run two downloaders in the same destination.

To prepare on a PC and move to the Mac, copy:

1. This runtime repository (or a fresh clone on the Mac).
2. `model-download/model-manifest.json`.
3. The entire `model-download/Edge0-35B-A3B-preview/` directory, preserving every config, tokenizer, adapter and prerouter file.

The part files can also be copied if you want to reconstruct on the Mac. With the manifest and complete reconstructed directory already present, offline verification skips the parts and checks the reconstructed files:

```bash
python3 download_model.py --offline --directory ./model-download
```

Do not rename safetensors to GGUF or take just the largest shards. Create packed sidecars **on the destination Mac after transfer**: they validate source timestamps and will reject stale metadata. Expert usage profiles also identify the model's canonical directory and metadata; start a fresh profile after moving the model. These profiles contain cache hints, not model training.

The first native build may fetch compiler dependencies and Rust crates. Downloading the checkpoint alone does not make the build offline-capable. Build once with the required dependencies available before relying on an offline installation.

## Linux / WSL portable tests

These are small CPU correctness tests, not an inference benchmark. No model download is required. You need Git, Python 3.9+, Bash, ripgrep, a C++20 compiler with AddressSanitizer/UndefinedBehaviorSanitizer, and nlohmann JSON headers. On Debian/Ubuntu, the packages are `git curl python3 ripgrep build-essential nlohmann-json3-dev`.

From this repository's root:

```bash
git clone https://github.com/Edge0-AI/Edge0.git edge0-kindi
git -C edge0-kindi checkout --detach 85b2a03b2a05f66a6526ff904e820dfb3509a7d9
python3 apply.py edge0-kindi
NLOHMANN_INCLUDE=/usr/include nice -n 15 bash test.sh edge0-kindi
python3 test_timing.py
```

If your headers are elsewhere, set `NLOHMANN_INCLUDE` to the directory containing `nlohmann/json.hpp`. The native tests use vendored MLX headers but do not execute the MLX/Metal runtime. In WSL, keep the checkout and test temporary files on the Linux filesystem. Do not interpret these test results as Windows or Mac tokens/s.

### If you want actual PC inference

The [pinned upstream Windows guide](https://github.com/Edge0-AI/Edge0/blob/85b2a03b2a05f66a6526ff904e820dfb3509a7d9/windows/README.md) describes its distinct Vulkan/llama.cpp application and conversion path. Its performance and memory requirements differ, and our Mac adaptive/profile patch is not included there. This repository supplies no supported Linux inference launcher for this integration. Porting it requires separate implementation and validation; do not run the Mac benchmark with a substituted library and label it equivalent.

## What to run on the Mac

[walkthrough.md](walkthrough.md) contains the complete sequence:

1. Check prerequisites and build one job at a time in an isolated upstream checkout.
2. Run the short guarded smoke test.
3. Measure TTFT, decode tokens/s and complete request time with longer output.
4. Optionally pack experts and compare that storage path.
5. Compare `--adaptive-cache`, then separately validate `--profile-restart`.
6. Test real coding tasks and longer context only after the short run passes.

Keep `libedge0_engine_native.dylib`, `libmlx.dylib` and `mlx.metallib` together in the generated `macos/target/engine` folder. Point the runner at that native library. This distribution has no prebuilt Mac binary and does not automatically install a service or configure OMP/OpenCode.

## Results and troubleshooting

Each benchmark needs a **new output directory**. It writes `summary.json` and per-trial/per-mode JSON and logs. Keep the entire output folder, the repository commit (`git rev-parse HEAD`), upstream revision, Mac chip/OS, and company-app memory usage when reporting a run.

| Symptom | Meaning / action |
| --- | --- |
| Patch refuses to apply | Check the exact upstream revision. Use a fresh checkout for a newer full patch; do not stack it on an older pilot patch. |
| Missing `nlohmann/json.hpp` | Set `NLOHMANN_INCLUDE`; the Mac walkthrough gives the build-tree path. |
| Missing dylib or Metal library | Complete the native build and preserve all three generated library/resource files together. |
| Download checksum mismatch | Stop that download. Preserve the error; move aside only the named corrupt file after inspection, then retry. Do not disable validation. |
| An `.assembling` file already exists | Check that no downloader is active. A forced termination may leave it behind; move that incomplete file aside before retrying. |
| Memory watchdog or engine budget error | The workload did not fit the current test. Keep its logs and reduce context/workload. Do not raise limits to claim an 8GB pass. |
| Invalid profile | Use a new profile path outside the model directory after changing/moving files. Inspect before deleting an old file. |
| Token mismatch | Treat the experiment as failed; retain both outputs and the settings. |
| Timeout / slow first token | Inspect loading versus prompt work and cache hits. A longer timeout does not improve speed or increase the memory allowance. |
| Output directory already exists | Choose a new `--output` name; the runner intentionally avoids overwriting results. |

A four-token smoke run is too short to establish throughput. Prefix reuse primarily reduces prefill/TTFT. The reported synthetic 17.3% miss reduction is not a measured model-speed gain. Mac inference, output parity, useful context length and full 8GB fit still need target validation.

## Update, rollback and handoff to another developer or agent

For an unmodified distribution checkout, `git pull --ff-only`, verify `SHA256SUMS.files`, and record the new commit. Preserve local edits before updating; do not force-reset them. Apply a new full patch to a **fresh** pinned upstream checkout and rebuild. Keep model downloads and benchmark results separate from source updates.

To roll back, use the previous isolated checkout/library. Reversing the patch changes source only and does not revert a compiled library; rebuild if you choose that route. No background service is installed by the documented benchmark commands.

A receiving developer or agent should read this file, [the walkthrough](walkthrough.md), [the implementation record](docs/strata-adaptive-residency.md), and [the original research/results](docs/research-and-results-20261020.md). Preserve the Mac budget and running work applications. Start with the portable tests; do not retrain weights, substitute models, launch full workloads automatically, or claim PC measurements prove Mac performance. The next missing evidence is a guarded target-Mac run, followed by real coding-task results.
