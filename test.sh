#!/usr/bin/env bash
set -euo pipefail
# Single compiler job; no model, CUDA, server or MLX runtime required.
checkout=${1:?usage: test.sh PATCHED_EDGE0_CHECKOUT}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir="$checkout/macos/crates/engine-native/src"
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/edge0-kindi-test.XXXXXX")
trap 'rm -rf -- "$build_dir"' EXIT
# Packed I/O shares the JSON dependency already used by the engine. Accept its
# build-tree copy, a system installation, or an explicit offline include root.
json_include=${NLOHMANN_INCLUDE:-/usr/include}
if [[ ! -f "$json_include/nlohmann/json.hpp" ]]; then
  candidate=$(rg --files --hidden --no-ignore "$checkout/macos/target" 2>/dev/null | rg '/nlohmann/json.hpp$' | head -1 || true)
  if [[ -n "$candidate" ]]; then json_include=$(dirname -- "$(dirname -- "$candidate")"); fi
fi
if [[ ! -f "$json_include/nlohmann/json.hpp" ]]; then
  echo 'Set NLOHMANN_INCLUDE to the directory containing nlohmann/json.hpp (available after the native build).' >&2
  exit 1
fi
"${CXX:-c++}" -std=c++20 -O1 -g -pthread \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I"$json_include" -I"$checkout/macos/third_party/mlx" -I"$source_dir" \
  "$script_dir/test_native.cpp" -o "$build_dir/test"
"$build_dir/test"
"${CXX:-c++}" -std=c++20 -O1 -g -fsanitize=address,undefined \
  -I"$json_include" -I"$source_dir" "$script_dir/test_packed.cpp" -o "$build_dir/test_packed"
python3 "$script_dir/test_pack_experts.py" "$build_dir/test_packed"

"${CXX:-c++}" -std=c++20 -O0 -g1 -pthread -fsanitize=address,undefined \
  -I"$json_include" -I"$source_dir" "$script_dir/test_residency.cpp" -o "$build_dir/test_residency"
"$build_dir/test_residency"
