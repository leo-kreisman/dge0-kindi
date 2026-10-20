#!/usr/bin/env bash
set -euo pipefail
# Single compiler job; no model, CUDA, server or MLX runtime required.
checkout=${1:?usage: test.sh PATCHED_EDGE0_CHECKOUT}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir="$checkout/macos/crates/engine-native/src"
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/edge0-kindi-test.XXXXXX")
trap 'rm -rf -- "$build_dir"' EXIT
"${CXX:-c++}" -std=c++20 -O1 -g -pthread \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I"$checkout/macos/third_party/mlx" -I"$source_dir" \
  "$script_dir/test_native.cpp" -o "$build_dir/test"
"$build_dir/test"
