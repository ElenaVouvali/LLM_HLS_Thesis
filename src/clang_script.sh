#!/usr/bin/env bash
set -euo pipefail

name="${1:?missing <name>}"
workdir="${2:?missing <workdir>}"
mode="${3:-}"

cd "$workdir"

src_c="${name}.c"
src_cpp="${name}.cpp"

# Decide which source file to use based on what exists
if [[ -f "$src_cpp" && -f "$src_c" ]]; then
  echo "Error: both $src_c and $src_cpp exist in $workdir, can't decide which to compile." >&2
  exit 1
elif [[ -f "$src_cpp" ]]; then
  src="$src_cpp"
  std_flag="-std=c++11"
elif [[ -f "$src_c" ]]; then
  src="$src_c"
  std_flag="-std=c11"
else
  echo "Error: neither $src_c nor $src_cpp found in $workdir" >&2
  exit 1
fi

LLVM10_HOME="${LLVM10_HOME:-$HOME/tools/llvm10/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04}"

CLANG_BIN="${CLANG_BIN:-}"
if [[ -z "$CLANG_BIN" ]]; then
  if command -v clang-10 >/dev/null 2>&1; then
    CLANG_BIN="$(command -v clang-10)"
  elif [[ -x /usr/lib/llvm-10/bin/clang ]]; then
    CLANG_BIN="/usr/lib/llvm-10/bin/clang"
  elif [[ -x "$LLVM10_HOME/bin/clang" ]]; then
    CLANG_BIN="$LLVM10_HOME/bin/clang"
  elif command -v clang >/dev/null 2>&1; then
    CLANG_BIN="$(command -v clang)"
  fi
fi

[[ -x "${CLANG_BIN:-}" ]] || { echo "Error: no usable clang found. Set CLANG_BIN or install clang-10." >&2; exit 127; }

GCC_VER="$(gcc -dumpversion || echo 12)"

EXTRA_INC=(
  -isystem "/usr/lib/gcc/x86_64-linux-gnu/${GCC_VER}/include"
  -isystem "/usr/include/c++/${GCC_VER}"
  -isystem "/usr/include/x86_64-linux-gnu/c++/${GCC_VER}"
  -isystem "/usr/include"
)

TOOLCHAIN=(--gcc-toolchain=/usr)

FLAGS=(-emit-llvm -fno-discard-value-names -S -c "$std_flag")
[[ "$mode" == "multi_modality" ]] && FLAGS+=(-g)

set -x
"$CLANG_BIN" "${FLAGS[@]}" "${TOOLCHAIN[@]}" "${EXTRA_INC[@]}" "$src" -o "${name}.ll"
set +x
echo "Wrote: $PWD/${name}.ll"



