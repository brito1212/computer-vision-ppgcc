#!/usr/bin/env bash
set -euo pipefail

target="${1:-ex1}"

if ! cmake -S . -B build; then
	echo "CMake cache appears stale. Recreating build metadata..."
	rm -f build/CMakeCache.txt
	rm -rf build/CMakeFiles
	cmake -S . -B build
fi

cmake --build build --target "$target"
./build/"$target"/"$target"