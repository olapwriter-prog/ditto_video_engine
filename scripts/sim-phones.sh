#!/usr/bin/env bash
# Run Ditto on emulated phone processors, and compile it for the phone systems that cannot
# be emulated here.
#
# What this is: the whole test suite and the command-line tool, built for ARM and executed
# instruction by instruction under QEMU as the processor classes phones have. It proves the
# code is correct on those processors and picks the right kernel on each.
#
# What this is not: an Android emulator or an iPhone simulator. No Android or iOS system,
# no camera, no GPU, no heat, and no speed: emulated time says nothing about a phone.
#
# Needs: rustup, qemu-user (Debian/Ubuntu: apt install qemu-user).
set -uo pipefail
cd "$(dirname "$0")/.."

# emulated processor | phones with this class of core | kernel rie-core should choose
ARM64_PROFILES=(
  "cortex-a53|2015-2019 and entry-level phones, PinePhone (plain NEON)|neon"
  "cortex-a76|2018-2021 phones: Snapdragon 845-888, Fairphone 4/5, Tensor G1 (dot product)|dotprod"
  "cortex-a710|2022 and later: Snapdragon 8 Gen 1+, Dimensity 9000+, Tensor G2+ (i8mm)|i8mm"
  "max|every instruction the emulator knows, closest to the newest cores|i8mm"
)
ARM32_PROFILES=(
  "cortex-a7|32-bit phones before about 2016|"
  "max|32-bit system on a newer 64-bit processor|"
)
CHECK_TARGETS=(aarch64-linux-android armv7-linux-androideabi x86_64-linux-android aarch64-apple-ios aarch64-apple-ios-sim)

pass=0; fail=0; lines=()
record() { # ok|FAIL, text
  if [ "$1" = ok ]; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
  lines+=("$(printf '%-5s %s' "$1" "$2")"); printf '%-5s %s\n' "$1" "$2"
}

command -v qemu-aarch64 >/dev/null && command -v qemu-arm >/dev/null || { echo "qemu-user is not installed" >&2; exit 1; }
rustup target add aarch64-unknown-linux-musl armv7-unknown-linux-musleabihf "${CHECK_TARGETS[@]}" >/dev/null 2>&1

export RUSTFLAGS="-C target-feature=+crt-static -C linker-flavor=ld.lld"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER=rust-lld
export CARGO_TARGET_ARMV7_UNKNOWN_LINUX_MUSLEABIHF_LINKER=rust-lld

# reference results from this machine, to compare the emulated ones against
RUSTFLAGS="" cargo build --release -q || exit 1
HOST=target/release/ditto
REF=$(mktemp -d); trap 'rm -rf "$REF"' EXIT
PIC=fixtures/browser-capture.png
"$HOST" trace "$PIC" -o "$REF/host.svg" >/dev/null
"$HOST" look --json "$PIC" > "$REF/host.json"
"$HOST" scripts --json "విజయవాడ ౪౦ कि.मी. ४० VIJAYAWADA گُلمَرگ ꯏꯝꯐꯥꯜ ᱨᱟᱺᱪᱤ" > "$REF/host.scripts"

run_profiles() { # target, qemu binary, profiles...
  local target=$1 qemu=$2; shift 2
  local var="CARGO_TARGET_$(echo "$target" | tr 'a-z-' 'A-Z_')_RUNNER"
  echo; echo "== building for $target"
  cargo build --release -q --target "$target" || { record FAIL "$target: build"; return; }
  local bin=target/$target/release/ditto
  for p in "$@"; do
    IFS='|' read -r cpu phones kernel <<<"$p"
    echo; echo "== $target as $cpu: $phones"
    local run="$qemu -cpu $cpu"
    # 1. the whole test suite
    local out
    out=$(env "$var=$run" DITTO_TEST_RUNNER="$run" cargo test --workspace --target "$target" 2>&1)
    local passed failed
    passed=$(echo "$out" | grep -E '^test result:' | sed -E 's/.* ([0-9]+) passed.*/\1/' | paste -sd+ | bc)
    failed=$(echo "$out" | grep -E '^test result:' | sed -E 's/.* ([0-9]+) failed.*/\1/' | paste -sd+ | bc)
    if [ "${failed:-1}" = 0 ] && [ "${passed:-0}" -gt 0 ] && ! echo "$out" | grep -q '^error'; then
      record ok "$cpu ($target): $passed tests pass"
    else
      record FAIL "$cpu ($target): ${passed:-0} passed, ${failed:-?} failed"; echo "$out" | grep -E 'FAILED|panicked|^error' | head -10
    fi
    # 2. the kernel chosen at run time
    local k
    k=$($run "$bin" bench --tokens 64 --text 8 --reps 1 --threads 2 --experts 8 --used 2 2>&1 | sed -n 's/.*kernel \([^,]*\),.*/\1/p')
    if [ -z "$kernel" ]; then
      [ -n "$k" ] && record ok "$cpu: denoiser block runs, kernel '$k'" || record FAIL "$cpu: denoiser block did not run"
    elif echo "$k" | grep -qi "$kernel"; then record ok "$cpu: kernel '$k', as expected for this processor"
    else record FAIL "$cpu: kernel '$k', expected one with '$kernel'"; fi
    $run "$bin" bench-conv --size 64 --reps 1 --threads 2 >/dev/null 2>&1 && record ok "$cpu: convolution stack runs" || record FAIL "$cpu: convolution stack"
    # 3. the same answers as on this machine, byte for byte
    rm -f "$REF/sim.svg"
    $run "$bin" trace "$PIC" -o "$REF/sim.svg" >/dev/null 2>&1
    cmp -s "$REF/host.svg" "$REF/sim.svg" && record ok "$cpu: SVG of a browser capture identical to this machine's" || record FAIL "$cpu: SVG differs"
    $run "$bin" look --json "$PIC" > "$REF/sim.json" 2>/dev/null
    cmp -s "$REF/host.json" "$REF/sim.json" && record ok "$cpu: picture facts identical to this machine's" || record FAIL "$cpu: picture facts differ"
    $run "$bin" scripts --json "విజయవాడ ౪౦ कि.मी. ४० VIJAYAWADA گُلمَرگ ꯏꯝꯐꯥꯜ ᱨᱟᱺᱪᱤ" > "$REF/sim.scripts" 2>/dev/null
    cmp -s "$REF/host.scripts" "$REF/sim.scripts" && record ok "$cpu: scripts and digits identical to this machine's" || record FAIL "$cpu: scripts differ"
  done
}

run_profiles aarch64-unknown-linux-musl qemu-aarch64 "${ARM64_PROFILES[@]}"
run_profiles armv7-unknown-linux-musleabihf qemu-arm "${ARM32_PROFILES[@]}"

echo; echo "== compile only (these systems cannot be run here)"
unset RUSTFLAGS
for t in "${CHECK_TARGETS[@]}"; do
  if out=$(cargo check -q --workspace --target "$t" 2>&1); then record ok "$t: compiles"
  else record FAIL "$t: does not compile"; echo "$out" | grep -E '^error' -A6 | head -20; fi
done

echo; echo "== summary: $pass ok, $fail failed"
printf '%s\n' "${lines[@]}"
[ "$fail" = 0 ]
