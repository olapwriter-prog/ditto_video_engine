#!/usr/bin/env bash
# Build a fully static ARM64 `ditto` for a postmarketOS phone (Alpine, musl).
# Needs only rustup: the linker is Rust's own lld, no cross compiler.
set -euo pipefail
cd "$(dirname "$0")/.."
rustup target add aarch64-unknown-linux-musl >/dev/null
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER=rust-lld \
RUSTFLAGS="-C target-feature=+crt-static -C linker-flavor=ld.lld" \
  cargo build --release --target aarch64-unknown-linux-musl
OUT=target/aarch64-unknown-linux-musl/release/ditto
echo "Built $OUT ($(du -h "$OUT" | cut -f1))."
echo "Next: scp $OUT scripts/phone-bench.sh user@172.16.42.1: && ssh user@172.16.42.1 sh phone-bench.sh"
