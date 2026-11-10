#!/bin/sh
# Phase 0 on the phone itself (busybox sh). Copy `ditto` and this file to the phone, plug
# it in, and run:   sh phone-bench.sh        (takes a few minutes; the phone will get warm)
# Paste the whole output into docs/PROGRESS.md.
set -eu
BIN=${BIN:-./ditto}
[ -x "$BIN" ] || { echo "no ditto binary at $BIN (set BIN=...)" >&2; exit 1; }

hottest() {
    max=""
    for f in /sys/class/thermal/thermal_zone*/temp; do
        [ -r "$f" ] || continue
        t=$(cat "$f" 2>/dev/null) || continue
        case "$t" in ''|*[!0-9]*) continue ;; esac
        if [ -z "$max" ] || [ "$t" -gt "$max" ]; then max=$t; fi
    done
    if [ -n "$max" ]; then echo "$((max / 1000)) C"; else echo "unknown"; fi
}

echo "== machine"
uname -srm
grep -m1 -i -E 'hardware|model name' /proc/cpuinfo || true
[ -r /sys/firmware/devicetree/base/model ] && { tr -d '\0' < /sys/firmware/devicetree/base/model; echo; }
grep -m1 Features /proc/cpuinfo || true
grep -E 'MemTotal|MemAvailable' /proc/meminfo
for b in /sys/class/power_supply/*; do
    [ -r "$b/status" ] && echo "$(basename "$b"): $(cat "$b/status") $(cat "$b/capacity" 2>/dev/null || true)"
done
echo "hottest zone before: $(hottest)"

run() {
    echo
    echo "== ditto bench $*"
    "$BIN" bench "$@"
    echo "hottest zone after: $(hottest)"
}

run --model pixart-0.6b --width 512 --height 512 --steps 4 --experts 8 --used 2
run --width 320 --height 192 --frames 17 --experts 8 --used 2
run --width 480 --height 272 --frames 33 --experts 8 --used 2
# the same clip on one thread: how well the cores scale
run --width 320 --height 192 --frames 17 --threads 1 --reps 1
