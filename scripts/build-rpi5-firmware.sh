#!/usr/bin/env bash
# Build the Raspberry Pi 5 UEFI firmware (TF-A + EDK2) from pinned source.
#
# NightRun policy (see docs/rpi5-uefi.md): firmware is built from source at
# an exact reviewed commit — never downloaded as a binary release. The fork
# delta vs the archived upstream was diff-reviewed on 2026-07-07.
#
# Host deps (Ubuntu): gcc-aarch64-linux-gnu acpica-tools uuid-dev
#                     python3 build-essential git
set -euo pipefail

REPO=https://github.com/NumberOneGit/rpi5-uefi
# master @ 2026-04-27 — D0 + C1 support, EDK2 updated to current.
PIN=ad501cf3aeb7060b1ce0324b9d8972a4daf19b38

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/vendor/rpi5-uefi"

command -v aarch64-linux-gnu-gcc >/dev/null || {
    echo "missing cross compiler: sudo apt install gcc-aarch64-linux-gnu acpica-tools uuid-dev" >&2
    exit 1
}
command -v iasl >/dev/null || {
    echo "missing iasl: sudo apt install acpica-tools" >&2
    exit 1
}

if [ ! -d "$DIR/.git" ]; then
    git clone "$REPO" "$DIR"
fi
cd "$DIR"
git fetch --quiet origin
git checkout --quiet "$PIN"
git submodule update --init --jobs 4

echo "== building TF-A + EDK2 (RELEASE, rpi5) at $PIN"
./build.sh --model 5

echo
echo "== firmware payload:"
sha256sum RPI_EFI.fd config.txt
echo "RPI_EFI.fd ready in $DIR — now: cargo xtask pi-image [--model models/<file>.nrm]"
