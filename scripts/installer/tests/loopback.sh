#!/usr/bin/env bash
# Loopback integration test for the installer's flash + verify machinery.
# Requires sudo (losetup). Writes ONLY to a loop device backed by a temp
# file — never to real media. Run manually:
#
#   sudo scripts/installer/tests/loopback.sh
#
# Exercises: chunked dd write path, sync/settle, readback verification
# (both the matching and the deliberately-corrupted case).

set -u -o pipefail

if [[ $EUID -ne 0 ]]; then
    echo "This integration test needs root for losetup: sudo $0" >&2
    exit 2
fi

TESTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# repo root kept for future use
# shellcheck disable=SC2034
NR_ROOT="$(cd -- "$TESTS_DIR/../../.." && pwd)"

WORK="$(mktemp -d /tmp/nightrun-loopback.XXXXXX)"
LOOP=""
cleanup() {
    [[ -n "$LOOP" ]] && losetup -d "$LOOP" 2>/dev/null
    rm -rf -- "$WORK"
}
trap cleanup EXIT

echo "== creating a 64 MiB image and a 128 MiB backing device"
dd if=/dev/urandom of="$WORK/image.img" bs=1M count=64 status=none
dd if=/dev/zero of="$WORK/media.img" bs=1M count=128 status=none
LOOP="$(losetup --find --show -- "$WORK/media.img")"
echo "   loop device: $LOOP"

IMAGE_BYTES=$(( 64 * 1024 * 1024 ))
IMAGE_SHA="$(sha256sum -- "$WORK/image.img" | cut -d' ' -f1)"

echo "== chunked write (the installer's exact slice pattern)"
block=$(( 4 * 1024 * 1024 ))
per_slice=8
total_blocks=$(( IMAGE_BYTES / block ))
written=0
while (( written < total_blocks )); do
    count=$(( total_blocks - written ))
    (( count > per_slice )) && count=$per_slice
    conv="notrunc"
    (( written + count >= total_blocks )) && conv="notrunc,fsync"
    dd if="$WORK/image.img" of="$LOOP" bs=4M skip="$written" seek="$written" \
       count="$count" conv="$conv" oflag=direct status=none
    written=$(( written + count ))
done
sync

echo "== readback verification (must match)"
got="$(dd if="$LOOP" bs=4M count="$total_blocks" iflag=direct status=none | sha256sum | cut -d' ' -f1)"
if [[ "$got" != "$IMAGE_SHA" ]]; then
    echo "FAIL: clean write did not verify" >&2
    exit 1
fi
echo "   ok: digests match"

echo "== corruption detection (must NOT match)"
printf 'CORRUPT' | dd of="$LOOP" bs=1 seek=$(( 32 * 1024 * 1024 )) conv=notrunc status=none
sync
got="$(dd if="$LOOP" bs=4M count="$total_blocks" iflag=direct status=none | sha256sum | cut -d' ' -f1)"
if [[ "$got" == "$IMAGE_SHA" ]]; then
    echo "FAIL: corruption was not detected" >&2
    exit 1
fi
echo "   ok: corruption detected"

echo
echo "loopback integration: ALL PASSED"
