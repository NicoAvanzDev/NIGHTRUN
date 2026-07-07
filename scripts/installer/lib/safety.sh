# shellcheck shell=bash disable=SC2034
# (SC2034: globals here are set for sibling modules sourced by install.sh)
# safety.sh — shared guards for the NightRun installer.
#
# Policy embodied here (see docs/installer.md):
#  - no eval, ever; commands are built as arrays
#  - every user-supplied path is canonicalized before use
#  - temporary state lives in one mktemp dir, removed by trap
#  - distinct exit codes so wrappers can tell failure classes apart
#
# Error-handling strategy: `set -u -o pipefail` plus explicit checks at
# every step that matters (deliberately NOT `set -e`, whose implicit
# behavior around conditionals is exactly the brittleness the spec warns
# about). Functions return nonzero and callers decide.

set -u -o pipefail

# Exit codes
readonly EXIT_OK=0
readonly EXIT_PREFLIGHT=2
readonly EXIT_USER_ABORT=3
readonly EXIT_BUILD=4
readonly EXIT_MEDIA=5
readonly EXIT_FLASH=6
readonly EXIT_VERIFY=7

# One temp dir for the whole run; cleaned on any exit.
NR_TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/nightrun-install.XXXXXX")" || {
    echo "fatal: cannot create temporary directory" >&2
    exit "$EXIT_PREFLIGHT"
}

nr_cleanup() {
    local rc=$?
    # Restore terminal state (colors off, cursor visible) before anything.
    if [[ -t 1 ]]; then printf '\033[0m\033[?25h'; fi
    rm -rf -- "$NR_TMPDIR"
    return "$rc"
}
trap nr_cleanup EXIT

# Ctrl+C: honest message, no success claims. flash.sh upgrades this trap
# while a write is in progress.
nr_sigint() {
    if [[ -t 1 ]]; then printf '\033[0m\033[?25h'; fi
    echo
    echo "Interrupted. Nothing has been claimed as complete."
    exit "$EXIT_USER_ABORT"
}
trap nr_sigint INT

# Canonicalize a user path: ~ expansion (only a leading ~/), then realpath.
# Prints the canonical path; returns 1 if it cannot be resolved.
nr_canon_path() {
    local p="$1"
    # shellcheck disable=SC2088  # literal tilde matching is the point here
    case "$p" in
        "~") p="$HOME" ;;
        "~/"*) p="$HOME/${p#\~/}" ;;
    esac
    realpath -e -- "$p" 2>/dev/null
}

# Human-readable bytes (GiB with one decimal, MiB below 1 GiB).
nr_human_bytes() {
    local b="$1"
    if (( b >= 1073741824 )); then
        printf '%d.%d GB' "$(( b / 1073741824 ))" "$(( b % 1073741824 * 10 / 1073741824 ))"
    else
        printf '%d MB' "$(( b / 1048576 ))"
    fi
}

# Free bytes on the filesystem holding $1.
nr_free_bytes() {
    df --output=avail -B1 -- "$1" 2>/dev/null | tail -1 | tr -d ' '
}

# The exact typed destructive confirmation. Accepts ONLY the literal
# string `FLASH <device>` for the CURRENT device path: no case folding,
# no partial match, no partition paths, no blank input.
nr_confirm_exact_flash() {
    local device="$1" typed="$2"
    [[ "$typed" == "FLASH $device" ]]
}

# True when $1 looks like a whole-disk block device path we could ever
# consider (sanity gate before any deeper checks).
nr_is_disk_path() {
    local p="$1"
    [[ "$p" == /dev/* && -b "$p" ]] || return 1
    # Reject obvious partition names: trailing digit after a letter-name
    # (sdb1) or pN suffix (nvme0n1p2, mmcblk0p1).
    [[ "$p" =~ ^/dev/(sd[a-z]+|vd[a-z]+)$ ]] && return 0
    [[ "$p" =~ ^/dev/(nvme[0-9]+n[0-9]+|mmcblk[0-9]+)$ ]] && return 0
    return 1
}
