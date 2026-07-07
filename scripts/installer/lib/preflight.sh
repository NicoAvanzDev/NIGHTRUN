# shellcheck shell=bash disable=SC2034
# (SC2034: globals here are set for sibling modules sourced by install.sh)
# preflight.sh — host checks before any workflow is shown.
#
# Philosophy: fail early with plain language; never install anything
# silently. When a dependency is missing we explain what and why, show the
# exact package-manager command, and only run it after explicit consent —
# with sudo scoped to that single command.

# Core tools the flow cannot run without (flash-time tools included so the
# user doesn't discover a gap after a 20-minute build).
NR_CORE_TOOLS=(lsblk findmnt mountpoint umount sync dd sha256sum stat realpath df)

nr_preflight() {
    nr_section "PREFLIGHT"

    if [[ "$(uname -s)" != "Linux" ]]; then
        nr_error "NightRun's installer supports Linux hosts only (found: $(uname -s))."
        nr_note  "Image building depends on Linux block-device and mount tooling;"
        nr_note  "on other systems, build images manually per README.md."
        exit "$EXIT_PREFLIGHT"
    fi

    if (( BASH_VERSINFO[0] < 4 )); then
        nr_error "Bash 4 or newer is required (found ${BASH_VERSION})."
        exit "$EXIT_PREFLIGHT"
    fi

    local missing=()
    local t
    for t in "${NR_CORE_TOOLS[@]}"; do
        command -v "$t" >/dev/null || missing+=("$t")
    done
    # A downloader: either curl or wget satisfies it.
    if ! command -v curl >/dev/null && ! command -v wget >/dev/null; then
        missing+=("curl")
    fi
    if (( ${#missing[@]} > 0 )); then
        nr_offer_install "${missing[@]}" || exit "$EXIT_PREFLIGHT"
    fi

    # NightRun build tooling. cargo drives both nrconvert and xtask.
    if ! command -v cargo >/dev/null; then
        nr_error "cargo (Rust) is missing — it builds NightRun and runs its tools."
        nr_note  "Install via https://rustup.rs then re-run. (No package-manager"
        nr_note  "offer here: rustup is the supported Rust installation path.)"
        exit "$EXIT_PREFLIGHT"
    fi
    if [[ ! -f "$NR_ROOT/Cargo.toml" || ! -d "$NR_ROOT/tools/nrconvert" ]]; then
        nr_error "This doesn't look like a NightRun checkout: $NR_ROOT"
        exit "$EXIT_PREFLIGHT"
    fi

    # Disk space: worst case = GGUF download + .nrm + image (~3x largest
    # model ≈ 8 GB). Warn under 10 GB, hard-stop under 3 GB.
    local free
    free="$(nr_free_bytes "$NR_ROOT")"
    if [[ -n "$free" ]]; then
        if (( free < 3 * 1024 * 1024 * 1024 )); then
            nr_error "Only $(nr_human_bytes "$free") free in $NR_ROOT — at least 3 GB is needed."
            exit "$EXIT_PREFLIGHT"
        elif (( free < 10 * 1024 * 1024 * 1024 )); then
            nr_warn "Only $(nr_human_bytes "$free") free — enough for small models; large downloads may not fit."
        fi
    fi

    # sudo will be needed at flash time (and possibly for umount). Probe
    # without prompting; just record whether a prompt will appear later.
    if command -v sudo >/dev/null; then
        if sudo -n true 2>/dev/null; then
            NR_SUDO_READY=1
        else
            NR_SUDO_READY=0
            nr_note "sudo will ask for your password at the flashing stage."
        fi
    else
        nr_error "sudo is not available — writing to a block device requires it."
        exit "$EXIT_PREFLIGHT"
    fi

    nr_ok "Linux host, Bash ${BASH_VERSION%%(*}, required tools present"
    nr_ok "NightRun checkout at $NR_ROOT"

    if [[ ! -f "$NR_MANIFEST" ]]; then
        nr_error "Model manifest missing: $NR_MANIFEST"
        exit "$EXIT_PREFLIGHT"
    fi
}

# Explain missing packages and offer to install them. Args: tool names.
nr_offer_install() {
    local tools=("$@")
    nr_error "Missing required tools: ${tools[*]}"
    nr_note  "They provide device listing, mount inspection, safe writing and checksums."

    local pm="" cmd=()
    if command -v apt-get >/dev/null; then
        pm="apt"; cmd=(sudo apt-get install -y "${tools[@]}")
    elif command -v dnf >/dev/null; then
        pm="dnf"; cmd=(sudo dnf install -y "${tools[@]}")
    elif command -v pacman >/dev/null; then
        pm="pacman"; cmd=(sudo pacman -S --noconfirm "${tools[@]}")
    fi

    if [[ -z "$pm" ]]; then
        nr_note "No supported package manager detected; install these manually and re-run."
        return 1
    fi

    echo
    nr_note "The exact command that would run:"
    printf '    %s\n' "${cmd[*]}"
    nr_ask "Install now via $pm? [1] Yes  [2] No (default)"
    if [[ "$REPLY" == "1" ]]; then
        "${cmd[@]}" || { nr_error "Package installation failed."; return 1; }
        local t
        for t in "${tools[@]}"; do
            command -v "$t" >/dev/null || { nr_error "$t is still missing after install."; return 1; }
        done
        nr_ok "Dependencies installed."
        return 0
    fi
    nr_note "Declined. Install manually (${cmd[*]#sudo }) and re-run ./install.sh."
    return 1
}
