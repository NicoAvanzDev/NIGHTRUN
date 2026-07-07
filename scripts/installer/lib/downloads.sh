# shellcheck shell=bash disable=SC2034
# (SC2034: globals here are set for sibling modules sourced by install.sh)
# downloads.sh — verified model acquisition from Hugging Face.
#
# Every download is pinned: the URL uses the manifest's revision (a repo
# commit, not a branch), lands in a temp file, must match the expected
# size and SHA-256, must carry GGUF magic, and only then is atomically
# renamed into the cache (models/). Secrets policy: a token is taken from
# HF_TOKEN or prompted with echo off, kept in memory only, and never
# written to logs or disk.

# Outcome global: NR_GGUF_PATH

nr_acquire_model() {
    local mid="$NR_MODEL_ID"
    local cache="$NR_ROOT/models/${NR_MF[$mid.file]}"

    while :; do
        nr_section "MODEL SOURCE"
        nr_item "1" "Download the verified GGUF automatically" \
                "$(nr_human_bytes "${NR_MF[$mid.size_bytes]}") from ${NR_MF[$mid.repo]} (pinned + checksummed)"
        nr_item "2" "Use an existing GGUF file from this computer"
        nr_item "B" "Back"
        nr_ask "Choose a source:"
        case "$REPLY" in
            1) nr_download_model "$mid" "$cache" && { NR_GGUF_PATH="$cache"; return 0; } ;;
            2) nr_pick_local_gguf && return 0 ;;
            [bB]) return 1 ;;
            *) nr_warn "Please choose 1, 2 or B." ;;
        esac
    done
}

nr_download_model() {
    local mid="$1" cache="$2"
    local url="https://huggingface.co/${NR_MF[$mid.repo]}/resolve/${NR_MF[$mid.revision]}/${NR_MF[$mid.file]}"
    local want_sha="${NR_MF[$mid.sha256]}"
    local want_size="${NR_MF[$mid.size_bytes]}"

    # Cached and already verified once? Re-verify cheaply by size, fully by
    # hash, and reuse.
    if [[ -f "$cache" ]]; then
        local have_size
        have_size="$(stat -c %s -- "$cache")"
        if [[ "$have_size" == "$want_size" ]]; then
            nr_note "Found cached download; verifying checksum..."
            if nr_sha256_check "$cache" "$want_sha"; then
                nr_ok "Cache hit: $cache (checksum verified)"
                nr_inspect_gguf "$cache" "DOWNLOADED MODEL CHECK" || return 1
                return 0
            fi
            nr_warn "Cached file failed its checksum; it will be re-downloaded."
        fi
    fi

    nr_section "DOWNLOAD"
    nr_kv "Provider"  "Hugging Face"
    nr_kv "Repository" "${NR_MF[$mid.repo]}"
    nr_kv "Artifact"  "${NR_MF[$mid.file]} (${NR_MF[$mid.quant]})"
    nr_kv "Revision"  "${NR_MF[$mid.revision]:0:12} (pinned)"
    nr_kv "Size"      "$(nr_human_bytes "$want_size")"
    nr_kv "Cache"     "$cache"
    nr_kv "License"   "${NR_MF[$mid.license]}"

    # Network check happens only now — the first moment it is needed.
    if ! nr_probe_url "$url"; then
        return 1
    fi

    # Resumable download into a stable .part path (survives re-runs).
    local part="${cache}.part"
    mkdir -p -- "$(dirname -- "$cache")"
    echo
    if ! nr_fetch "$url" "$part" "$want_size"; then
        nr_error "Download failed. The partial file is kept for resuming:"
        nr_note  "$part"
        nr_note  "Re-run the installer to resume, or fetch manually per README.md."
        return 1
    fi

    local got_size
    got_size="$(stat -c %s -- "$part")"
    if [[ "$got_size" != "$want_size" ]]; then
        nr_error "Size mismatch: expected $want_size bytes, got $got_size."
        rm -f -- "$part"
        return 1
    fi
    nr_note "Verifying SHA-256 (this reads the whole file)..."
    if ! nr_sha256_check "$part" "$want_sha"; then
        nr_error "Checksum mismatch — the download does not match the pinned artifact."
        nr_note  "The corrupt file was removed; re-run to try again."
        rm -f -- "$part"
        return 1
    fi
    if ! nr_check_gguf_magic "$part"; then
        nr_error "Downloaded file is not a GGUF (magic bytes mismatch)."
        rm -f -- "$part"
        return 1
    fi
    mv -f -- "$part" "$cache"
    nr_ok "Downloaded and verified: $(basename "$cache")"

    nr_inspect_gguf "$cache" "DOWNLOADED MODEL CHECK" || return 1
    return 0
}

nr_sha256_check() {
    local file="$1" want="$2" got
    got="$(sha256sum -- "$file" | cut -d' ' -f1)"
    [[ "$got" == "$want" ]]
}

# HEAD-probe the pinned URL; detects offline hosts and gated repos, and
# handles the optional token without ever echoing it.
nr_probe_url() {
    # Never trace token handling (debug mode writes xtrace to a log).
    { [[ -n "${NIGHTRUN_INSTALL_DEBUG:-}" ]] && set +x; } 2>/dev/null
    local url="$1" code
    code="$(curl -sIL -o /dev/null -w '%{http_code}' --max-time 20 \
                 ${NR_HF_TOKEN:+-H "Authorization: Bearer $NR_HF_TOKEN"} "$url")" || code=000
    case "$code" in
        200) return 0 ;;
        000)
            nr_error "Cannot reach huggingface.co — check your network connection."
            return 1
            ;;
        401 | 403)
            nr_warn "This repository requires Hugging Face authorization (HTTP $code)."
            nr_note "If the model is gated, accept its terms on the official model page"
            nr_note "first: https://huggingface.co/${NR_MF[$NR_MODEL_ID.repo]}"
            if [[ -n "${HF_TOKEN:-}" && -z "${NR_HF_TOKEN:-}" ]]; then
                nr_note "Trying the token from the HF_TOKEN environment variable..."
                NR_HF_TOKEN="$HF_TOKEN"
                nr_probe_url "$url"
                return $?
            fi
            if [[ -z "${NR_HF_TOKEN:-}" ]]; then
                printf '%sHugging Face token (input hidden, never stored): %s' "$C_ORG" "$C_OFF"
                IFS= read -rs NR_HF_TOKEN
                echo
                if [[ -n "$NR_HF_TOKEN" ]]; then
                    nr_probe_url "$url"
                    return $?
                fi
            fi
            nr_error "Not authorized for this artifact."
            return 1
            ;;
        404)
            nr_error "Artifact not found at the pinned revision (HTTP 404)."
            nr_note  "The manifest pin may be stale; see docs/installer.md."
            return 1
            ;;
        *)
            nr_error "Unexpected response from Hugging Face (HTTP $code)."
            return 1
            ;;
    esac
}

# Fetch with resume + progress. Never prints the token (curl config file
# via stdin keeps it off the process list too).
nr_fetch() {
    # Never trace token handling (debug mode writes xtrace to a log).
    { [[ -n "${NIGHTRUN_INSTALL_DEBUG:-}" ]] && set +x; } 2>/dev/null
    local url="$1" out="$2" total="$3"
    local args=(-L -C - -o "$out" --retry 3 --fail --silent --show-error)
    if [[ -n "${NR_HF_TOKEN:-}" ]]; then
        # --config from stdin: the header never appears in `ps` or logs.
        curl "${args[@]}" --config <(printf 'header = "Authorization: Bearer %s"\n' "$NR_HF_TOKEN") "$url" &
    else
        curl "${args[@]}" "$url" &
    fi
    local pid=$!
    # Progress from the growing file (curl's own meter is suppressed so the
    # UI stays consistent).
    while kill -0 "$pid" 2>/dev/null; do
        local sz=0
        [[ -f "$out" ]] && sz="$(stat -c %s -- "$out" 2>/dev/null || echo 0)"
        nr_progress "$sz" "$total" "downloading $(basename "$out" .part)"
        sleep 1
    done
    wait "$pid"
    local rc=$?
    (( rc == 0 )) && nr_progress "$total" "$total" "downloaded"
    return "$rc"
}
