# shellcheck shell=bash disable=SC2034
# (SC2034: globals here are set for sibling modules sourced by install.sh)
# media.sh — removable-media detection with a conservative exclusion
# engine. The rules (docs/installer.md) in one line: only whole disks,
# only removable USB/SD transports, never anything backing the running
# system, never a guess.
#
# Testability: the two data sources are injectable. When NR_LSBLK_FIXTURE
# is set, device rows come from that file instead of lsblk; when
# NR_PROTECTED_FIXTURE is set, the protected-disk list comes from there
# instead of findmnt/swaps. Tests drive the whole engine on fixture data —
# no test ever touches a real disk.

# Candidate arrays (parallel, indexed):
#   NR_DEV_PATH[i] NR_DEV_SIZE[i] NR_DEV_TRAN[i] NR_DEV_MODEL[i]
#   NR_DEV_SERIAL[i] NR_DEV_MAJMIN[i] NR_DEV_MOUNTS[i]
# Selection outcome:
#   NR_SEL_PATH + NR_SEL_FP (identity fingerprint) + NR_SEL_* details

nr_lsblk_rows() {
    if [[ -n "${NR_LSBLK_FIXTURE:-}" ]]; then
        cat -- "$NR_LSBLK_FIXTURE"
    else
        lsblk -P -b -n -o PATH,TYPE,SIZE,RM,TRAN,MODEL,SERIAL,MAJ:MIN,MOUNTPOINTS 2>/dev/null
    fi
}

# Whole disks that back /, /boot, /home or active swap — never candidates.
nr_protected_disks() {
    if [[ -n "${NR_PROTECTED_FIXTURE:-}" ]]; then
        cat -- "$NR_PROTECTED_FIXTURE"
        return
    fi
    local src mp
    {
        for mp in / /boot /boot/efi /home; do
            src="$(findmnt -no SOURCE --target "$mp" 2>/dev/null)" || continue
            [[ "$src" == /dev/* ]] && lsblk -no PKNAME -- "$src" 2>/dev/null
            # A disk mounted directly (no partition) has no PKNAME.
            [[ "$src" == /dev/* ]] && lsblk -dno NAME -- "$src" 2>/dev/null
        done
        # Active swap devices.
        awk '$1 ~ /^\/dev\// {print $1}' /proc/swaps 2>/dev/null | while IFS= read -r src; do
            lsblk -no PKNAME -- "$src" 2>/dev/null
            lsblk -dno NAME -- "$src" 2>/dev/null
        done
    } | sed 's|^|/dev/|; s|^/dev//dev/|/dev/|' | sort -u
}

# Parse one lsblk -P row into the associative array `row`.
nr_parse_row() {
    local line="$1"
    row=()
    while [[ "$line" =~ ([A-Z:]+)=\"([^\"]*)\"[[:space:]]* ]]; do
        row["${BASH_REMATCH[1]}"]="${BASH_REMATCH[2]}"
        line="${line:${#BASH_REMATCH[0]}}"
    done
}

# Build the candidate list. Args: minimum size in bytes.
# Populates the NR_DEV_* arrays; returns 0 even when empty (UI decides).
nr_media_candidates() {
    local min_bytes="$1"
    NR_DEV_PATH=() NR_DEV_SIZE=() NR_DEV_TRAN=() NR_DEV_MODEL=()
    NR_DEV_SERIAL=() NR_DEV_MAJMIN=() NR_DEV_MOUNTS=()

    local protected
    protected="$(nr_protected_disks)"

    # First pass: collect every row for mount lookups per disk.
    local -a all_lines=()
    local line
    while IFS= read -r line; do
        [[ -n "$line" ]] && all_lines+=("$line")
    done < <(nr_lsblk_rows)

    local -A row
    for line in "${all_lines[@]}"; do
        nr_parse_row "$line"
        [[ "${row[TYPE]:-}" == "disk" ]] || continue
        local path="${row[PATH]:-}"
        [[ -n "$path" ]] || continue

        # Virtual / non-target device classes, by name.
        case "$path" in
            /dev/loop* | /dev/zram* | /dev/ram* | /dev/sr* | /dev/dm-* | /dev/md*) continue ;;
        esac
        # Protected system disks.
        if grep -qxF -- "$path" <<<"$protected"; then continue; fi
        # Transport: USB always considered; SD/MMC via the mmcblk name
        # (lsblk reports empty TRAN for mmc). Internal disks (sata/nvme/
        # ata/none) are unsafe by default — even if they claim RM=1,
        # which some card readers and enclosures falsely do or omit.
        local tran="${row[TRAN]:-}"
        if [[ "$tran" != "usb" ]]; then
            [[ "$path" == /dev/mmcblk* ]] || continue
            tran="sd/mmc"
        fi
        # Capacity.
        local size="${row[SIZE]:-0}"
        (( size >= min_bytes )) || continue

        # Mounted partitions of this disk (informational + unmount flow).
        local mounts="" l
        local -A prow
        for l in "${all_lines[@]}"; do
            nr_parse_row_into prow "$l"
            if [[ "${prow[TYPE]:-}" == "part" && "${prow[PATH]:-}" == "$path"* && -n "${prow[MOUNTPOINTS]:-}" ]]; then
                mounts+="${prow[PATH]}:${prow[MOUNTPOINTS]}"$'\n'
            fi
        done

        NR_DEV_PATH+=("$path")
        NR_DEV_SIZE+=("$size")
        NR_DEV_TRAN+=("$tran")
        NR_DEV_MODEL+=("${row[MODEL]:-unknown}")
        NR_DEV_SERIAL+=("${row[SERIAL]:-unknown}")
        NR_DEV_MAJMIN+=("${row[MAJ:MIN]:-}")
        NR_DEV_MOUNTS+=("$mounts")
    done
    return 0
}

# Helper: parse a row into a named associative array (nameref).
nr_parse_row_into() {
    local -n _dst="$1"
    local line="$2"
    _dst=()
    while [[ "$line" =~ ([A-Z:]+)=\"([^\"]*)\"[[:space:]]* ]]; do
        _dst["${BASH_REMATCH[1]}"]="${BASH_REMATCH[2]}"
        line="${line:${#BASH_REMATCH[0]}}"
    done
}

# Identity fingerprint for re-verification before destructive steps.
nr_fingerprint() {
    printf '%s|%s|%s|%s|%s' "$1" "$2" "$3" "$4" "$5" # path|size|model|serial|majmin
}

# Interactive selection. Uses NR_MIN_MEDIA_GB/IMAGE size from build.
nr_select_media() {
    local min_bytes="$NR_IMAGE_BYTES"
    while :; do
        nr_media_candidates "$min_bytes"
        if (( ${#NR_DEV_PATH[@]} == 0 )); then
            nr_section "NO SAFE REMOVABLE MEDIA DETECTED"
            nr_note "Only removable USB drives and SD cards at least"
            nr_note "$(nr_human_bytes "$min_bytes") large are eligible; system disks are never shown."
            nr_note "Insert a $NR_TARGET_MEDIA, then:"
            nr_item "R" "Rescan devices"
            nr_item "Q" "Quit"
            nr_ask "Choose:"
            case "$REPLY" in
                [rR]) continue ;;
                *) nr_note "Nothing was written."; exit "$EXIT_USER_ABORT" ;;
            esac
        fi

        nr_section "AVAILABLE BOOT MEDIA"
        local i
        for i in "${!NR_DEV_PATH[@]}"; do
            local mounted="not mounted"
            [[ -n "${NR_DEV_MOUNTS[$i]}" ]] && mounted="$(head -1 <<<"${NR_DEV_MOUNTS[$i]}" | cut -d: -f2-) (+)"
            nr_item "$(( i + 1 ))" "${NR_DEV_PATH[$i]}" ""
            nr_kv "Size"      "$(nr_human_bytes "${NR_DEV_SIZE[$i]}")"
            nr_kv "Transport" "${NR_DEV_TRAN[$i]}"
            nr_kv "Model"     "${NR_DEV_MODEL[$i]}"
            nr_kv "Serial"    "${NR_DEV_SERIAL[$i]}"
            nr_kv "Mounted"   "$mounted"
        done
        nr_item "R" "Rescan devices"
        nr_item "Q" "Quit"
        nr_ask "Select the device to flash:"
        case "$REPLY" in
            [rR]) continue ;;
            [qQ]) nr_note "Nothing was written."; exit "$EXIT_USER_ABORT" ;;
            *)
                if [[ "$REPLY" =~ ^[0-9]+$ ]] && (( REPLY >= 1 && REPLY <= ${#NR_DEV_PATH[@]} )); then
                    i=$(( REPLY - 1 ))
                    NR_SEL_PATH="${NR_DEV_PATH[$i]}"
                    NR_SEL_SIZE="${NR_DEV_SIZE[$i]}"
                    NR_SEL_TRAN="${NR_DEV_TRAN[$i]}"
                    NR_SEL_MODEL="${NR_DEV_MODEL[$i]}"
                    NR_SEL_SERIAL="${NR_DEV_SERIAL[$i]}"
                    NR_SEL_MAJMIN="${NR_DEV_MAJMIN[$i]}"
                    NR_SEL_FP="$(nr_fingerprint "$NR_SEL_PATH" "$NR_SEL_SIZE" "$NR_SEL_MODEL" "$NR_SEL_SERIAL" "$NR_SEL_MAJMIN")"
                    # Immediate re-scan: the device must still exist with
                    # the same identity (path swaps happen when drives are
                    # plugged/unplugged between scan and choice).
                    if ! nr_reverify_selection; then
                        nr_warn "The selected device changed or disappeared; the list was refreshed."
                        continue
                    fi
                    return 0
                fi
                nr_warn "Please pick a listed device number, R or Q."
                ;;
        esac
    done
}

# Re-enumerate and confirm the selected device still matches its recorded
# identity. Used at selection time and again immediately before writing.
nr_reverify_selection() {
    nr_media_candidates 0
    local i
    for i in "${!NR_DEV_PATH[@]}"; do
        if [[ "${NR_DEV_PATH[$i]}" == "$NR_SEL_PATH" ]]; then
            local fp
            fp="$(nr_fingerprint "${NR_DEV_PATH[$i]}" "${NR_DEV_SIZE[$i]}" "${NR_DEV_MODEL[$i]}" "${NR_DEV_SERIAL[$i]}" "${NR_DEV_MAJMIN[$i]}")"
            [[ "$fp" == "$NR_SEL_FP" ]] || return 1
            NR_SEL_MOUNTS="${NR_DEV_MOUNTS[$i]}"
            return 0
        fi
    done
    return 1
}
