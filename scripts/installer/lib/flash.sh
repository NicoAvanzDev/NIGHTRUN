# shellcheck shell=bash disable=SC2034
# (SC2034: globals here are set for sibling modules sourced by install.sh)
# flash.sh — unmount consent, the two-stage destructive confirmation, full
# revalidation, and the write itself.
#
# The write happens in 128 MiB slices of `dd bs=4M oflag=direct` with
# conv=fsync on the final slice, giving a dependency-free progress bar and
# bounded buffering. The exact command template and the exact device are
# always displayed. Ctrl+C during the write leaves an honest "incomplete
# media" message — never a success claim.

nr_flash_flow() {
    nr_unmount_selected || return 1
    nr_final_confirmations || return 1
    nr_revalidate_before_write || return 1
    nr_write_image || return 1
    return 0
}

nr_unmount_selected() {
    nr_reverify_selection || {
        nr_error "The selected device changed or disappeared; aborting."
        return 1
    }
    [[ -z "${NR_SEL_MOUNTS:-}" ]] && return 0

    nr_section "MOUNTED PARTITIONS ON $NR_SEL_PATH"
    local entry part mp
    while IFS= read -r entry; do
        [[ -n "$entry" ]] || continue
        part="${entry%%:*}"
        mp="${entry#*:}"
        nr_kv "$part" "$mp"
    done <<<"$NR_SEL_MOUNTS"
    nr_note "They must be unmounted before flashing (writes to a mounted"
    nr_note "device corrupt both the write and the mounted filesystem)."
    nr_ask "Unmount all partitions on $NR_SEL_PATH now? [1] Yes, unmount safely  [2] No, cancel (default)"
    [[ "$REPLY" == "1" ]] || { nr_note "Cancelled. Nothing was written."; return 1; }

    while IFS= read -r entry; do
        [[ -n "$entry" ]] || continue
        part="${entry%%:*}"
        if command -v udisksctl >/dev/null; then
            nr_note "udisksctl unmount -b $part"
            udisksctl unmount -b "$part" >/dev/null 2>&1 && continue
        fi
        nr_note "sudo umount -- $part"
        sudo umount -- "$part" || {
            nr_error "Could not unmount $part — flashing is blocked by this partition."
            nr_note  "Close applications using it (file managers, terminals cd'd into it) and retry."
            return 1
        }
    done <<<"$NR_SEL_MOUNTS"

    # Recheck: state must actually be clean now.
    nr_reverify_selection || { nr_error "Device state changed during unmount; aborting."; return 1; }
    if [[ -n "${NR_SEL_MOUNTS:-}" ]]; then
        nr_error "Some partitions are still mounted:"
        printf '%s\n' "$NR_SEL_MOUNTS" | sed 's/^/    /'
        return 1
    fi
    nr_ok "All partitions unmounted."
    return 0
}

nr_final_confirmations() {
    nr_section "FINAL FLASH SUMMARY"
    nr_kv "NightRun target" "$NR_TARGET_LABEL"
    nr_kv "Image"           "$(basename "$NR_IMAGE_PATH")"
    nr_kv "Image size"      "$(nr_human_bytes "$NR_IMAGE_BYTES")"
    echo
    nr_kv "Selected drive"  "$NR_SEL_PATH"
    nr_kv "Capacity"        "$(nr_human_bytes "$NR_SEL_SIZE")"
    nr_kv "Transport"       "$NR_SEL_TRAN"
    nr_kv "Model"           "$NR_SEL_MODEL"
    nr_kv "Serial"          "$NR_SEL_SERIAL"
    echo
    printf '%s\n' "${C_MAG}${C_BOLD}  WARNING: this permanently erases every partition and all data${C_OFF}"
    printf '%s\n' "${C_MAG}${C_BOLD}  on ${NR_SEL_PATH}. This cannot be undone.${C_OFF}"

    nr_ask "Do you want to permanently erase this exact device? [1] Continue  [2] Cancel (default)"
    [[ "$REPLY" == "1" ]] || { nr_note "Cancelled. Nothing was written."; return 1; }

    echo
    nr_note "Type exactly:  FLASH $NR_SEL_PATH"
    nr_note "to permanently erase this drive and write NightRun."
    printf '%s> %s' "$C_ORG" "$C_OFF"
    IFS= read -r REPLY
    if ! nr_confirm_exact_flash "$NR_SEL_PATH" "$REPLY"; then
        nr_error "Confirmation did not match exactly. Nothing was written."
        return 1
    fi
    return 0
}

nr_revalidate_before_write() {
    nr_note "Revalidating before writing..."
    [[ -b "$NR_SEL_PATH" ]] || { nr_error "$NR_SEL_PATH no longer exists."; return 1; }
    nr_is_disk_path "$NR_SEL_PATH" || { nr_error "$NR_SEL_PATH is not a whole-disk device."; return 1; }
    nr_reverify_selection || { nr_error "Device identity changed since selection."; return 1; }
    [[ -z "${NR_SEL_MOUNTS:-}" ]] || { nr_error "Device has mounted partitions again."; return 1; }
    if grep -qxF -- "$NR_SEL_PATH" <<<"$(nr_protected_disks)"; then
        nr_error "$NR_SEL_PATH backs the running system — refusing."
        return 1
    fi
    (( NR_SEL_SIZE >= NR_IMAGE_BYTES )) || { nr_error "Device is smaller than the image."; return 1; }
    # The image must be exactly what we hashed at build time.
    local sha
    sha="$(sha256sum -- "$NR_IMAGE_PATH" | cut -d' ' -f1)"
    [[ "$sha" == "$NR_IMAGE_SHA" ]] || { nr_error "Image changed on disk since it was built."; return 1; }
    nr_ok "All checks passed."
    return 0
}

nr_write_image() {
    nr_section "FLASHING NIGHTRUN"
    nr_note "Writing $(nr_human_bytes "$NR_IMAGE_BYTES") to $NR_SEL_PATH"
    nr_note "Command per 128 MiB slice:"
    nr_note "  sudo dd if=$NR_IMAGE_PATH of=$NR_SEL_PATH bs=4M skip=N seek=N count=32 conv=notrunc oflag=direct"

    if (( ! ${NR_SUDO_READY:-0} )); then
        nr_note "sudo needs your password for the write:"
        sudo -v || { nr_error "sudo authorization failed."; return 1; }
    fi

    local log="$NR_LOG_DIR/flash.log"
    mkdir -p -- "$NR_LOG_DIR"

    # Ctrl+C during the write must tell the truth about media state.
    trap 'nr_flash_interrupted' INT

    local block=$(( 4 * 1024 * 1024 ))
    local per_slice=32                          # 4M x 32 = 128 MiB
    local total_blocks=$(( (NR_IMAGE_BYTES + block - 1) / block ))
    local written=0 rc=0
    while (( written < total_blocks )); do
        local count=$(( total_blocks - written ))
        (( count > per_slice )) && count=$per_slice
        local conv="notrunc"
        (( written + count >= total_blocks )) && conv="notrunc,fsync"
        # shellcheck disable=SC2024  # the log is meant to be user-owned
        if ! sudo dd if="$NR_IMAGE_PATH" of="$NR_SEL_PATH" bs=4M \
                  skip="$written" seek="$written" count="$count" \
                  conv="$conv" oflag=direct status=none >>"$log" 2>&1; then
            rc=1
            break
        fi
        written=$(( written + count ))
        nr_progress $(( written * block )) "$NR_IMAGE_BYTES" "writing to $NR_SEL_PATH"
    done
    trap nr_sigint INT

    if (( rc != 0 )); then
        nr_error "Write failed at block $written. The media is INCOMPLETE and not bootable."
        nr_note  "Log: $log"
        return 1
    fi

    nr_note "Flushing device buffers (sync)..."
    sync
    sudo partprobe -- "$NR_SEL_PATH" 2>>"$log" || true
    sleep 1
    nr_ok "Write complete."
    return 0
}

nr_flash_interrupted() {
    trap nr_sigint INT
    if [[ -t 1 ]]; then printf '\033[0m\033[?25h'; fi
    echo
    nr_error "Interrupted during the write. $NR_SEL_PATH is INCOMPLETE and must not be trusted to boot."
    nr_note  "Re-run the installer to flash it again from the start."
    nr_note  "Log: $NR_LOG_DIR/flash.log"
    exit "$EXIT_FLASH"
}
