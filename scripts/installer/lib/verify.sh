# shellcheck shell=bash disable=SC2034
# (SC2034: globals here are set for sibling modules sourced by install.sh)
# verify.sh — post-flash readback verification and the completion screen.
#
# Verification reads back exactly the image-sized region from the device
# (never the unused remainder) and compares its SHA-256 with the source
# image digest computed at build time. Full verification is the default;
# skipping requires an explicit, warned confirmation.

nr_verify_flow() {
    nr_section "VERIFYING BOOT MEDIA"
    nr_ask "Verify the written image by reading it back? [1] Yes, verify (default)  [2] Skip"
    if [[ "$REPLY" == "2" ]]; then
        nr_warn "Skipping verification means write errors would only show up as boot failures."
        nr_ask "Really skip? [1] Yes, skip  [2] No, verify (default)"
        if [[ "$REPLY" == "1" ]]; then
            NR_VERIFY_STATUS="SKIPPED (at your request)"
            nr_warn "Verification skipped."
            return 0
        fi
    fi

    nr_note "Comparing written NightRun image with the source image..."
    local log="$NR_LOG_DIR/verify.log"
    local got
    got="$(nr_readback_sha "$NR_SEL_PATH" "$NR_IMAGE_BYTES" "$log")"

    if [[ "$got" != "$NR_IMAGE_SHA" ]]; then
        nr_error "VERIFICATION FAILED — the written data does not match the image."
        nr_note  "This media may be unreliable (worn flash or a failing reader are"
        nr_note  "common causes). The drive is NOT known to be bootable."
        nr_note  "Try re-flashing; if it fails again, use a different card/drive."
        nr_note  "Log: $log"
        return 1
    fi
    NR_VERIFY_STATUS="Verified"
    nr_ok "Image verification passed."
    return 0
}

nr_completion_screen() {
    nr_section "NIGHTRUN IS READY"
    nr_note "Boot media created successfully."
    echo
    nr_kv "Target" "$NR_TARGET_LABEL"
    nr_kv "Device" "$NR_SEL_PATH"
    nr_kv "Model"  "$NR_MODEL_NAME"
    nr_kv "Status" "${NR_VERIFY_STATUS:-unknown}"
    echo
    if [[ "$NR_TARGET" == "rpi5" ]]; then
        nr_note "Next:"
        nr_note "  1. Safely remove the microSD card."
        nr_note "  2. Insert it into the Raspberry Pi 5."
        nr_note "  3. Connect an HDMI display (port next to USB-C) and a USB keyboard."
        nr_note "  4. Power on. First boot: QR screen -> Pi logo -> NightRun."
        nr_note "  Note: the Pi's EEPROM must be from 2025-06-09 or newer"
        nr_note "  (see docs/rpi5-uefi.md if the screen stays black)."
    else
        nr_note "Next:"
        nr_note "  1. Safely remove the USB drive."
        nr_note "  2. Insert it into a compatible x86_64 computer."
        nr_note "  3. Open the firmware boot menu (usually F12, F10 or Esc at power-on)."
        nr_note "  4. Select the NightRun USB drive (Secure Boot must be off)."
        nr_note "  5. Boot into NightRun."
    fi
    echo
    nr_item "E" "Eject / power off the media safely"
    nr_item "D" "Done"
    nr_ask "Choose:"
    if [[ "$REPLY" == [eE] ]]; then
        if command -v udisksctl >/dev/null; then
            nr_note "udisksctl power-off -b $NR_SEL_PATH"
            if udisksctl power-off -b "$NR_SEL_PATH" 2>/dev/null; then
                nr_ok "Media powered off — safe to remove."
            else
                nr_warn "Power-off not supported here; the media is already synced and safe to remove."
            fi
        else
            nr_note "udisksctl is not available; the media is synced and safe to remove."
        fi
    fi
    echo
    nr_wordmark
    echo
}

# Privileged read wrapper — a seam the test suite overrides to exercise
# the readback math against regular files without sudo or real devices.
nr_dd() { sudo dd "$@"; }

# SHA-256 of exactly `bytes` from the start of `dev`, streamed in 4 MiB
# slices plus a byte-exact tail (images need not be block-multiples).
# Progress goes to stderr so it never contaminates the hashed stream.
# Fail-safe by construction: any failed read yields a partial-stream
# digest, which can never equal the expected image digest.
nr_readback_sha() {
    local dev="$1" bytes="$2" log="$3"
    local block=$(( 4 * 1024 * 1024 ))
    local per_slice=32
    local total_blocks=$(( bytes / block ))
    local tail_bytes=$(( bytes % block ))
    {
        local read_blocks=0 count
        while (( read_blocks < total_blocks )); do
            count=$(( total_blocks - read_blocks ))
            (( count > per_slice )) && count=$per_slice
            nr_dd if="$dev" bs=4M skip="$read_blocks" count="$count" \
                 iflag=direct status=none 2>>"$log" || exit 1
            read_blocks=$(( read_blocks + count ))
            nr_progress $(( read_blocks * block )) "$bytes" "reading back" >&2
        done
        if (( tail_bytes > 0 )); then
            nr_dd if="$dev" bs=1 skip=$(( total_blocks * block )) \
                 count="$tail_bytes" status=none 2>>"$log" || exit 1
            nr_progress "$bytes" "$bytes" "reading back" >&2
        fi
    } | sha256sum | cut -d' ' -f1
}
