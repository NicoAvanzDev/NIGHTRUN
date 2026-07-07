# shellcheck shell=bash disable=SC2034
# (SC2034: globals here are set for sibling modules sourced by install.sh)
# targets.sh — the first interactive decision: which hardware NightRun
# will boot on. The two targets are genuinely different images and
# packaging paths (GPT/ESP x86 image vs MBR SD image with Pi firmware);
# everything downstream (model filtering, builder, media hints, boot
# instructions) keys off NR_TARGET.

# Outcome globals:
#   NR_TARGET          x86_64 | rpi5
#   NR_TARGET_LABEL    human name
#   NR_TARGET_MEDIA    "USB drive" | "microSD card"

nr_select_target() {
    while :; do
        nr_section "SELECT NIGHTRUN TARGET"
        nr_item "1" "x86_64 UEFI" "Boots from a USB drive on compatible UEFI PCs and laptops."
        nr_item "2" "Raspberry Pi 5" "Boots from a microSD card on Raspberry Pi 5 (D0 and C1 boards)."
        nr_item "Q" "Quit"
        nr_ask "Choose a target:"
        case "$REPLY" in
            1)
                NR_TARGET="x86_64"
                NR_TARGET_LABEL="x86_64 UEFI"
                NR_TARGET_MEDIA="USB drive"
                break
                ;;
            2)
                NR_TARGET="rpi5"
                NR_TARGET_LABEL="Raspberry Pi 5"
                NR_TARGET_MEDIA="microSD card"
                break
                ;;
            [qQ])
                nr_note "Nothing was changed."
                exit "$EXIT_USER_ABORT"
                ;;
            *) nr_warn "Please choose 1, 2 or Q." ;;
        esac
    done

    nr_section "TARGET SELECTED"
    nr_kv "Platform"   "$NR_TARGET_LABEL"
    nr_kv "Boot media" "$NR_TARGET_MEDIA"
    nr_kv "Image type" "NightRun $NR_TARGET_LABEL boot image"

    # Pi extra: the SD image embeds UEFI firmware built from pinned source.
    if [[ "$NR_TARGET" == "rpi5" ]]; then
        if [[ ! -f "$NR_ROOT/vendor/rpi5-uefi/RPI_EFI.fd" ]]; then
            echo
            nr_warn "The Raspberry Pi 5 image needs UEFI firmware built from source"
            nr_warn "(vendor/rpi5-uefi/RPI_EFI.fd is not present yet)."
            nr_note "scripts/build-rpi5-firmware.sh builds it from a pinned, reviewed"
            nr_note "commit; it needs: gcc-aarch64-linux-gnu acpica-tools uuid-dev."
            nr_ask "Build the Pi firmware now? [1] Yes  [2] No, quit (default)"
            if [[ "$REPLY" == "1" ]]; then
                if ! "$NR_ROOT/scripts/build-rpi5-firmware.sh"; then
                    nr_error "Firmware build failed — see its output above."
                    exit "$EXIT_BUILD"
                fi
            else
                nr_note "See docs/rpi5-uefi.md, then re-run ./install.sh."
                exit "$EXIT_USER_ABORT"
            fi
        fi
        if ! rustup target list --installed 2>/dev/null | grep -q aarch64-unknown-uefi; then
            nr_note "Adding the Rust aarch64-unknown-uefi target (rustup)..."
            rustup target add aarch64-unknown-uefi || {
                nr_error "rustup target add aarch64-unknown-uefi failed."
                exit "$EXIT_PREFLIGHT"
            }
        fi
    fi
}
