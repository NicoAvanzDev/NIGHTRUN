# Raspberry Pi 5 UEFI firmware: selection, pins, and known limitations

NightRun on Pi 5 runs as a standard AArch64 UEFI application on top of a
TF-A + EDK2 firmware port. This file records which firmware we use, why,
how it is built, and what is known not to work. **Read this before
touching the firmware setup; update it whenever a pin changes.**

## Firmware selection (researched 2026-07-07)

| option | status | verdict |
|---|---|---|
| [worproject/rpi5-uefi](https://github.com/worproject/rpi5-uefi) | **archived Feb 2025**, final release v0.3 (Mar 2024) | origin of the port; supports only early **C1**-stepping boards |
| [NumberOneGit/rpi5-uefi](https://github.com/NumberOneGit/rpi5-uefi) | community fork, active (HEAD 2026-04) | **selected** — adds D0 boards (all 2 GB/16 GB + late-2024+ units), keeps a `C1` branch, tracks current EDK2 |
| edk2-rk3588 (RK3588 boards) | actively maintained | recommended by worproject's maintainers, but it's different hardware — out of scope |

There is no institutionally maintained Pi 5 UEFI. The user explicitly
accepted this risk with mitigations: **build from source at a pinned,
diff-reviewed commit; never use binary releases; pin firmware + EEPROM as
a matched pair.**

## Pinned source

- Repo: `https://github.com/NumberOneGit/rpi5-uefi`
- Commit: `ad501cf3aeb7060b1ce0324b9d8972a4daf19b38` (master, 2026-04-27)
- Submodules (pinned by that commit):
  - `arm-trusted-firmware` @ `000fe221b859` (ARM-software upstream, rpi5 branch)
  - `edk2` @ `15590903fe01` (fork of tianocore/edk2)
  - `edk2-platforms` @ `4e426104a1f6` (fork of tianocore/edk2-platforms)
  - `edk2-non-osi` @ `07fe302e6eaf`

### Diff review (fork vs archived upstream, done 2026-07-07)

Scope: top repo + `edk2-platforms` (where all Pi platform code lives),
fork pin vs worproject pin `8e1779b5`.

- Top repo: DTB region moved `0x1F0000` → `0x3E0000` (build.sh +
  config.txt, consistent pair), submodule pointer bumps, README. Nothing
  else.
- `edk2-platforms`: ~890 insertions / ~1400 deletions across 39 files —
  the D0 pinctrl remap (`Bcm2712Pinctrl.h` rework + `Bcm2712GpioLib`),
  RpiFirmwareDxe mailbox modernization, FdtDxe/boot-manager/SMBIOS/MMC
  board-support changes, and mechanical `__FUNCTION__` → `__func__`
  renames (EDK2 API modernization). The network driver (BcmGenetDxe)
  delta is *only* those renames. **No suspicious code found.**

Safety posture: the firmware lives as files on the SD card; the Pi's
EEPROM bootloader (official Raspberry Pi firmware) loads it fresh each
boot and is never modified — a bad build cannot brick the board. The
firmware never executes on the development machine (QEMU testing uses
Ubuntu's AAVMF).

## Building

```sh
sudo apt install gcc-aarch64-linux-gnu acpica-tools uuid-dev
scripts/build-rpi5-firmware.sh      # clone @ pin, build, print SHA-256s
cargo xtask pi-image                # MBR SD image: firmware + BOOTAA64.EFI + model
sudo dd if=nightrun-pi5.img of=/dev/sdX bs=4M status=progress oflag=direct
```

Record the printed `RPI_EFI.fd` SHA-256 here after each rebuild:

- 2026-07-07, commit `ad501cf3`, gcc-aarch64-linux-gnu (Ubuntu noble):
  - `RPI_EFI.fd` = `8136a19d07c67c4b0804eaffd7e11f4c0de90a15aec7ef11d0073d57cf9a94c6`
  - `config.txt` = repo-owned `assets/pi5/config.txt` (vendor settings +
    `os_check=0`), superseding the vendor file hashed at
    `9c34ec9c0eee...` — see the bring-up checklist for why.

## EEPROM requirement (matched-pair rule)

D0 boards need the Pi EEPROM from **2025-06-09 or later** — older EEPROMs
break the UEFI framebuffer on D0 (confirmed by the fork author with the
Raspberry Pi Foundation). Update via Raspberry Pi OS
(`sudo rpi-eeprom-update -a`) *before* first UEFI boot, then **stop
updating the EEPROM blindly**: EEPROM changes have broken UEFI graphics
before. Record the working EEPROM version next to the firmware hash above
when bring-up succeeds.

## Board support

- **D0 stepping** (all 2 GB/16 GB, late-2024+ 4/8 GB, CM5): the pinned
  master build. This is the revision we test on (user board: D0, 8 GB —
  all three models fit, including Qwen3 4B).
- **C1 stepping** (2023/early-2024 4/8 GB): the fork keeps a `C1` branch;
  the archived worproject v0.3 also covers it. Supported by tooling,
  **untested by us** — no C1 board available.

## Known limitations (from the fork README + upstream)

- Ethernet, GPIO control, PWM, EEPROM access, CM5 eMMC: not functional in
  UEFI (irrelevant to NightRun — we use GOP, USB keyboard, SD/FAT, MP).
- SD cards run up to SDR104 (~90 MB/s best case); a ~2 GB model loads in
  roughly 25–60 s depending on the card. A1/A2-rated cards recommended.
- PCIe/NVMe exists (Gen 2 default) — possible fast-model-storage upgrade
  path, untested with NightRun.
- Serial console: the Pi 5 **dedicated 3-pin debug UART connector**
  (between the HDMI ports, JST-SH 1.0 mm), *not* GPIO 14/15. 115200 8n1.
- Power: 5 V/3 A minimum, official 25 W supply recommended; active
  cooling required for sustained inference (the SoC throttles at 85 °C).
- Fan header: OS-managed on Pi 5 — off under UEFI until NightRun's RP1
  driver forces it to 100% (nr-boot/src/fan.rs; verified working on the
  D0 board 2026-07-07). No thermal curve: when you can't regulate,
  overcool.
- `/bye` (UEFI shutdown) goes through PSCI; behavior on real hardware
  (power-off vs reboot) to be observed at bring-up and recorded here.

## Bring-up checklist (R5 — real D0 board, run in order)

Stage gates; record each result (and the UART transcript) here. Serial:
3-pin debug UART connector, 115200 8n1.

1. **EEPROM**: from Pi OS, `sudo rpi-eeprom-update` — must be 2025-06-09
   or later (update with `-a` if older, then note the version here).
2. **Flash**: `sudo dd if=nightrun-pi5.img of=/dev/sdX bs=4M oflag=direct status=progress && sync`.
3. **Firmware boots**: QR screen → Pi logo + progress bar (fork firmware
   alive). Real-hardware findings from the D0 bring-up (2026-07-07),
   each producing a black screen or bootloader stop:
   - "installed OS does not indicate support for Raspberry Pi 5" →
     `os_check=0` missing (newer EEPROMs apply an OS-support check UEFI
     armstubs can't satisfy). Our `assets/pi5/config.txt` includes it.
   - Bootloader probes `kernel_2712.img`/`kernel8.img` then dies →
     **DTBs missing**: the Pi 5 bootloader refuses to start any armstub
     without a board-matching `bcm2712*.dtb` on the FAT root. pi-image
     ships all Pi 5 variants + `overlays/bcm2712d0.dtbo` (pinned from
     raspberrypi/firmware @ 958bfb0a).
   - Diagnosis tool: press Esc during the bootloader phase — its HDMI
     log shows config parsing and every fs_open attempt.
   Never hand-edit files on the card; rebuild with `cargo xtask
   pi-image` and re-flash.
4. **NightRun boots**: `[nightrun] vX.Y.Z boot layer up` on UART;
   `[cpu] aarch64 neon baseline; dotprod=true fp16=true` expected on A76.
5. **GOP**: synthwave splash on HDMI; note the mode
   (`[vid] ...` serial line).
6. **Keyboard**: any key advances the splash; typed text echoes in the
   input bar.
7. **Storage**: model load progress + MB/s (expect ~40-90 MB/s SDR104;
   ~25-60 s for Granite); streaming CRC must pass.
8. **MP services**: `[smp] N workers` — record N (4 expected). If
   "no MP services protocol": single-core fallback works but note it.
9. **Generation**: prompt → reply; record pp/ftl/tok/s from the status
   bar; NEON is active by default.
10. **Chat commands**: multi-turn, /clear, /bye (record whether PSCI
    shutdown powers off or reboots).
11. **Sustained bench**: 3+ long generations back-to-back with the
    cooling noted (active cooler / heatsink / bare); record tok/s drift
    (throttling) into docs/benchmarks.md under a "Pi 5" section.

Expected decode (LPDDR4X ~10-13 GB/s usable): Llama 1B ≈ 7-9 tok/s,
Granite 3B ≈ 4.5-6 tok/s. Numbers are recorded when measured, not before.

## Merge gate status (do not merge feature/rpi5-support until all true)

- [x] x86 tests + parity green after every shared change (51 tests)
- [x] x86 benches within noise of docs/baseline-x86.md (5.8s boot / 57 tok/s pp re-measured after R1)
- [x] aarch64 kernels bit-identical to scalar (qemu-user test rig)
- [x] full engine e2e on aarch64 UEFI in QEMU (chat + generation)
- [ ] x86 USB boot re-verified on real hardware
- [x] Pi 5 boots NightRun via UEFI (real D0 board, 2026-07-07)
- [x] GOP / keyboard / storage / timer / RAM load / MP verified on-device
      (4c shown; Granite loaded + streaming CRC passed from SD)
- [x] ≥1 model RAM-resident generating locally on the Pi
      (Granite 3B: pp 6.2 tok/s, ftl 1.9 s, decode 3.0 tok/s)
- [x] no Linux / host process / streamed weights anywhere in the Pi path
- [x] firmware pins, EEPROM pair, board assumptions documented
- [ ] Pi benchmarks with thermal conditions documented (numbers taken;
      cooling setup + sustained drift still to record)
