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

- (pending first build on this machine)

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
  master build. This is the revision we test on (user board: `d04170`,
  4 GB).
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
- `/bye` (UEFI shutdown) goes through PSCI; behavior on real hardware
  (power-off vs reboot) to be observed at bring-up and recorded here.
