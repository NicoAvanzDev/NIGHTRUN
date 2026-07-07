//! Raspberry Pi 5 cooling-fan bring-up (aarch64 only).
//!
//! The Pi 5 fan header is driven by the RP1's PWM block and is normally
//! managed by the OS — under UEFI nothing programs it, so the fan stays
//! off while inference heats the SoC (real-hardware finding). NightRun
//! has no thermal management, so we do the safe thing: drive the fan at
//! 100%. The PWM input is active-low (PWM_POLARITY_INVERTED in the Pi's
//! device tree), so a constant low level = full speed.
//!
//! Mechanism: find the RP1 (PCIe endpoint 1de4:0001) through the PCI
//! root bridge, then configure RP1 GPIO45 (FAN_PWM) as a SYS_RIO output
//! driven low. Register layout per the public RP1 datasheet: GPIO bank 2
//! holds GPIO 34..=53; each bank has IO (status/ctrl), SYS_RIO
//! (out/oe/in with +0x2000 SET / +0x3000 CLR atomic aliases) and PADS
//! blocks. Only GPIO45's registers are touched — nothing else on RP1
//! (USB lives there too).

use uefi::proto::pci::root_bridge::PciRootBridgeIo;
use uefi::proto::pci::PciIoAddress;

const RP1_ID: u32 = (0x0001u32 << 16) | 0x1de4; // device | vendor

const IO_BANK2: u64 = 0x0d_8000;
const SYS_RIO2: u64 = 0x0e_8000;
const PADS_BANK2: u64 = 0x0f_8000;
const ATOM_SET: u64 = 0x2000;
const ATOM_CLR: u64 = 0x3000;

/// GPIO45 = FAN_PWM; bank 2 starts at GPIO34.
const PIN: u64 = 45 - 34;
/// CTRL.FUNCSEL value routing the pin to SYS_RIO.
const FUNCSEL_SYS_RIO: u32 = 5;
/// Pad: output enabled (OD=0), input disabled, 8 mA drive, no pulls.
const PAD_OUTPUT_8MA: u32 = 0x10;

/// EFI PCI address encoding: bus[31:24] dev[23:16] fun[15:8] reg[7:0].
fn cfg(bus: u8, dev: u8, reg: u8) -> PciIoAddress {
    PciIoAddress::from(((bus as u64) << 24) | ((dev as u64) << 16) | reg as u64)
}

/// Force the fan to 100%. Returns true when the RP1 was found and
/// programmed. Harmless no-op elsewhere (QEMU virt has no RP1).
pub fn spin_up() -> bool {
    let Ok(handles) = uefi::boot::find_handles::<PciRootBridgeIo>() else {
        return false;
    };
    for handle in handles {
        let params = uefi::boot::OpenProtocolParams {
            handle,
            agent: uefi::boot::image_handle(),
            controller: None,
        };
        // Shared access: exclusive would disconnect the bus drivers
        // (including RP1's xHCI — our keyboard).
        // SAFETY: GetProtocol; we only read config space and poke
        // GPIO45's registers, which no firmware driver manages.
        let Ok(mut rb) = (unsafe {
            uefi::boot::open_protocol::<PciRootBridgeIo>(
                params,
                uefi::boot::OpenProtocolAttributes::GetProtocol,
            )
        }) else {
            continue;
        };

        for bus in 0u8..8 {
            for dev in 0u8..32 {
                let Ok(id) = rb.pci().read_one::<u32>(cfg(bus, dev, 0x00)) else {
                    continue;
                };
                if id != RP1_ID {
                    continue;
                }
                // BAR1 (offset 0x14) holds the 16 MB peripheral window;
                // handle a 64-bit BAR (high half in BAR2's slot).
                let Ok(bar_lo) = rb.pci().read_one::<u32>(cfg(bus, dev, 0x14)) else {
                    continue;
                };
                let mut base = (bar_lo & 0xffff_fff0) as u64;
                if bar_lo & 0x6 == 0x4 {
                    if let Ok(bar_hi) = rb.pci().read_one::<u32>(cfg(bus, dev, 0x18)) {
                        base |= (bar_hi as u64) << 32;
                    }
                }
                if base == 0 {
                    continue;
                }
                serial_println!("[fan] RP1 at {bus:02x}:{dev:02x}, peri base {base:#x}");

                let mut w = |off: u64, val: u32| {
                    let _ = rb.memory().write_one::<u32>(base + off, val);
                };
                // Order: pad + RIO level first, then route the pin, so
                // the fan line never glitches high (fan off) mid-setup.
                w(PADS_BANK2 + 0x04 + PIN * 4, PAD_OUTPUT_8MA);
                w(SYS_RIO2 + ATOM_CLR, 1 << PIN); // OUT low = 100% (active-low)
                w(SYS_RIO2 + ATOM_SET + 0x04, 1 << PIN); // OE on
                w(IO_BANK2 + PIN * 8 + 4, FUNCSEL_SYS_RIO); // CTRL.FUNCSEL
                serial_println!("[fan] GPIO45 -> RIO output low (fan 100%)");
                return true;
            }
        }
    }
    serial_println!("[fan] RP1 not found - no fan control");
    false
}
