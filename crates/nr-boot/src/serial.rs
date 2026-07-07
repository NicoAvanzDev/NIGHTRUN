//! Debug serial output, per platform: x86 writes COM1 (0x3F8) directly
//! (115200 8N1, no firmware involvement); aarch64 goes through the UEFI
//! Serial I/O protocol (PL011 on QEMU virt and the Pi 5 port), attached
//! once boot services are up and silently dropped if absent.

use core::fmt::{self, Write};

#[cfg(target_arch = "x86_64")]
mod imp {
    use core::arch::asm;

    const COM1: u16 = 0x3f8;

    #[inline]
    unsafe fn outb(port: u16, val: u8) {
        asm!("out dx, al", in("dx") port, in("al") val, options(nostack, nomem));
    }

    #[inline]
    unsafe fn inb(port: u16) -> u8 {
        let val: u8;
        asm!("in al, dx", in("dx") port, out("al") val, options(nostack, nomem));
        val
    }

    pub fn init() {
        unsafe {
            outb(COM1 + 1, 0x00); // disable interrupts
            outb(COM1 + 3, 0x80); // DLAB on
            outb(COM1, 0x01); // divisor 1 -> 115200
            outb(COM1 + 1, 0x00);
            outb(COM1 + 3, 0x03); // 8N1
            outb(COM1 + 2, 0xc7); // FIFO on, clear, 14-byte threshold
            outb(COM1 + 4, 0x0b); // DTR + RTS + OUT2
        }
    }

    /// Port I/O needs no boot services; nothing to attach.
    pub fn attach() {}

    pub fn write_byte(b: u8) {
        unsafe {
            while inb(COM1 + 5) & 0x20 == 0 {}
            outb(COM1, b);
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod imp {
    use core::sync::atomic::{AtomicPtr, Ordering};
    use uefi::proto::console::serial::Serial;

    static SERIAL: AtomicPtr<Serial> = AtomicPtr::new(core::ptr::null_mut());

    /// Nothing usable before boot services are initialized.
    pub fn init() {}

    /// Locate the firmware's Serial I/O protocol; logging stays a no-op
    /// if the platform has none.
    pub fn attach() {
        let Ok(handle) = uefi::boot::get_handle_for_protocol::<Serial>() else {
            return;
        };
        let params = uefi::boot::OpenProtocolParams {
            handle,
            agent: uefi::boot::image_handle(),
            controller: None,
        };
        // GetProtocol (non-exclusive): the firmware console may also own
        // this port; we only ever append output bytes.
        let Ok(serial) = (unsafe {
            uefi::boot::open_protocol::<Serial>(
                params,
                uefi::boot::OpenProtocolAttributes::GetProtocol,
            )
        }) else {
            return;
        };
        let leaked: &'static mut Serial = alloc::boxed::Box::leak(alloc::boxed::Box::new(serial))
            .get_mut()
            .expect("serial protocol interface");
        SERIAL.store(leaked, Ordering::Release);
    }

    pub fn write_byte(b: u8) {
        let ptr = SERIAL.load(Ordering::Acquire);
        if ptr.is_null() {
            return;
        }
        // SAFETY: set once from a leaked ScopedProtocol; single-threaded
        // (only the BSP logs; AP workers never call firmware).
        let serial = unsafe { &mut *ptr };
        let _ = serial.write(&[b]);
    }
}

pub fn init() {
    imp::init();
}

/// Call once after `uefi::helpers::init()`.
pub fn attach() {
    imp::attach();
}

pub struct SerialPort;

impl Write for SerialPort {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for b in s.bytes() {
            if b == b'\n' {
                imp::write_byte(b'\r');
            }
            imp::write_byte(b);
        }
        Ok(())
    }
}

#[macro_export]
macro_rules! serial_println {
    ($($arg:tt)*) => {{
        use core::fmt::Write as _;
        let _ = writeln!($crate::serial::SerialPort, $($arg)*);
    }};
}
