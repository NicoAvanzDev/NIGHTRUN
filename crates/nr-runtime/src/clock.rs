//! Monotonic tick clock, calibrated once against a firmware stall.
//! Tick source per architecture: x86 TSC, aarch64 generic timer.

#[derive(Clone, Copy)]
pub struct Clock {
    ticks_per_ms: u64,
}

/// Read the raw monotonic tick counter (name kept from the x86-only days).
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn rdtsc() -> u64 {
    // SAFETY: RDTSC is unprivileged reading of the timestamp counter.
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// Read the raw monotonic tick counter (aarch64 generic timer, CNTVCT_EL0
/// — unprivileged, constant-rate).
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn rdtsc() -> u64 {
    let ticks: u64;
    // SAFETY: CNTVCT_EL0 reads are permitted at all ELs UEFI runs at.
    unsafe { core::arch::asm!("mrs {}, cntvct_el0", out(reg) ticks, options(nostack, nomem)) };
    ticks
}

impl Clock {
    /// Calibrate using a known-duration stall (microseconds).
    pub fn calibrate(stall_us: impl Fn(u64)) -> Clock {
        let t0 = rdtsc();
        stall_us(20_000);
        let t1 = rdtsc();
        Clock { ticks_per_ms: ((t1 - t0) / 20).max(1) }
    }

    #[inline]
    pub fn now(&self) -> u64 {
        rdtsc()
    }

    #[inline]
    pub fn ticks_to_ms(&self, dt: u64) -> u64 {
        dt / self.ticks_per_ms
    }

    /// Milli-units per second given `count` events over `dt` ticks
    /// (e.g. tokens -> milli-tokens/sec).
    pub fn rate_milli(&self, count: u64, dt_ticks: u64) -> u32 {
        let ms = self.ticks_to_ms(dt_ticks).max(1);
        (count * 1_000_000 / ms) as u32
    }
}
