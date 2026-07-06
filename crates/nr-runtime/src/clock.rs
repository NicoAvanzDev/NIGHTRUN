//! TSC-based wall clock. Calibrated once against a firmware stall.

#[derive(Clone, Copy)]
pub struct Clock {
    ticks_per_ms: u64,
}

#[inline]
pub fn rdtsc() -> u64 {
    // SAFETY: RDTSC is unprivileged reading of the timestamp counter.
    unsafe { core::arch::x86_64::_rdtsc() }
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
