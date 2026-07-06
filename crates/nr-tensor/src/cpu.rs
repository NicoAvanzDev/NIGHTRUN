//! CPUID feature detection (works in no_std / UEFI and on the host).
//!
//! AVX use additionally requires XCR0 to enable YMM state; on bare metal
//! nr-boot performs the XSETBV dance, on an OS the kernel already has.

use core::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Features {
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
}

static CACHED: AtomicU8 = AtomicU8::new(0);

const PROBED: u8 = 1;
const AVX2: u8 = 2;
const FMA: u8 = 4;
const F16C: u8 = 8;

pub fn features() -> Features {
    let mut bits = CACHED.load(Ordering::Relaxed);
    if bits & PROBED == 0 {
        bits = probe();
        CACHED.store(bits, Ordering::Relaxed);
    }
    Features { avx2: bits & AVX2 != 0, fma: bits & FMA != 0, f16c: bits & F16C != 0 }
}

/// True when the fast (AVX2+FMA) kernel paths can be used.
pub fn fast_path() -> bool {
    let f = features();
    f.avx2 && f.fma
}

fn probe() -> u8 {
    use core::arch::x86_64::{__cpuid, __cpuid_count};

    let mut bits = PROBED;
    let leaf1 = __cpuid(1);
    let osxsave = leaf1.ecx & (1 << 27) != 0;
    let avx = leaf1.ecx & (1 << 28) != 0;
    if !(osxsave && avx) {
        return bits;
    }
    // Check the OS/firmware enabled YMM state (XCR0 bits 1|2).
    let xcr0: u64 = unsafe {
        let lo: u32;
        let hi: u32;
        core::arch::asm!("xgetbv", in("ecx") 0u32, out("eax") lo, out("edx") hi, options(nostack, nomem));
        ((hi as u64) << 32) | lo as u64
    };
    if xcr0 & 0b110 != 0b110 {
        return bits;
    }
    if leaf1.ecx & (1 << 12) != 0 {
        bits |= FMA;
    }
    if leaf1.ecx & (1 << 29) != 0 {
        bits |= F16C;
    }
    let leaf7 = __cpuid_count(7, 0);
    if leaf7.ebx & (1 << 5) != 0 {
        bits |= AVX2;
    }
    bits
}
