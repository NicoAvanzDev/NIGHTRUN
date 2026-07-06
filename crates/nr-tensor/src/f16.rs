//! IEEE 754 half-precision conversion (softfloat; no F16C needed).

pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let out = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                // Subnormal: value = frac * 2^-24. Normalize the 10-bit frac.
                let msb = 31 - frac.leading_zeros(); // 0..=9
                let exp = 127 - 24 + msb;
                let frac = (frac << (23 - msb)) & 0x7f_ffff;
                sign | (exp << 23) | frac
            }
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13), // inf / nan
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(out)
}

pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) as u16) << 15;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7f_ffff;

    if exp == 0xff {
        // Inf / NaN.
        return sign | 0x7c00 | if frac != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if e <= 0 {
        if e < -10 {
            return sign; // underflow -> zero
        }
        // Subnormal with round-to-nearest.
        let frac = frac | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        return sign | ((frac + half) >> shift) as u16;
    }
    // Normal with round-to-nearest-even.
    let mut out = sign as u32 | ((e as u32) << 10) | (frac >> 13);
    let round = frac & 0x1fff;
    if round > 0x1000 || (round == 0x1000 && out & 1 != 0) {
        out += 1;
    }
    out as u16
}
