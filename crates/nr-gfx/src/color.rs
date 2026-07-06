//! Color helpers over 0x00RRGGBB u32 values.

#[inline]
pub const fn rgb(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

#[inline]
pub const fn channels(c: u32) -> (u8, u8, u8) {
    (((c >> 16) & 0xff) as u8, ((c >> 8) & 0xff) as u8, (c & 0xff) as u8)
}

/// Linear interpolation between two colors, `t` in 0..=256 (fixed point).
#[inline]
pub fn lerp(a: u32, b: u32, t: u32) -> u32 {
    let t = t.min(256);
    let (ar, ag, ab) = channels(a);
    let (br, bg, bb) = channels(b);
    let mix = |x: u8, y: u8| -> u8 {
        (((x as u32) * (256 - t) + (y as u32) * t) >> 8) as u8
    };
    rgb(mix(ar, br), mix(ag, bg), mix(ab, bb))
}

/// Sample a multi-stop gradient. `stops` are (position 0..=1000, color),
/// sorted ascending. `t` in 0..=1000.
pub fn gradient(stops: &[(u32, u32)], t: u32) -> u32 {
    match stops {
        [] => 0,
        [only] => only.1,
        _ => {
            if t <= stops[0].0 {
                return stops[0].1;
            }
            for w in stops.windows(2) {
                let (p0, c0) = w[0];
                let (p1, c1) = w[1];
                if t <= p1 {
                    let span = (p1 - p0).max(1);
                    return lerp(c0, c1, (t - p0) * 256 / span);
                }
            }
            stops[stops.len() - 1].1
        }
    }
}

/// Scale color brightness by `f` in 0..=256.
#[inline]
pub fn scale(c: u32, f: u32) -> u32 {
    let (r, g, b) = channels(c);
    rgb(
        ((r as u32 * f) >> 8).min(255) as u8,
        ((g as u32 * f) >> 8).min(255) as u8,
        ((b as u32 * f) >> 8).min(255) as u8,
    )
}
