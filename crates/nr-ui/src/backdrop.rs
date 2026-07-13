//! The outrun backdrop: gradient sky, stars, striped sun, perspective grid.

use nr_gfx::{color, draw, theme, Surface};

struct XorShift(u32);

impl XorShift {
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
}

/// Draw the full sunset scene. `horizon_frac_pm` is the horizon height in
/// per-mille of surface height (e.g. 660).
pub fn draw(surf: &mut Surface, horizon_frac_pm: u32) {
    let w = surf.width as i32;
    let h = surf.height as i32;
    let horizon = (h as u32 * horizon_frac_pm / 1000) as i32;

    draw::vertical_gradient(surf, 0, 0, w, horizon, theme::SKY_STOPS);
    draw::vertical_gradient(surf, 0, horizon, w, h - horizon, theme::GROUND_STOPS);

    stars(surf, horizon);
    sun(surf, w / 2, horizon, h * 18 / 100);
    grid(surf, horizon);

    // Glowing horizon line.
    for dy in -2..=2i32 {
        let a = if dy == 0 { 200 } else { 60 / dy.unsigned_abs() };
        surf.blend_rect(0, horizon + dy, w, 1, theme::NEON_MAGENTA, a);
    }
}

fn stars(surf: &mut Surface, horizon: i32) {
    let mut rng = XorShift(0x9e3779b9);
    let w = surf.width as u32;
    let count = surf.width / 6;
    for _ in 0..count {
        let x = (rng.next() % w) as i32;
        let y = (rng.next() % (horizon as u32 * 3 / 4)) as i32;
        let b = 90 + (rng.next() % 130);
        surf.blend(x, y, 0xdcd6ff, b);
        if rng.next().is_multiple_of(7) {
            surf.blend(x + 1, y, 0xdcd6ff, b / 3);
            surf.blend(x - 1, y, 0xdcd6ff, b / 3);
            surf.blend(x, y + 1, 0xdcd6ff, b / 3);
            surf.blend(x, y - 1, 0xdcd6ff, b / 3);
        }
    }
}

/// Retro sun: gradient disc sitting on the horizon, horizontal stripe gaps
/// widening toward the bottom.
fn sun(surf: &mut Surface, cx: i32, horizon: i32, r: i32) {
    let cy = horizon - r * 15 / 100;
    let top = cy - r;
    for y in top..horizon {
        let dy = y - cy;
        let span2 = r * r - dy * dy;
        if span2 <= 0 {
            continue;
        }
        let half = isqrt(span2 as u32) as i32;

        // Stripe mask, walked up from the horizon: the solid disc ends in
        // three floating bars that shrink toward the horizon, completing
        // the classic sunset. Thicknesses are permille of the radius.
        const SEGS: [(i32, bool); 7] = [
            (25, true), // breathing room above the horizon line
            (40, false), // smallest bar
            (45, true),
            (55, false), // middle bar
            (52, true),
            (75, false), // largest bar
            (62, true), // cut between the disc and the bars; solid above
        ];
        let d = horizon - y; // rows above the horizon, >= 1
        let mut acc = 0;
        let mut in_gap = false;
        for (perm, is_gap) in SEGS {
            acc += (perm * r / 1000).max(2);
            if d <= acc {
                in_gap = is_gap;
                break;
            }
        }
        if in_gap {
            continue;
        }

        let c = color::gradient(theme::SUN_STOPS, ((y - top) * 1000 / (2 * r).max(1)) as u32);
        surf.fill_rect(cx - half, y, 2 * half, 1, c);
        // Side glow.
        for g in 1..=12 {
            let a = (36 - g * 3).max(4) as u32;
            surf.blend(cx - half - g, y, c, a);
            surf.blend(cx + half + g - 1, y, c, a);
        }
    }
}

/// Perspective floor grid converging on the vanishing point.
fn grid(surf: &mut Surface, horizon: i32) {
    let w = surf.width as i32;
    let h = surf.height as i32;
    let depth = h - horizon;
    let vp_x = w / 2;

    // Horizontal lines, packed toward the horizon.
    let rows = 11;
    for i in 1..=rows {
        let t = i * i;
        let y = horizon + depth * t / (rows * rows);
        let a = 60 + (i as u32) * 12;
        draw::line(surf, 0, y, w, y, theme::NEON_MAGENTA, a.min(180));
        surf.blend_rect(0, y + 1, w, 1, theme::NEON_MAGENTA, a.min(180) / 4);
    }

    // Radial lines from the vanishing point through evenly spaced floor points.
    let cols = 17;
    for i in 0..=cols {
        let bottom_x = -w + (3 * w) * i / cols;
        draw::line(surf, vp_x, horizon, bottom_x, h, theme::NEON_MAGENTA, 120);
    }
}

fn isqrt(v: u32) -> u32 {
    let mut x = v;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = (x + v / x) / 2;
    }
    if v == 0 {
        0
    } else {
        x
    }
}
