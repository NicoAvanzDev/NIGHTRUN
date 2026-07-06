//! Model-loading screen: backdrop + translucent panel with staged checklist
//! and a neon progress bar.

use nr_gfx::{color, draw, theme, Surface};

use crate::fonts::Fonts;

pub const BAR_STOPS: &[(u32, u32)] = &[
    (0, theme::NEON_CYAN),
    (500, theme::NEON_MAGENTA),
    (1000, theme::NEON_ORANGE),
];

pub struct LoadState<'a> {
    pub stages: &'a [&'a str],
    /// Index of the stage in progress; stages before it render as done.
    pub current: usize,
    /// Progress of the current stage, per-mille.
    pub progress_pm: u32,
    /// Detail line under the bar, e.g. "812 / 1364 MB".
    pub detail: &'a str,
    /// Animation frame for the working indicator.
    pub frame: u32,
}

pub fn draw(surf: &mut Surface, fonts: &Fonts, st: &LoadState) {
    crate::backdrop::draw(surf, 660);

    let w = surf.width as i32;
    let h = surf.height as i32;
    let pw = w * 56 / 100;
    let ph = h * 52 / 100;
    let px = (w - pw) / 2;
    let py = h * 22 / 100;

    panel(surf, px, py, pw, ph);

    // Title.
    let title = "NIGHTRUN // SYSTEM BOOT";
    let tw = draw::text_width(&fonts.head, title, 1, 2);
    draw::text_glow(surf, &fonts.head, px + (pw - tw) / 2, py + 22, title, 1, 2, 0, theme::NEON_MAGENTA, 2);
    draw::text(surf, &fonts.head, px + (pw - tw) / 2, py + 22, title, theme::TEXT_PRIMARY, 1, 2);
    draw::line(surf, px + 24, py + 66, px + pw - 24, py + 66, theme::BORDER, 255);

    // Stage checklist.
    let font = &fonts.body;
    let mut y = py + 88;
    for (i, stage) in st.stages.iter().enumerate() {
        let (marker, mcol, tcol) = match i.cmp(&st.current) {
            core::cmp::Ordering::Less => ("[ OK ]", theme::NEON_CYAN, theme::TEXT_PRIMARY),
            core::cmp::Ordering::Equal => {
                let dots = (st.frame / 4) % 4;
                let m = match dots {
                    0 => "[    ]",
                    1 => "[ .  ]",
                    2 => "[ .. ]",
                    _ => "[ ...]",
                };
                (m, theme::NEON_MAGENTA, theme::NEON_PINK)
            }
            core::cmp::Ordering::Greater => ("[    ]", theme::BORDER, theme::TEXT_DIM),
        };
        draw::text(surf, font, px + 36, y, marker, mcol, 1, 0);
        draw::text(surf, font, px + 36 + draw::text_width(font, "[ OK ] ", 1, 0), y, stage, tcol, 1, 0);
        y += font.height as i32 + 10;
    }

    // Progress bar.
    let bx = px + 36;
    let bw = pw - 72;
    let by = py + ph - 96;
    let bh = 20;
    let overall = ((st.current as u32 * 1000 + st.progress_pm.min(1000)) / st.stages.len().max(1) as u32).min(1000);
    bar(surf, bx, by, bw, bh, overall, st.frame);

    // Percent + detail.
    let mut pct_buf = [0u8; 8];
    let pct = format_pct(overall, &mut pct_buf);
    draw::text(surf, &fonts.body, bx + bw + 8 - draw::text_width(&fonts.body, pct, 1, 0), by - fonts.body.height as i32 - 6, pct, theme::NEON_CYAN, 1, 0);
    draw::text(surf, &fonts.small, bx, by + bh + 12, st.detail, theme::TEXT_DIM, 1, 0);

    draw::scanlines(surf, 36);
}

fn panel(surf: &mut Surface, x: i32, y: i32, w: i32, h: i32) {
    surf.blend_rect(x, y, w, h, 0x0b0420, 216);
    // Border.
    surf.fill_rect(x, y, w, 1, theme::BORDER);
    surf.fill_rect(x, y + h - 1, w, 1, theme::BORDER);
    surf.fill_rect(x, y, 1, h, theme::BORDER);
    surf.fill_rect(x + w - 1, y, 1, h, theme::BORDER);
    // Neon corner accents.
    let l = 26;
    for (cx, cy, dx, dy) in [
        (x, y, 1, 1),
        (x + w - 1, y, -1, 1),
        (x, y + h - 1, 1, -1),
        (x + w - 1, y + h - 1, -1, -1),
    ] {
        surf.fill_rect(if dx > 0 { cx } else { cx - l + 1 }, cy, l, 2 * dy.max(0) + 1, theme::NEON_CYAN);
        surf.fill_rect(cx, if dy > 0 { cy } else { cy - l + 1 }, 2 * dx.max(0) + 1, l, theme::NEON_CYAN);
    }
}

fn bar(surf: &mut Surface, x: i32, y: i32, w: i32, h: i32, pm: u32, frame: u32) {
    // Track.
    surf.fill_rect(x, y, w, h, 0x1a0b33);
    surf.fill_rect(x, y, w, 1, theme::BORDER);
    surf.fill_rect(x, y + h - 1, w, 1, theme::BORDER);
    // Fill with gradient + travelling highlight.
    let fill = (w as u32 * pm / 1000) as i32;
    for col in 0..fill {
        let c = color::gradient(BAR_STOPS, (col as u32) * 1000 / w.max(1) as u32);
        let hl = (col as u32 + frame * 6) % 160;
        let c = if hl < 22 { color::lerp(c, 0xffffff, 90) } else { c };
        surf.fill_rect(x + col, y + 2, 1, h - 4, c);
    }
    // Glow under the fill edge.
    if fill > 2 {
        for g in 1..=4 {
            surf.blend_rect(x + fill, y + 2, g, h - 4, theme::NEON_ORANGE, (40 / g) as u32);
        }
    }
}

fn format_pct(pm: u32, buf: &mut [u8; 8]) -> &str {
    let pct = pm / 10;
    let mut n = 0;
    if pct >= 100 {
        buf[n] = b'1';
        n += 1;
        buf[n] = b'0' + ((pct / 10) % 10) as u8;
        n += 1;
    } else if pct >= 10 {
        buf[n] = b'0' + (pct / 10) as u8;
        n += 1;
    }
    buf[n] = b'0' + (pct % 10) as u8;
    n += 1;
    buf[n] = b'%';
    n += 1;
    core::str::from_utf8(&buf[..n]).unwrap()
}
