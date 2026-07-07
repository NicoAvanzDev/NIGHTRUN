//! Boot splash: outrun backdrop + NIGHTRUN logo.

use nr_gfx::{color, draw, theme, Surface};

use crate::fonts::Fonts;

pub const LOGO: &str = "NIGHTRUN";
pub const TAGLINE: &str = "BARE-METAL LLM RUNTIME";

/// Draw the full splash. `status` is the footer line (e.g. version / hint).
pub fn draw(surf: &mut Surface, fonts: &Fonts, status: &str) {
    crate::backdrop::draw(surf, 660);
    logo(surf, fonts);
    footer(surf, fonts, status);
    draw::scanlines(surf, 36);
}

fn logo(surf: &mut Surface, fonts: &Fonts) {
    let w = surf.width as i32;
    let h = surf.height as i32;

    // Headline lettering: sheared, sunset-gradient fill over a dark drop
    // shadow, wrapped in a magenta glow.
    let scale = if w >= 1600 { 3 } else { 2 };
    let tracking = 6 * scale as i32;
    let shear = 10 * scale as i32;
    let tw = draw::text_width(&fonts.big, LOGO, scale, tracking);
    let x = (w - tw) / 2;
    let y = h * 17 / 100;
    let font = &fonts.big;

    draw::text_glow(
        surf,
        font,
        x,
        y,
        LOGO,
        scale,
        tracking,
        shear,
        theme::NEON_MAGENTA,
        4 * scale as i32,
    );
    // Drop shadow.
    let sh = 3 * scale as i32;
    draw::text_fx(
        surf,
        font,
        x + sh,
        y + sh,
        LOGO,
        scale,
        tracking,
        shear,
        |_, _| 0x14042a,
        230,
    );
    // Gradient fill with a darkened "chrome" band around 55% height.
    draw::text_fx(
        surf,
        font,
        x,
        y,
        LOGO,
        scale,
        tracking,
        shear,
        |row, rows| {
            let t = (row as u32) * 1000 / rows as u32;
            let c = color::gradient(theme::LOGO_STOPS, t);
            if (520..580).contains(&t) {
                color::scale(c, 120)
            } else {
                c
            }
        },
        255,
    );

    // Tagline with wide tracking, cyan glow.
    let tag_scale = 1;
    let tag_tracking = fonts.head.width as i32;
    let tt = draw::text_width(&fonts.head, TAGLINE, tag_scale, tag_tracking);
    let tx = (w - tt) / 2;
    let ty = y + (font.height * scale) as i32 + h * 4 / 100;
    draw::text_glow(
        surf,
        &fonts.head,
        tx,
        ty,
        TAGLINE,
        tag_scale,
        tag_tracking,
        0,
        theme::NEON_CYAN,
        2,
    );
    draw::text(
        surf,
        &fonts.head,
        tx,
        ty,
        TAGLINE,
        0xccf9ff,
        tag_scale,
        tag_tracking,
    );

    // Separator rules on both sides of the tagline.
    let rule_y = ty + (fonts.head.height / 2) as i32;
    let gap = 24;
    draw::line(surf, x, rule_y, tx - gap, rule_y, theme::NEON_CYAN, 110);
    draw::line(
        surf,
        tx + tt + gap,
        rule_y,
        x + tw,
        rule_y,
        theme::NEON_CYAN,
        110,
    );
}

fn footer(surf: &mut Surface, fonts: &Fonts, status: &str) {
    let w = surf.width as i32;
    let h = surf.height as i32;
    let tw = draw::text_width(&fonts.small, status, 1, 1);
    draw::text(
        surf,
        &fonts.small,
        (w - tw) / 2,
        h - 28,
        status,
        theme::TEXT_DIM,
        1,
        1,
    );
}
