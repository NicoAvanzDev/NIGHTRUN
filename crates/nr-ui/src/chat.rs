//! Chat screen: status bar, scrollback with `user:` / `llama:` turns,
//! input line with cursor.

use alloc::string::String;
use alloc::vec::Vec;

use nr_gfx::{draw, theme, Surface};

use crate::fonts::Fonts;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Llama,
    System,
}

impl Role {
    pub fn prefix(&self) -> &'static str {
        match self {
            Role::User => "user: ",
            Role::Llama => "llama: ",
            Role::System => "",
        }
    }

    fn color(&self) -> u32 {
        match self {
            Role::User => theme::NEON_CYAN,
            Role::Llama => theme::NEON_PINK,
            Role::System => theme::TEXT_DIM,
        }
    }
}

pub struct Turn {
    pub role: Role,
    pub text: String,
}

pub struct Stats<'a> {
    pub model: &'a str,
    pub mem_used_mb: u32,
    pub mem_total_mb: u32,
    /// Generation rate, milli-tokens per second; 0 hides the readout.
    pub tok_s_milli: u32,
    /// Prompt-processing rate, milli-tokens per second; 0 hides it.
    pub pp_milli: u32,
    pub ctx_used: u32,
    pub ctx_max: u32,
    pub cores: u32,
    pub generating: bool,
}

const MARGIN: i32 = 24;
const STATUS_H: i32 = 40;
const INPUT_H: i32 = 46;

pub fn draw(
    surf: &mut Surface,
    fonts: &Fonts,
    turns: &[Turn],
    input: &str,
    cursor_on: bool,
    stats: &Stats,
) {
    surf.clear(theme::BG_DEEP);
    status_bar(surf, fonts, stats);
    input_bar(surf, fonts, input, cursor_on, stats.generating);
    scrollback(surf, fonts, turns, stats.generating);
    draw::scanlines(surf, 26);
}

fn status_bar(surf: &mut Surface, fonts: &Fonts, stats: &Stats) {
    let w = surf.width as i32;
    surf.fill_rect(0, 0, w, STATUS_H, theme::BG_PANEL);
    // Sunset separator line.
    for x in 0..w {
        let c = nr_gfx::color::gradient(
            &[(0, theme::NEON_CYAN), (500, theme::NEON_MAGENTA), (1000, theme::NEON_ORANGE)],
            x as u32 * 1000 / w.max(1) as u32,
        );
        surf.fill_rect(x, STATUS_H - 2, 1, 2, c);
    }

    let f = &fonts.small;
    let ty = (STATUS_H - f.height as i32) / 2 - 1;
    draw::text_glow(surf, f, MARGIN, ty, "NIGHTRUN", 1, 2, 0, theme::NEON_MAGENTA, 2);
    draw::text(surf, f, MARGIN, ty, "NIGHTRUN", theme::TEXT_PRIMARY, 1, 2);

    let model_x = MARGIN + draw::text_width(f, "NIGHTRUN", 1, 2) + 28;
    let mut buf = String::new();
    buf.push_str("// ");
    buf.push_str(stats.model);
    draw::text(surf, f, model_x, ty, &buf, theme::TEXT_DIM, 1, 0);

    // Right side: cores | memory | context | prefill | generation rate.
    let mut right = String::new();
    fmt_u32(&mut right, stats.cores);
    right.push_str("c  ");
    fmt_mem(&mut right, stats.mem_used_mb, stats.mem_total_mb);
    right.push_str("  ctx ");
    fmt_u32(&mut right, stats.ctx_used);
    right.push('/');
    fmt_u32(&mut right, stats.ctx_max);
    right.push_str("  ");
    if stats.pp_milli > 0 {
        right.push_str("pp ");
        fmt_milli(&mut right, stats.pp_milli);
        right.push_str("  ");
    }
    if stats.tok_s_milli > 0 {
        fmt_milli(&mut right, stats.tok_s_milli);
        right.push_str(" tok/s");
    } else if stats.generating {
        right.push_str("... tok/s");
    } else {
        right.push_str("idle");
    }
    let rx = surf.width as i32 - MARGIN - draw::text_width(f, &right, 1, 0);
    let rate_col = if stats.generating { theme::NEON_YELLOW } else { theme::TEXT_DIM };
    draw::text(surf, f, rx, ty, &right, rate_col, 1, 0);
}

fn input_bar(surf: &mut Surface, fonts: &Fonts, input: &str, cursor_on: bool, generating: bool) {
    let w = surf.width as i32;
    let h = surf.height as i32;
    let y0 = h - INPUT_H;
    surf.fill_rect(0, y0, w, INPUT_H, theme::BG_PANEL);
    surf.fill_rect(0, y0, w, 1, theme::BORDER);

    let f = &fonts.body;
    let ty = y0 + (INPUT_H - f.height as i32) / 2;
    if generating {
        draw::text(surf, f, MARGIN, ty, ">> generating - press ESC to stop", theme::TEXT_DIM, 1, 0);
        return;
    }
    draw::text(surf, f, MARGIN, ty, "user: ", theme::NEON_CYAN, 1, 0);
    let tx = MARGIN + draw::text_width(f, "user: ", 1, 0);

    // Show the tail of the input if it overflows.
    let max_cols = ((w - tx - MARGIN - f.width as i32) / f.width as i32).max(1) as usize;
    let shown: String = if input.chars().count() > max_cols {
        input.chars().skip(input.chars().count() - max_cols).collect()
    } else {
        input.into()
    };
    draw::text(surf, f, tx, ty, &shown, theme::TEXT_PRIMARY, 1, 0);
    if cursor_on {
        let cx = tx + draw::text_width(f, &shown, 1, 0) + 2;
        surf.fill_rect(cx, ty + 2, f.width as i32 - 2, f.height as i32 - 4, theme::NEON_CYAN);
    }
}

fn scrollback(surf: &mut Surface, fonts: &Fonts, turns: &[Turn], generating: bool) {
    let f = &fonts.body;
    let w = surf.width as i32;
    let h = surf.height as i32;
    let cols = ((w - 2 * MARGIN) / f.width as i32).max(10) as usize;
    let top = STATUS_H + 10;
    let bottom = h - INPUT_H - 10;

    // Wrap all turns into (color, indent, line) records.
    let mut lines: Vec<(u32, usize, String)> = Vec::new();
    for (i, turn) in turns.iter().enumerate() {
        if i > 0 {
            lines.push((0, 0, String::new())); // blank separator
        }
        let prefix = turn.role.prefix();
        let indent = prefix.chars().count();
        let body_cols = cols.saturating_sub(indent).max(8);
        let streaming_tail = generating && i == turns.len() - 1 && turn.role == Role::Llama;
        let mut first = true;
        for line in wrap(&turn.text, body_cols, streaming_tail) {
            if first {
                let mut s = String::from(prefix);
                s.push_str(&line);
                lines.push((turn.role.color(), 0, s));
                first = false;
            } else {
                lines.push((theme::TEXT_PRIMARY, indent, line));
            }
        }
        if first {
            // Empty turn (streaming just started): show bare prefix.
            lines.push((turn.role.color(), 0, String::from(prefix.trim_end())));
        }
    }

    // Bottom-anchored: keep the last lines that fit.
    let line_h = f.height as i32 + 2;
    let max_lines = ((bottom - top) / line_h).max(1) as usize;
    let start = lines.len().saturating_sub(max_lines);
    let mut y = top;
    for (color, indent, line) in &lines[start..] {
        if !line.is_empty() {
            // Prefix in role color, continuation text in primary.
            let has_prefix = *indent == 0 && (line.starts_with("user:") || line.starts_with("llama:"));
            if has_prefix {
                let split = line.find(' ').map(|i| i + 1).unwrap_or(line.len());
                let (pre, rest) = line.split_at(split);
                draw::text(surf, f, MARGIN, y, pre, *color, 1, 0);
                draw::text(surf, f, MARGIN + draw::text_width(f, pre, 1, 0), y, rest, theme::TEXT_PRIMARY, 1, 0);
            } else {
                let x = MARGIN + *indent as i32 * f.width as i32;
                let c = if *indent > 0 { theme::TEXT_PRIMARY } else { *color };
                draw::text(surf, f, x, y, line, c, 1, 0);
            }
        }
        y += line_h;
    }
}

/// Word wrap to `cols` columns. If `cursor_tail`, append a streaming marker.
fn wrap(text: &str, cols: usize, cursor_tail: bool) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for para in text.split('\n') {
        let mut line = String::new();
        for word in para.split(' ') {
            let wlen = word.chars().count();
            let llen = line.chars().count();
            if llen == 0 {
                push_word(&mut out, &mut line, word, cols);
            } else if llen + 1 + wlen <= cols {
                line.push(' ');
                line.push_str(word);
            } else {
                out.push(core::mem::take(&mut line));
                push_word(&mut out, &mut line, word, cols);
            }
        }
        out.push(line);
    }
    if cursor_tail {
        if let Some(last) = out.last_mut() {
            if last.chars().count() + 1 <= cols {
                last.push('_');
            } else {
                out.push(String::from("_"));
            }
        }
    }
    out
}

/// Push a word onto the current line, hard-splitting if longer than `cols`.
fn push_word(out: &mut Vec<String>, line: &mut String, word: &str, cols: usize) {
    let mut rest: Vec<char> = word.chars().collect();
    while rest.len() > cols {
        let chunk: String = rest.drain(..cols).collect();
        out.push(chunk);
    }
    line.extend(rest);
}

fn fmt_mem(out: &mut String, used: u32, total: u32) {
    fmt_u32(out, used);
    out.push('/');
    fmt_u32(out, total);
    out.push_str(" MB");
}

fn fmt_u32(out: &mut String, v: u32) {
    let mut buf = [0u8; 10];
    let mut i = buf.len();
    let mut v = v;
    loop {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    out.push_str(core::str::from_utf8(&buf[i..]).unwrap());
}

fn fmt_milli(out: &mut String, milli: u32) {
    fmt_u32(out, milli / 1000);
    out.push('.');
    fmt_u32(out, (milli % 1000) / 100);
}
