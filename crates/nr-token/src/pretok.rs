//! Pretokenizer: a hand-rolled implementation of the Llama-3 split regex
//!
//!   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//!   |[^\r\n\p{L}\p{N}]?\p{L}+
//!   |\p{N}{1,3}
//!   | ?[^\s\p{L}\p{N}]+[\r\n]*
//!   |\s*[\r\n]+
//!   |\s+(?!\S)
//!   |\s+
//!
//! Character classes use core's Unicode tables (`is_alphabetic`,
//! `is_numeric`, `is_whitespace`); alternatives are tried in regex order.
//!
//! The Qwen2 family regex is identical except numbers match a single
//! `\p{N}` (not `{1,3}`).

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Style {
    Llama3,
    Qwen2,
}

pub struct Chunks<'a> {
    rest: &'a str,
    style: Style,
}

pub fn chunks(text: &str, style: Style) -> Chunks<'_> {
    Chunks { rest: text, style }
}

impl<'a> Iterator for Chunks<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        if self.rest.is_empty() {
            return None;
        }
        let len = match_one(self.rest, self.style);
        let (chunk, rest) = self.rest.split_at(len);
        self.rest = rest;
        Some(chunk)
    }
}

#[inline]
fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

#[inline]
fn is_num(c: char) -> bool {
    c.is_numeric()
}

#[inline]
fn is_ws(c: char) -> bool {
    c.is_whitespace()
}

/// Byte length of the chunk starting at the beginning of (non-empty) `s`.
fn match_one(s: &str, style: Style) -> usize {
    let c0 = s.chars().next().unwrap();
    let b = s.as_bytes();

    // 1. Contractions: (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if c0 == '\'' && s.len() >= 2 {
        match b[1].to_ascii_lowercase() {
            b's' | b't' | b'm' | b'd' => return 2,
            c1 @ (b'r' | b'v' | b'l') if s.len() >= 3 => {
                let c2 = b[2].to_ascii_lowercase();
                if (c1 == b'r' && c2 == b'e') || (c1 == b'v' && c2 == b'e') || (c1 == b'l' && c2 == b'l') {
                    return 3;
                }
            }
            _ => {}
        }
    }

    // 2. [^\r\n\p{L}\p{N}]?\p{L}+
    {
        let letters_from = if is_letter(c0) {
            Some(0)
        } else if c0 != '\r' && c0 != '\n' && !is_num(c0) {
            // Optional prefix char; requires a letter right after.
            let after = c0.len_utf8();
            match s[after..].chars().next() {
                Some(c) if is_letter(c) => Some(after),
                _ => None,
            }
        } else {
            None
        };
        if let Some(start) = letters_from {
            let mut end = start;
            for c in s[start..].chars() {
                if is_letter(c) {
                    end += c.len_utf8();
                } else {
                    break;
                }
            }
            return end;
        }
    }

    // 3. \p{N}{1,3} (llama3) / single \p{N} (qwen2)
    if is_num(c0) {
        let max = match style {
            Style::Llama3 => 3,
            Style::Qwen2 => 1,
        };
        let mut end = 0;
        let mut n = 0;
        for c in s.chars() {
            if n < max && is_num(c) {
                end += c.len_utf8();
                n += 1;
            } else {
                break;
            }
        }
        return end;
    }

    // 4. ` ?[^\s\p{L}\p{N}]+[\r\n]*`
    {
        let start = if c0 == ' ' { 1 } else { 0 };
        let mut end = start;
        for c in s[start..].chars() {
            if !is_ws(c) && !is_letter(c) && !is_num(c) {
                end += c.len_utf8();
            } else {
                break;
            }
        }
        if end > start {
            for c in s[end..].chars() {
                if c == '\r' || c == '\n' {
                    end += 1;
                } else {
                    break;
                }
            }
            return end;
        }
    }

    // 5-7. Whitespace runs.
    if is_ws(c0) {
        let mut run_end = 0;
        for c in s.chars() {
            if is_ws(c) {
                run_end += c.len_utf8();
            } else {
                break;
            }
        }
        let run = &s[..run_end];
        // 5. \s*[\r\n]+ : up to and including the last newline of the run.
        if let Some(last_nl) = run.rfind(['\r', '\n']) {
            return last_nl + 1;
        }
        // 6. \s+(?!\S): keep the last whitespace char for the next word.
        if run_end < s.len() {
            let last_len = run.chars().next_back().unwrap().len_utf8();
            if run_end > last_len {
                return run_end - last_len;
            }
        }
        // 7. \s+ (whole run: at end of text, or a single ws before \S).
        return run_end;
    }

    // Unreachable for text covered by the classes above; be safe anyway.
    c0.len_utf8()
}
