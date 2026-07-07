//! Keyboard input via the firmware's Simple Text Input protocol (works with
//! USB keyboards through the firmware's own USB stack).

use uefi::proto::console::text::{Key, ScanCode};

pub enum InputEvent {
    Char(char),
    Enter,
    Backspace,
    Escape,
    Up,
    Down,
    Left,
    Right,
    Delete,
    PageUp,
    PageDown,
}

/// Non-blocking poll for one key event.
pub fn poll() -> Option<InputEvent> {
    let key = uefi::system::with_stdin(|stdin| stdin.read_key().ok().flatten())?;
    match key {
        Key::Printable(c) => {
            let ch = char::from(c);
            match ch {
                '\r' => Some(InputEvent::Enter),
                '\u{8}' => Some(InputEvent::Backspace),
                c if !c.is_control() => Some(InputEvent::Char(c)),
                _ => None,
            }
        }
        Key::Special(code) => match code {
            ScanCode::ESCAPE => Some(InputEvent::Escape),
            ScanCode::UP => Some(InputEvent::Up),
            ScanCode::DOWN => Some(InputEvent::Down),
            ScanCode::LEFT => Some(InputEvent::Left),
            ScanCode::RIGHT => Some(InputEvent::Right),
            ScanCode::DELETE => Some(InputEvent::Delete),
            ScanCode::PAGE_UP => Some(InputEvent::PageUp),
            ScanCode::PAGE_DOWN => Some(InputEvent::PageDown),
            _ => None,
        },
    }
}
