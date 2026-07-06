//! NightRun visual identity: outrun / sunset / synthwave palette.

/// Near-black indigo backdrop.
pub const BG_DEEP: u32 = 0x07020f;
/// Slightly lighter panel background.
pub const BG_PANEL: u32 = 0x120829;
/// Panel border / separator lines.
pub const BORDER: u32 = 0x3d1d6e;

pub const NEON_MAGENTA: u32 = 0xff2d95;
pub const NEON_PINK: u32 = 0xff71ce;
pub const NEON_ORANGE: u32 = 0xff9e3d;
pub const NEON_YELLOW: u32 = 0xffd319;
pub const NEON_CYAN: u32 = 0x00e5ff;
pub const NEON_PURPLE: u32 = 0x9d4edd;

pub const TEXT_PRIMARY: u32 = 0xede7ff;
pub const TEXT_DIM: u32 = 0x8f86b8;

/// Sunset sky gradient (position 0..=1000, color), top to horizon.
pub const SKY_STOPS: &[(u32, u32)] = &[
    (0, 0x050110),
    (380, 0x160a33),
    (620, 0x3b1155),
    (820, 0x71175f),
    (1000, 0xa62057),
];

/// Sun disc gradient, top to bottom.
pub const SUN_STOPS: &[(u32, u32)] = &[
    (0, 0xffe95e),
    (450, 0xffb02e),
    (750, 0xff5e62),
    (1000, 0xff2d95),
];

/// Logo lettering gradient, top to bottom.
pub const LOGO_STOPS: &[(u32, u32)] = &[
    (0, 0xfdfbff),
    (220, 0xffe95e),
    (520, 0xff9e3d),
    (780, 0xff2d95),
    (1000, 0xc21bb4),
];

/// Ground gradient below the horizon.
pub const GROUND_STOPS: &[(u32, u32)] = &[
    (0, 0x2c0b45),
    (350, 0x1b0630),
    (1000, 0x090114),
];
