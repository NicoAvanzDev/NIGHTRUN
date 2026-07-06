//! Embedded Spleen PSF2 fonts (BSD licensed, see assets/fonts/SPLEEN-LICENSE).

use nr_gfx::PsfFont;

static SMALL: &[u8] = include_bytes!("../../../assets/fonts/spleen-8x16.psfu");
static BODY: &[u8] = include_bytes!("../../../assets/fonts/spleen-12x24.psfu");
static HEAD: &[u8] = include_bytes!("../../../assets/fonts/spleen-16x32.psfu");
static BIG: &[u8] = include_bytes!("../../../assets/fonts/spleen-32x64.psfu");

pub struct Fonts {
    pub small: PsfFont<'static>,
    pub body: PsfFont<'static>,
    pub head: PsfFont<'static>,
    pub big: PsfFont<'static>,
}

impl Fonts {
    pub fn load() -> Fonts {
        Fonts {
            small: PsfFont::parse(SMALL).expect("spleen 8x16"),
            body: PsfFont::parse(BODY).expect("spleen 12x24"),
            head: PsfFont::parse(HEAD).expect("spleen 16x32"),
            big: PsfFont::parse(BIG).expect("spleen 32x64"),
        }
    }
}
