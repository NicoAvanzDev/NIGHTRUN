//! Load model.nrm from the boot volume into RAM (firmware pages that stay
//! resident for the whole session).

use uefi::boot::{self, AllocateType, MemoryType};
use uefi::proto::media::file::{File, FileAttribute, FileInfo, FileMode};
use uefi::CString16;

#[derive(Debug)]
pub enum LoadError {
    Fs(uefi::Status),
    NotFound,
    OutOfMemory,
    ReadFailed(uefi::Status),
}

const CHUNK: usize = 16 * 1024 * 1024;

/// Reads `\model.nrm` from the volume NightRun booted from, reporting
/// (bytes_done, bytes_total) after each chunk.
pub fn load(progress: &mut dyn FnMut(usize, usize)) -> Result<&'static mut [u8], LoadError> {
    let mut fs = boot::get_image_file_system(boot::image_handle())
        .map_err(|e| LoadError::Fs(e.status()))?;
    let mut root = fs.open_volume().map_err(|e| LoadError::Fs(e.status()))?;
    let path = CString16::try_from("model.nrm").unwrap();
    let handle = root
        .open(&path, FileMode::Read, FileAttribute::empty())
        .map_err(|_| LoadError::NotFound)?;
    let mut file = handle.into_regular_file().ok_or(LoadError::NotFound)?;

    let info = file.get_boxed_info::<FileInfo>().map_err(|e| LoadError::Fs(e.status()))?;
    let size = info.file_size() as usize;
    progress(0, size);

    let pages = size.div_ceil(4096);
    let base = boot::allocate_pages(AllocateType::AnyPages, MemoryType::LOADER_DATA, pages)
        .map_err(|_| LoadError::OutOfMemory)?;
    // SAFETY: freshly allocated region of `pages * 4096 >= size` bytes.
    let buf = unsafe { core::slice::from_raw_parts_mut(base.as_ptr(), size) };

    let mut done = 0;
    while done < size {
        let end = (done + CHUNK).min(size);
        let n = file.read(&mut buf[done..end]).map_err(|e| LoadError::ReadFailed(e.status()))?;
        if n == 0 {
            return Err(LoadError::ReadFailed(uefi::Status::END_OF_FILE));
        }
        done += n;
        progress(done, size);
    }
    Ok(buf)
}
