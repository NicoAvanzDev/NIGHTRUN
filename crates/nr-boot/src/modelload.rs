//! Load model.nrm from the boot volume into RAM (firmware pages that stay
//! resident for the whole session).

use core::sync::atomic::{AtomicBool, Ordering};

use uefi::boot::{self, AllocateType, MemoryType};

/// Set once the model is resident: any further attempt to read model data
/// from storage is a hard fault. This enforces the RAM-residency contract
/// structurally.
static STORAGE_SEALED: AtomicBool = AtomicBool::new(false);

/// Seal storage after the model is resident in RAM.
pub fn seal_storage() {
    STORAGE_SEALED.store(true, Ordering::SeqCst);
    crate::serial_println!("[boot] storage sealed - model reads from disk are now forbidden");
}
use uefi::proto::media::file::{File, FileAttribute, FileInfo, FileMode};
use uefi::CString16;

#[derive(Debug)]
#[allow(dead_code)] // fields are read via Debug on the failure screen
pub enum LoadError {
    Fs(uefi::Status),
    NotFound,
    /// Streaming checksum mismatch (file corrupt on the boot medium).
    Corrupt(nr_model::verify::VerifyError),
    OutOfMemory,
    ReadFailed(uefi::Status),
}

const CHUNK: usize = 16 * 1024 * 1024;

/// Reads `\model.nrm` from the volume NightRun booted from, reporting
/// (bytes_done, bytes_total) after each chunk.
pub fn load(progress: &mut dyn FnMut(usize, usize)) -> Result<&'static mut [u8], LoadError> {
    assert!(
        !STORAGE_SEALED.load(Ordering::SeqCst),
        "model storage is sealed: no reads after RAM residency"
    );
    let mut fs =
        boot::get_image_file_system(boot::image_handle()).map_err(|e| LoadError::Fs(e.status()))?;
    let mut root = fs.open_volume().map_err(|e| LoadError::Fs(e.status()))?;
    let path = CString16::try_from("model.nrm").unwrap();
    let handle = root
        .open(&path, FileMode::Read, FileAttribute::empty())
        .map_err(|_| LoadError::NotFound)?;
    let mut file = handle.into_regular_file().ok_or(LoadError::NotFound)?;

    let info = file
        .get_boxed_info::<FileInfo>()
        .map_err(|e| LoadError::Fs(e.status()))?;
    let size = info.file_size() as usize;
    progress(0, size);

    let pages = size.div_ceil(4096);
    let base = boot::allocate_pages(AllocateType::AnyPages, MemoryType::LOADER_DATA, pages)
        .map_err(|_| LoadError::OutOfMemory)?;
    // SAFETY: freshly allocated region of `pages * 4096 >= size` bytes.
    let buf = unsafe { core::slice::from_raw_parts_mut(base.as_ptr(), size) };

    // Checksums are computed while the chunks stream in — no second pass
    // over the model after loading.
    let mut verifier: Option<nr_model::verify::StreamingVerifier> = None;
    let mut done = 0;
    while done < size {
        let end = (done + CHUNK).min(size);
        let n = file
            .read(&mut buf[done..end])
            .map_err(|e| LoadError::ReadFailed(e.status()))?;
        if n == 0 {
            return Err(LoadError::ReadFailed(uefi::Status::END_OF_FILE));
        }
        let prev = done;
        done += n;
        if verifier.is_none() && done >= nr_model::format::HEADER_SIZE {
            verifier = Some(
                nr_model::verify::StreamingVerifier::new(&buf[..done])
                    .map_err(LoadError::Corrupt)?,
            );
            // The first feed covers everything read so far.
            verifier.as_mut().unwrap().feed(0, &buf[..done]);
        } else if let Some(v) = verifier.as_mut() {
            v.feed(prev, &buf[prev..done]);
        }
        progress(done, size);
    }
    match verifier {
        Some(v) => v.finish().map_err(LoadError::Corrupt)?,
        None => return Err(LoadError::ReadFailed(uefi::Status::END_OF_FILE)),
    }
    Ok(buf)
}
