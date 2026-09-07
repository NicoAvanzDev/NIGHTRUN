use super::{prepare_image, Arch};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "nightrun-image-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(root.join("models")).unwrap();
        std::fs::create_dir(root.join("target")).unwrap();
        std::fs::write(root.join("models/model.nrm"), b"llama model").unwrap();
        std::fs::write(
            root.join("models/ternary-bonsai.nrm"),
            b"ternary bonsai model",
        )
        .unwrap();
        std::fs::write(root.join("boot.efi"), b"old firmware").unwrap();
        Self(root)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn read_image_file(img: &Path, name: &str) -> Vec<u8> {
    let mut file = std::fs::File::open(img).unwrap();
    let disk = gpt::GptConfig::new()
        .writable(false)
        .open_from_device(Box::new(&mut file))
        .unwrap();
    let p = disk.partitions()[&1].clone();
    drop(disk);
    file.seek(SeekFrom::Start(0)).unwrap();
    let slice =
        fscommon::StreamSlice::new(&mut file, p.first_lba * 512, (p.last_lba + 1) * 512).unwrap();
    let fs = fatfs::FileSystem::new(slice, fatfs::FsOptions::new()).unwrap();
    let mut bytes = Vec::new();
    fs.root_dir()
        .open_file(name)
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    bytes
}

#[test]
fn run_keeps_selected_model_even_without_source_or_sidecar() {
    for arch in [Arch::X86, Arch::Aarch64] {
        let fixture = Fixture::new();
        let root = &fixture.0;
        let efi = root.join("boot.efi");
        let img = prepare_image(root, &efi, true, Some("models/ternary-bonsai.nrm"), arch);
        std::fs::remove_file(root.join("models/ternary-bonsai.nrm")).unwrap();
        std::fs::remove_dir_all(root.join("target")).unwrap();
        std::fs::write(&efi, b"new firmware").unwrap();
        assert_eq!(prepare_image(root, &efi, false, None, arch), img);
        assert_eq!(read_image_file(&img, "model.nrm"), b"ternary bonsai model");
        assert_eq!(
            read_image_file(&img, &format!("EFI/BOOT/{}", arch.boot_file())),
            b"new firmware"
        );
    }
}

#[test]
fn first_run_uses_default_and_explicit_model_switches_it() {
    let fixture = Fixture::new();
    let root = &fixture.0;
    let efi = root.join("boot.efi");
    let img = prepare_image(root, &efi, false, None, Arch::X86);
    assert_eq!(read_image_file(&img, "model.nrm"), b"llama model");
    prepare_image(
        root,
        &efi,
        false,
        Some("models/ternary-bonsai.nrm"),
        Arch::X86,
    );
    assert_eq!(read_image_file(&img, "model.nrm"), b"ternary bonsai model");
}

#[test]
fn missing_explicit_model_does_not_replace_existing_image() {
    let fixture = Fixture::new();
    let root = &fixture.0;
    let efi = root.join("boot.efi");
    let img = prepare_image(
        root,
        &efi,
        true,
        Some("models/ternary-bonsai.nrm"),
        Arch::X86,
    );
    assert!(std::panic::catch_unwind(|| {
        prepare_image(root, &efi, false, Some("models/missing.nrm"), Arch::X86);
    })
    .is_err());
    assert_eq!(read_image_file(&img, "model.nrm"), b"ternary bonsai model");
}
