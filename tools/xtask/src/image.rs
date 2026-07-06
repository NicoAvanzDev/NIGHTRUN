//! Build nightrun.img: GPT disk with a single FAT32 ESP holding
//! EFI/BOOT/BOOTX64.EFI and model.nrm.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

const LB: u64 = 512;

pub fn build(img_path: &Path, efi: &Path, model: Option<&Path>) {
    let model_size = model.map(|m| std::fs::metadata(m).expect("model file").len()).unwrap_or(0);
    let efi_size = std::fs::metadata(efi).expect("efi binary").len();

    // ESP sized for contents + FAT32 overhead + slack.
    let contents = model_size + efi_size;
    let esp_bytes = (contents + contents / 50 + 64 * 1024 * 1024).next_multiple_of(1024 * 1024);
    // GPT: MBR + primary header/table (34 lb) up front, backup (33 lb) at end.
    let disk_bytes = esp_bytes + 4 * 1024 * 1024;

    println!(
        "image: {} ({} MB esp, model {} MB)",
        img_path.display(),
        esp_bytes / (1024 * 1024),
        model_size / (1024 * 1024)
    );

    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(img_path)
        .expect("create image");
    file.set_len(disk_bytes).unwrap();

    // Protective MBR + fresh GPT.
    let mbr = gpt::mbr::ProtectiveMBR::with_lb_size(
        u32::try_from(disk_bytes / LB - 1).unwrap_or(0xffff_ffff),
    );
    mbr.overwrite_lba0(&mut file).expect("write mbr");

    let mut disk = gpt::GptConfig::new()
        .writable(true)
        .initialized(false)
        .logical_block_size(gpt::disk::LogicalBlockSize::Lb512)
        .open_from_device(Box::new(&mut file))
        .expect("open gpt");
    disk.update_partitions(Default::default()).expect("init partitions");
    disk.add_partition(
        "NIGHTRUN",
        esp_bytes,
        gpt::partition_types::EFI,
        0,
        None,
    )
    .expect("add esp");
    let (start_lba, end_lba) = {
        let p = &disk.partitions()[&1];
        (p.first_lba, p.last_lba)
    };
    disk.write().expect("write gpt");

    // Format the partition as FAT32 and copy files in.
    let start = start_lba * LB;
    let end = (end_lba + 1) * LB;
    let slice = fscommon::StreamSlice::new(&mut file, start, end).unwrap();
    let mut buf = fscommon::BufStream::new(slice);
    fatfs::format_volume(
        &mut buf,
        fatfs::FormatVolumeOptions::new().volume_label(*b"NIGHTRUN   "),
    )
    .expect("format fat");

    let fs = fatfs::FileSystem::new(buf, fatfs::FsOptions::new()).expect("mount fat");
    {
        let root = fs.root_dir();
        let boot = root.create_dir("EFI").unwrap().create_dir("BOOT").unwrap();
        copy_into(&boot, "BOOTX64.EFI", efi);
        if let Some(model) = model {
            copy_into(&root, "model.nrm", model);
        }
    }
    fs.unmount().expect("unmount");
    file.flush().unwrap();
    println!("image ready: {}", img_path.display());
}

fn copy_into<IO: fatfs::ReadWriteSeek>(
    dir: &fatfs::Dir<'_, IO>,
    name: &str,
    src: &Path,
) {
    let mut src_file = std::fs::File::open(src).expect("open source");
    let mut dst = dir.create_file(name).expect("create in image");
    dst.truncate().unwrap();
    let mut buf = vec![0u8; 16 * 1024 * 1024];
    let total = src_file.metadata().unwrap().len();
    let mut done = 0u64;
    loop {
        let n = src_file.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        dst.write_all(&buf[..n]).unwrap();
        done += n as u64;
        if total > 64 * 1024 * 1024 {
            print!("\r  {name}: {} / {} MB", done / (1024 * 1024), total / (1024 * 1024));
            std::io::stdout().flush().unwrap();
        }
    }
    if total > 64 * 1024 * 1024 {
        println!();
    }
    dst.flush().unwrap();
}

/// Overwrite just BOOTX64.EFI inside an existing image (fast dev loop).
pub fn update_efi(img_path: &Path, efi: &Path) -> bool {
    let Ok(mut file) = std::fs::OpenOptions::new().read(true).write(true).open(img_path) else {
        return false;
    };
    let Ok(disk) = gpt::GptConfig::new()
        .writable(false)
        .logical_block_size(gpt::disk::LogicalBlockSize::Lb512)
        .open_from_device(Box::new(&mut file))
    else {
        return false;
    };
    let Some(p) = disk.partitions().get(&1).cloned() else { return false };
    drop(disk);
    file.seek(SeekFrom::Start(0)).unwrap();
    let slice = fscommon::StreamSlice::new(&mut file, p.first_lba * LB, (p.last_lba + 1) * LB).unwrap();
    let buf = fscommon::BufStream::new(slice);
    let Ok(fs) = fatfs::FileSystem::new(buf, fatfs::FsOptions::new()) else { return false };
    {
        let root = fs.root_dir();
        let Ok(boot) = root.open_dir("EFI/BOOT") else { return false };
        copy_into(&boot, "BOOTX64.EFI", efi);
    }
    fs.unmount().is_ok()
}
