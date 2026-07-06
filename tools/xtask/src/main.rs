//! NightRun build/run automation.
//!
//! Commands:
//!   cargo xtask build                       build BOOTX64.EFI and stage the ESP dir
//!   cargo xtask run [opts]                  boot the staged ESP in QEMU + OVMF
//!     --window            show a display window (default: headless)
//!     --mem <size>        guest RAM (default 2G)
//!     --secs <n>          quit QEMU after n seconds
//!     --shot <t>:<path>   screendump PNG at t seconds (repeatable)
//!     --keys <t>:<text>   type text at t seconds (repeatable; "\n" = Enter)

mod image;

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("build") => {
            build_and_stage();
        }
        Some("image") => {
            build_image(true);
        }
        Some("run") => run(parse_run_opts(&args[1..])),
        _ => {
            eprintln!("usage: cargo xtask <build|image|run> [options]");
            std::process::exit(2);
        }
    }
}

/// Build nightrun.img. When `fresh` (or no image exists) the whole image is
/// rebuilt with the model; otherwise only BOOTX64.EFI is refreshed.
fn build_image(fresh: bool) -> PathBuf {
    let root = root();
    let (_, efi) = build_and_stage();
    let img = root.join("nightrun.img");
    if !fresh && img.exists() && image::update_efi(&img, &efi) {
        println!("updated BOOTX64.EFI in {}", img.display());
        return img;
    }
    let model = root.join("models/model.nrm");
    let model = model.exists().then_some(model);
    if model.is_none() {
        println!("note: models/model.nrm missing - building image without model");
    }
    image::build(&img, &efi, model.as_deref());
    img
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn build_and_stage() -> (PathBuf, PathBuf) {
    let root = root();
    // Custom hard-float UEFI target (the builtin one is soft-float, which
    // both breaks AVX intrinsics and would cripple f32 math), so build
    // core/alloc from source on nightly.
    let status = Command::new("cargo")
        .current_dir(&root)
        .env_remove("CARGO") // don't let the outer stable cargo leak in
        .args([
            "+nightly",
            "build",
            "--release",
            "-p",
            "nr-boot",
            "-Zbuild-std=core,alloc",
            "-Zjson-target-spec",
            "--target",
            "x86_64-nightrun-uefi.json",
        ])
        .status()
        .expect("run cargo (is nightly installed? rustup toolchain install nightly --component rust-src)");
    assert!(status.success(), "nr-boot build failed");

    let efi = root.join("target/x86_64-nightrun-uefi/release/nr-boot.efi");
    let esp = root.join("target/esp");
    let boot_dir = esp.join("EFI/BOOT");
    std::fs::create_dir_all(&boot_dir).unwrap();
    std::fs::copy(&efi, boot_dir.join("BOOTX64.EFI")).unwrap();
    println!("staged {}", esp.display());
    (esp, efi)
}

#[derive(Default)]
struct RunOpts {
    window: bool,
    img: bool,
    mem: Option<String>,
    secs: Option<u64>,
    shots: Vec<(u64, String)>,
    keys: Vec<(u64, String)>,
}

fn parse_run_opts(args: &[String]) -> RunOpts {
    let mut o = RunOpts::default();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().expect("missing value").clone();
        match a.as_str() {
            "--window" => o.window = true,
            "--img" => o.img = true,
            "--mem" => o.mem = Some(val()),
            "--secs" => o.secs = Some(val().parse().unwrap()),
            "--shot" => {
                let v = val();
                let (t, p) = v.split_once(':').expect("--shot t:path");
                o.shots.push((t.parse().unwrap(), p.into()));
            }
            "--keys" => {
                let v = val();
                let (t, k) = v.split_once(':').expect("--keys t:text");
                o.keys.push((t.parse().unwrap(), k.replace("\\n", "\n")));
            }
            other => panic!("unknown option {other}"),
        }
    }
    o
}

fn run(opts: RunOpts) {
    let root = root();
    // --img boots the real GPT/FAT32 image (required for the model, which
    // exceeds QEMU's virtual-FAT limits); the default boots the staged ESP
    // directory for a fast dev loop.
    let boot_drive = if opts.img {
        let img = build_image(false);
        format!("format=raw,file={}", img.display())
    } else {
        let (esp, _) = build_and_stage();
        format!("format=raw,file=fat:rw:{}", esp.display())
    };
    let target = root.join("target");

    let ovmf_code = "/usr/share/OVMF/OVMF_CODE_4M.fd";
    let vars = target.join("OVMF_VARS.fd");
    if !vars.exists() {
        std::fs::copy("/usr/share/OVMF/OVMF_VARS_4M.fd", &vars).expect("copy OVMF vars");
    }

    let qmp_sock = target.join("qmp.sock");
    let _ = std::fs::remove_file(&qmp_sock);
    let serial_log = target.join("serial.log");

    let mut cmd = Command::new("qemu-system-x86_64");
    cmd.current_dir(&root)
        .args(["-machine", "q35"])
        .args(["-accel", "kvm", "-accel", "tcg"])
        .args(["-cpu", "max"])
        .args(["-m", opts.mem.as_deref().unwrap_or("2G")])
        .args(["-drive", &format!("if=pflash,format=raw,readonly=on,file={ovmf_code}")])
        .args(["-drive", &format!("if=pflash,format=raw,file={}", vars.display())])
        .args(["-drive", &boot_drive])
        .args(["-serial", &format!("file:{}", serial_log.display())])
        .args(["-qmp", &format!("unix:{},server=on,wait=off", qmp_sock.display())])
        .args(["-monitor", "none"]);
    if !opts.window {
        cmd.arg("-display").arg("none");
    }
    println!("qemu: {:?}", cmd.get_args().collect::<Vec<_>>().join(" ".as_ref()));
    let mut child = cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit()).spawn().expect("spawn qemu");

    let mut qmp = Qmp::connect(&qmp_sock, Duration::from_secs(10));

    // Timeline of scheduled actions.
    let mut events: Vec<(u64, Event)> = Vec::new();
    for (t, p) in &opts.shots {
        events.push((*t, Event::Shot(p.clone())));
    }
    for (t, k) in &opts.keys {
        events.push((*t, Event::Keys(k.clone())));
    }
    events.sort_by_key(|(t, _)| *t);

    let start = Instant::now();
    for (t, ev) in events {
        let due = Duration::from_secs(t);
        if let Some(rem) = due.checked_sub(start.elapsed()) {
            std::thread::sleep(rem);
        }
        match ev {
            Event::Shot(path) => {
                if let Some(qmp) = qmp.as_mut() {
                    let abs = root.join(&path);
                    qmp.screendump(abs.to_str().unwrap());
                    println!("screendump -> {path}");
                }
            }
            Event::Keys(text) => {
                if let Some(qmp) = qmp.as_mut() {
                    qmp.type_text(&text);
                    println!("typed {text:?}");
                }
            }
        }
    }

    if let Some(secs) = opts.secs {
        if let Some(rem) = Duration::from_secs(secs).checked_sub(start.elapsed()) {
            std::thread::sleep(rem);
        }
        if let Some(qmp) = qmp.as_mut() {
            qmp.cmd(r#"{"execute":"quit"}"#);
        }
        wait_or_kill(&mut child, Duration::from_secs(5));
    } else {
        let _ = child.wait();
    }

    if let Ok(log) = std::fs::read_to_string(&serial_log) {
        let tail: Vec<&str> = log.lines().rev().take(30).collect();
        println!("--- serial tail ---");
        for line in tail.iter().rev() {
            println!("{line}");
        }
    }
}

enum Event {
    Shot(String),
    Keys(String),
}

fn wait_or_kill(child: &mut Child, timeout: Duration) {
    let start = Instant::now();
    loop {
        if let Ok(Some(_)) = child.try_wait() {
            return;
        }
        if start.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

struct Qmp {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
}

impl Qmp {
    fn connect(path: &Path, timeout: Duration) -> Option<Qmp> {
        let start = Instant::now();
        let stream = loop {
            match UnixStream::connect(path) {
                Ok(s) => break s,
                Err(_) if start.elapsed() < timeout => {
                    std::thread::sleep(Duration::from_millis(100));
                }
                Err(e) => {
                    eprintln!("qmp connect failed: {e}");
                    return None;
                }
            }
        };
        let writer = stream.try_clone().unwrap();
        let mut qmp = Qmp { reader: BufReader::new(stream), writer };
        qmp.read_line(); // greeting
        qmp.cmd(r#"{"execute":"qmp_capabilities"}"#);
        Some(qmp)
    }

    fn read_line(&mut self) -> String {
        let mut line = String::new();
        let _ = self.reader.read_line(&mut line);
        line
    }

    /// Send a command and read until its response (skipping async events).
    fn cmd(&mut self, json: &str) -> String {
        let _ = writeln!(self.writer, "{json}");
        loop {
            let line = self.read_line();
            if line.is_empty() || line.contains("\"return\"") || line.contains("\"error\"") {
                return line;
            }
        }
    }

    fn screendump(&mut self, path: &str) {
        let resp = self.cmd(&format!(
            r#"{{"execute":"screendump","arguments":{{"filename":"{path}","format":"png"}}}}"#
        ));
        if resp.contains("\"error\"") {
            eprintln!("screendump failed: {resp}");
        }
    }

    fn sendkey(&mut self, key: &str) {
        self.cmd(&format!(
            r#"{{"execute":"human-monitor-command","arguments":{{"command-line":"sendkey {key}"}}}}"#
        ));
        std::thread::sleep(Duration::from_millis(35));
    }

    fn type_text(&mut self, text: &str) {
        for ch in text.chars() {
            let key = match ch {
                'a'..='z' | '0'..='9' => ch.to_string(),
                'A'..='Z' => format!("shift-{}", ch.to_ascii_lowercase()),
                ' ' => "spc".into(),
                '\n' => "ret".into(),
                '.' => "dot".into(),
                ',' => "comma".into(),
                '-' => "minus".into(),
                '_' => "shift-minus".into(),
                '/' => "slash".into(),
                '?' => "shift-slash".into(),
                '!' => "shift-1".into(),
                ':' => "shift-semicolon".into(),
                ';' => "semicolon".into(),
                '\'' => "apostrophe".into(),
                _ => continue,
            };
            self.sendkey(&key);
        }
    }
}
