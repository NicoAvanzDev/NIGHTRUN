//! Multi-core bring-up via EFI_MP_SERVICES_PROTOCOL: application
//! processors are started once into the nr-tensor worker pool (pure
//! compute + atomics; APs never call firmware services).

use core::ffi::c_void;
use core::time::Duration;

use uefi::boot;
use uefi::proto::pi::mp::MpServices;

extern "efiapi" fn worker_entry(_arg: *mut c_void) {
    // Each AP needs its own AVX enable (XCR0 is per-core).
    crate::enable_simd_quiet();
    nr_tensor::parallel::POOL.worker_loop();
}

/// Start all APs as inference workers. Returns the number of workers
/// activated and a short status note for the boot screen (important on
/// hardware without a serial hookup).
pub fn start_workers() -> (usize, &'static str) {
    let Ok(handle) = boot::get_handle_for_protocol::<MpServices>() else {
        serial_println!("[smp] no MP services protocol");
        return (0, "no MP services protocol");
    };
    let Ok(mp) = boot::open_protocol_exclusive::<MpServices>(handle) else {
        serial_println!("[smp] MP services busy");
        return (0, "MP services busy");
    };
    let Ok(count) = mp.get_number_of_processors() else {
        return (0, "MP processor count failed");
    };
    let aps = count.enabled.saturating_sub(1);
    serial_println!("[smp] {} processors ({} enabled), {} APs", count.total, count.enabled, aps);
    if aps == 0 {
        return (0, "single core");
    }

    // The workers never return from their procedure, so the startup call
    // can never "complete" — how we dispatch differs per platform:
    //  - x86 (CpuDxe/MpInitLib): non-blocking with a dummy event; its
    //    blocking mode resets stragglers on timeout, which would kill
    //    the workers.
    //  - aarch64 (ArmPsciMpServicesDxe): non-blocking is refused after
    //    READY_TO_BOOT (i.e. always, for a boot app) — real-hardware
    //    finding. Blocking with a short timeout instead: PSCI cannot
    //    preempt an AP, so EFI_TIMEOUT just hands control back with the
    //    workers left running.
    let started = if cfg!(target_arch = "x86_64") {
        let event = unsafe {
            boot::create_event(boot::EventType::empty(), boot::Tpl::APPLICATION, None, None)
        };
        let Ok(event) = event else { return (0, "event create failed") };
        unsafe {
            mp.startup_all_aps(false, worker_entry, core::ptr::null_mut(), Some(event), None)
        }
    } else {
        match unsafe {
            mp.startup_all_aps(
                false,
                worker_entry,
                core::ptr::null_mut(),
                None,
                Some(Duration::from_millis(120)),
            )
        } {
            // Timeout is the expected outcome: workers keep running.
            Err(e) if e.status() == uefi::Status::TIMEOUT => Ok(()),
            other => other,
        }
    };
    if let Err(e) = started {
        serial_println!("[smp] startup_all_aps failed: {:?}", e.status());
        return (0, "AP startup failed");
    }
    core::mem::forget(mp);

    // Wait for every AP to reach the spin loop before activating the pool.
    let deadline = 500u32;
    let mut waited = 0;
    while nr_tensor::parallel::POOL.ready_workers() < aps && waited < deadline {
        boot::stall(Duration::from_millis(2));
        waited += 1;
    }
    let ready = nr_tensor::parallel::POOL.ready_workers().min(aps);
    if ready == 0 {
        serial_println!("[smp] no workers came up");
        return (0, "workers did not come up");
    }
    nr_tensor::parallel::POOL.activate(ready);
    serial_println!("[smp] {} workers active", ready);
    (ready, "ok")
}
