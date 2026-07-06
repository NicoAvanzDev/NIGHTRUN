//! Multi-core bring-up via EFI_MP_SERVICES_PROTOCOL: application
//! processors are started once into the nr-tensor worker pool (pure
//! compute + atomics; APs never call firmware services).

use core::ffi::c_void;
use core::time::Duration;

use uefi::boot;
use uefi::proto::pi::mp::MpServices;

extern "efiapi" fn worker_entry(_arg: *mut c_void) {
    // Each AP needs its own AVX enable (XCR0 is per-core).
    crate::enable_avx_quiet();
    nr_tensor::parallel::POOL.worker_loop();
}

/// Start all APs as inference workers. Returns the number of workers
/// activated (0 when MP services are unavailable or single-core).
pub fn start_workers() -> usize {
    let Ok(handle) = boot::get_handle_for_protocol::<MpServices>() else {
        serial_println!("[smp] no MP services protocol");
        return 0;
    };
    let Ok(mp) = boot::open_protocol_exclusive::<MpServices>(handle) else {
        serial_println!("[smp] MP services busy");
        return 0;
    };
    let Ok(count) = mp.get_number_of_processors() else {
        return 0;
    };
    let aps = count.enabled.saturating_sub(1);
    serial_println!("[smp] {} processors ({} enabled), {} APs", count.total, count.enabled, aps);
    if aps == 0 {
        return 0;
    }

    // Fire-and-forget: the APs never return, so use the non-blocking mode
    // with a dummy event and keep the protocol open forever.
    let event = unsafe {
        boot::create_event(boot::EventType::empty(), boot::Tpl::APPLICATION, None, None)
    };
    let Ok(event) = event else { return 0 };
    let started = unsafe {
        mp.startup_all_aps(false, worker_entry, core::ptr::null_mut(), Some(event), None)
    };
    if let Err(e) = started {
        serial_println!("[smp] startup_all_aps failed: {:?}", e.status());
        return 0;
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
        return 0;
    }
    nr_tensor::parallel::POOL.activate(ready);
    serial_println!("[smp] {} workers active", ready);
    ready
}
