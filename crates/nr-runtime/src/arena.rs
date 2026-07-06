//! Bump arena over a raw memory region (firmware-allocated pages).
//!
//! All large, long-lived buffers (model weights, KV cache, scratch) come
//! from arenas so the generation path never touches the heap.

use core::mem::{align_of, size_of};

pub struct Arena {
    base: *mut u8,
    size: usize,
    offset: usize,
}

// The arena is only used from the single boot processor.
unsafe impl Send for Arena {}

impl Arena {
    /// # Safety
    /// `base..base+size` must be exclusively owned, writable memory that
    /// outlives the arena.
    pub unsafe fn new(base: *mut u8, size: usize) -> Arena {
        Arena { base, size, offset: 0 }
    }

    pub fn capacity(&self) -> usize {
        self.size
    }

    pub fn used(&self) -> usize {
        self.offset
    }

    pub fn alloc_bytes(&mut self, len: usize, align: usize) -> Option<&'static mut [u8]> {
        debug_assert!(align.is_power_of_two());
        let start = self.offset.checked_add(align - 1)? & !(align - 1);
        let end = start.checked_add(len)?;
        if end > self.size {
            return None;
        }
        self.offset = end;
        // SAFETY: range is in-bounds and never handed out twice.
        unsafe { Some(core::slice::from_raw_parts_mut(self.base.add(start), len)) }
    }

    /// Allocate a zeroed slice of `n` elements of a plain-data type.
    pub fn alloc_zeroed<T: Copy>(&mut self, n: usize) -> Option<&'static mut [T]> {
        let bytes = self.alloc_bytes(n * size_of::<T>(), align_of::<T>().max(64))?;
        bytes.fill(0);
        // SAFETY: correctly sized, aligned, zero is a valid T (T: Copy plain data).
        unsafe {
            Some(core::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, n))
        }
    }
}
