//! Q8_0 quantization: blocks of 32 int8 values with one f16 scale.
//! Byte layout matches GGUF's Q8_0 so converted tensors can be viewed
//! in place.

use crate::f16::{f16_to_f32, f32_to_f16};

pub const QK8_0: usize = 32;

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct BlockQ8_0 {
    /// f16 bits.
    pub d: u16,
    pub qs: [i8; QK8_0],
}

const _: () = assert!(core::mem::size_of::<BlockQ8_0>() == 34);

impl BlockQ8_0 {
    #[inline]
    pub fn scale(&self) -> f32 {
        f16_to_f32(self.d)
    }
}

/// Quantize an f32 slice (length divisible by 32) into Q8_0 blocks.
pub fn quantize(src: &[f32], out: &mut [BlockQ8_0]) {
    assert_eq!(src.len(), out.len() * QK8_0);
    for (block, chunk) in out.iter_mut().zip(src.chunks_exact(QK8_0)) {
        let amax = chunk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        // Quantize against the f16-rounded scale so dequantization sees the
        // exact same d and per-element error stays within d/2.
        let d_bits = f32_to_f16(amax / 127.0);
        let d = f16_to_f32(d_bits);
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        block.d = d_bits;
        for (q, &v) in block.qs.iter_mut().zip(chunk) {
            let x = v * inv;
            // round-to-nearest, clamped
            let r = if x >= 0.0 { x + 0.5 } else { x - 0.5 };
            *q = (r as i32).clamp(-127, 127) as i8;
        }
    }
}

/// Dequantize blocks back to f32 (mainly for tests and embeddings).
pub fn dequantize(blocks: &[BlockQ8_0], out: &mut [f32]) {
    assert_eq!(out.len(), blocks.len() * QK8_0);
    for (block, chunk) in blocks.iter().zip(out.chunks_exact_mut(QK8_0)) {
        let d = block.scale();
        for (o, &q) in chunk.iter_mut().zip(block.qs.iter()) {
            *o = q as f32 * d;
        }
    }
}

/// Reinterpret a raw byte slice as Q8_0 blocks (converter guarantees
/// 64-byte tensor alignment; blocks need 2).
pub fn cast_blocks(bytes: &[u8]) -> &[BlockQ8_0] {
    assert_eq!(bytes.len() % core::mem::size_of::<BlockQ8_0>(), 0);
    // SAFETY: BlockQ8_0 is repr(C, packed), valid for any byte pattern.
    unsafe {
        core::slice::from_raw_parts(
            bytes.as_ptr() as *const BlockQ8_0,
            bytes.len() / core::mem::size_of::<BlockQ8_0>(),
        )
    }
}
