//! aarch64 SIMD backend. R1 state: scalar delegation stubs so the crate
//! compiles for aarch64 (cpu::fast_path() is false, so these are never
//! actually called yet); NEON implementations land next (R3).

pub mod kernels {
    use crate::q8::BlockQ8_0;

    /// # Safety
    /// No special requirements yet (scalar delegation).
    pub unsafe fn dot_q8(w: &[BlockQ8_0], x: &[BlockQ8_0]) -> f32 {
        crate::kernels::dot_q8_scalar(w, x)
    }

    /// # Safety
    /// No special requirements yet (scalar delegation).
    pub unsafe fn dot_q8_x4(w: &[BlockQ8_0], xs: [&[BlockQ8_0]; 4]) -> [f32; 4] {
        [dot_q8(w, xs[0]), dot_q8(w, xs[1]), dot_q8(w, xs[2]), dot_q8(w, xs[3])]
    }

    /// # Safety
    /// No special requirements yet (scalar delegation).
    pub unsafe fn dot_f16(k: &[u16], q: &[f32]) -> f32 {
        let mut s = 0f32;
        for (&kb, &qv) in k.iter().zip(q) {
            s += crate::f16::f16_to_f32(kb) * qv;
        }
        s
    }

    /// # Safety
    /// No special requirements yet (scalar delegation).
    pub unsafe fn axpy_f16(out: &mut [f32], a: f32, v: &[u16]) {
        for (o, &vb) in out.iter_mut().zip(v) {
            *o += a * crate::f16::f16_to_f32(vb);
        }
    }
}

pub mod kquant {
    use crate::kquant::{BlockQ4K, BlockQ6K, BlockQ8K};

    /// # Safety
    /// No special requirements yet (scalar delegation).
    pub unsafe fn dot_q4k(w: &[BlockQ4K], x: &[BlockQ8K]) -> f32 {
        crate::kquant::dot_q4k_scalar(w, x)
    }

    /// # Safety
    /// No special requirements yet (scalar delegation).
    pub unsafe fn dot_q4k_x4(w: &[BlockQ4K], xs: [&[BlockQ8K]; 4]) -> [f32; 4] {
        [dot_q4k(w, xs[0]), dot_q4k(w, xs[1]), dot_q4k(w, xs[2]), dot_q4k(w, xs[3])]
    }

    /// # Safety
    /// No special requirements yet (scalar delegation).
    pub unsafe fn dot_q6k(w: &[BlockQ6K], x: &[BlockQ8K]) -> f32 {
        crate::kquant::dot_q6k_scalar(w, x)
    }

    /// # Safety
    /// No special requirements yet (scalar delegation).
    pub unsafe fn dot_q6k_x4(w: &[BlockQ6K], xs: [&[BlockQ8K]; 4]) -> [f32; 4] {
        [dot_q6k(w, xs[0]), dot_q6k(w, xs[1]), dot_q6k(w, xs[2]), dot_q6k(w, xs[3])]
    }
}
