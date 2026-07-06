//! f32 vector operations: rmsnorm, softmax, silu, dot, saxpy, argmax.

/// out = x * rsqrt(mean(x^2) + eps) * weight
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let mut ss = 0f32;
    for &v in x {
        ss += v * v;
    }
    let scale = 1.0 / libm::sqrtf(ss / n as f32 + eps);
    for i in 0..n {
        out[i] = x[i] * scale * weight[i];
    }
}

/// In-place RMSNorm: x = x * rsqrt(mean(x^2) + eps) * weight.
pub fn rmsnorm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let mut ss = 0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let scale = 1.0 / libm::sqrtf(ss / n as f32 + eps);
    for (v, &w) in x.iter_mut().zip(weight) {
        *v *= scale * w;
    }
}

/// In-place numerically-stable softmax.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = libm::expf(*v - max);
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// SwiGLU gate: gate[i] = silu(gate[i]) * up[i]
pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    for (g, &u) in gate.iter_mut().zip(up) {
        let x = *g;
        let silu = x / (1.0 + libm::expf(-x));
        *g = silu * u;
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for (x, y) in a.iter().zip(b) {
        s += x * y;
    }
    s
}

/// x *= s
pub fn scale_inplace(x: &mut [f32], s: f32) {
    for v in x.iter_mut() {
        *v *= s;
    }
}

/// y += a * x
pub fn saxpy(y: &mut [f32], a: f32, x: &[f32]) {
    for (yv, &xv) in y.iter_mut().zip(x) {
        *yv += a * xv;
    }
}

pub fn add_assign(y: &mut [f32], x: &[f32]) {
    for (yv, &xv) in y.iter_mut().zip(x) {
        *yv += xv;
    }
}

pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}
