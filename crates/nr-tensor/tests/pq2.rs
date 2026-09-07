use nr_tensor::f32_to_f16;
use nr_tensor::pq2::{self, BlockPQ2_0, PQK2_0};
use nr_tensor::q8::{self, BlockQ8_0};

#[test]
fn gguf_layout_and_bit_order() {
    // Two 34-byte blocks at an intentionally odd address. Distinct codes
    // pin little-endian packing, zeros, and the reserved +2 code.
    let mut bytes = [0u8; 69];
    bytes[1..3].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
    bytes[3..35].fill(0xe4); // 00, 01, 10, 11
    bytes[35..37].copy_from_slice(&f32_to_f16(2.0).to_le_bytes());
    bytes[37..].fill(0x55); // all zeros after dequantization
    let mut out = [0.0; 256];
    pq2::dequantize(pq2::cast_blocks(&bytes[1..]), &mut out);
    for (i, &v) in out[..128].iter().enumerate() {
        assert_eq!(v, [-0.5, 0.0, 0.5, 1.0][i % 4]);
    }
    assert_eq!(out[128..], [0.0; 128]);
}

fn weights(n: usize) -> Vec<BlockPQ2_0> {
    (0..n)
        .map(|i| BlockPQ2_0 {
            d: f32_to_f16(((i + 1) % 9) as f32 / 16.0),
            qs: core::array::from_fn(|j| (i * 37 + j * 71) as u8),
        })
        .collect()
}

fn activations(n: usize) -> Vec<BlockQ8_0> {
    (0..n)
        .map(|i| BlockQ8_0 {
            d: f32_to_f16((i % 7 + 1) as f32 / 32.0),
            // Include -128 and +127 to catch overflow in sign application.
            qs: core::array::from_fn(|j| (i * 61 + j * 17) as i8),
        })
        .collect()
}

#[test]
fn dots_match_dequantized_reference() {
    for n in [1, 2, 7, 32] {
        let w = weights(n);
        let x = activations(n * 4);
        let mut wf = vec![0.0; n * PQK2_0];
        let mut xf = wf.clone();
        pq2::dequantize(&w, &mut wf);
        q8::dequantize(&x, &mut xf);
        // Power-of-two scales make every term and sum exactly representable.
        let expect: f32 = wf.iter().zip(&xf).map(|(w, x)| w * x).sum();
        assert_eq!(pq2::dot_scalar(&w, &x), expect);
        #[cfg(target_arch = "x86_64")]
        if nr_tensor::cpu::fast_path() {
            assert_eq!(unsafe { pq2::dot_avx2(&w, &x) }, expect);
        }
        #[cfg(target_arch = "aarch64")]
        assert_eq!(unsafe { pq2::dot_neon(&w, &x) }, expect);
    }
}

#[test]
fn prefill_matches_decode_for_odd_batches_and_rows() {
    let rows = 7;
    let cols = 384;
    let w = weights(rows * cols / PQK2_0);
    for batch in [1, 3, 4, 5, 17, 64] {
        let x = activations(batch * cols / 32);
        let mut batched = vec![0.0; rows * batch];
        pq2::matmul(&mut batched, &w, &x, rows, cols, batch);
        for b in 0..batch {
            let xrow = &x[b * cols / 32..(b + 1) * cols / 32];
            let mut decoded = vec![0.0; rows];
            pq2::matvec(&mut decoded, &w, xrow, rows, cols);
            assert_eq!(&batched[b * rows..(b + 1) * rows], decoded);
            for r in 0..rows {
                assert_eq!(decoded[r], pq2::dot_scalar(&w[r * 3..(r + 1) * 3], xrow));
            }
        }
    }
}

#[test]
#[should_panic]
fn rejects_partial_weight_blocks() {
    pq2::cast_blocks(&[0; 33]);
}

#[test]
#[should_panic]
fn rejects_rows_that_split_blocks() {
    pq2::matvec(&mut [0.0; 2], &weights(1), &activations(2), 2, 64);
}
