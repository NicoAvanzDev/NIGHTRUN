use nr_tensor::f32_to_f16;
use nr_tensor::q1::{self, BlockQ1_0, QK1_0};
use nr_tensor::q8::{self, BlockQ8_0};

#[test]
fn gguf_layout_and_bit_order() {
    // Two 18-byte blocks at an intentionally odd address.
    let mut bytes = [0u8; 37];
    bytes[1..3].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
    bytes[3] = 0x81;
    bytes[18] = 0x80;
    bytes[19..21].copy_from_slice(&f32_to_f16(2.0).to_le_bytes());
    bytes[21..].fill(0xff);
    let mut out = [0.0; 256];
    q1::dequantize(q1::cast_blocks(&bytes[1..]), &mut out);
    for (i, &v) in out[..128].iter().enumerate() {
        assert_eq!(v, if matches!(i, 0 | 7 | 127) { 0.5 } else { -0.5 });
    }
    assert_eq!(out[128..], [2.0; 128]);
}

fn weights(n: usize) -> Vec<BlockQ1_0> {
    (0..n)
        .map(|i| BlockQ1_0 {
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
        let mut wf = vec![0.0; n * QK1_0];
        let mut xf = wf.clone();
        q1::dequantize(&w, &mut wf);
        q8::dequantize(&x, &mut xf);
        // Power-of-two scales make every term and sum exactly representable.
        let expect: f32 = wf.iter().zip(&xf).map(|(w, x)| w * x).sum();
        assert_eq!(q1::dot_scalar(&w, &x), expect);
        #[cfg(target_arch = "x86_64")]
        if nr_tensor::cpu::fast_path() {
            assert_eq!(unsafe { q1::dot_avx2(&w, &x) }, expect);
        }
        #[cfg(target_arch = "aarch64")]
        assert_eq!(unsafe { q1::dot_neon(&w, &x) }, expect);
    }
}

#[test]
fn prefill_matches_decode_for_odd_batches_and_rows() {
    let rows = 7;
    let cols = 384;
    let w = weights(rows * cols / QK1_0);
    for batch in [1, 3, 4, 5, 17, 64] {
        let x = activations(batch * cols / 32);
        let mut batched = vec![0.0; rows * batch];
        q1::matmul(&mut batched, &w, &x, rows, cols, batch);
        for b in 0..batch {
            let xrow = &x[b * cols / 32..(b + 1) * cols / 32];
            let mut decoded = vec![0.0; rows];
            q1::matvec(&mut decoded, &w, xrow, rows, cols);
            assert_eq!(&batched[b * rows..(b + 1) * rows], decoded);
            for r in 0..rows {
                assert_eq!(decoded[r], q1::dot_scalar(&w[r * 3..(r + 1) * 3], xrow));
            }
        }
    }
}

#[test]
#[should_panic]
fn rejects_partial_weight_blocks() {
    q1::cast_blocks(&[0; 17]);
}

#[test]
#[should_panic]
fn rejects_rows_that_split_blocks() {
    q1::matvec(&mut [0.0; 2], &weights(1), &activations(2), 2, 64);
}

#[test]
fn activations_keep_the_original_scale_for_rounding() {
    let mut src = [0.0; 32];
    src[..5].copy_from_slice(&[127.0, 0.5, 1.5, -0.5, -1.5]);
    let mut blocks = activations(1);
    q1::quantize_activations(&src, &mut blocks);
    assert_eq!(&blocks[0].qs[..5], &[127, 0, 2, 0, -2]);
    src[..5].copy_from_slice(&[1.0, -1.0, 0.5, 0.25, -0.25]);
    q1::quantize_activations(&src, &mut blocks);
    assert_eq!(&blocks[0].qs[..5], &[127, -127, 64, 32, -32]);
    assert_eq!(
        blocks[0].scale(),
        nr_tensor::f16_to_f32(f32_to_f16(1.0 / 127.0))
    );
    q1::quantize_activations(&[0.0; 32], &mut blocks);
    assert_eq!(blocks[0].scale(), 0.0);
    assert_eq!(blocks[0].qs, [0; 32]);
}
