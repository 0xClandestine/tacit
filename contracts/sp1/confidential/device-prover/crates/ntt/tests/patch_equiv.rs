// The patch's dft_batch, GPU on, against the real CPU implementation with the GPU off, over the whole output — the
// same comparison `tacit-prover check` makes. Every row and column, not a sample.
use p3_dft::{NaiveDft, Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::AbstractField;
use p3_koala_bear::KoalaBear;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

fn mat(h: usize, w: usize, seed: u64) -> RowMajorMatrix<KoalaBear> {
    let mut s = seed | 1;
    let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s % 0x7f000001) as u32 };
    RowMajorMatrix::new((0..h * w).map(|_| KoalaBear::from_canonical_u32(next())).collect(), w)
}

fn flat(m: p3_matrix::bitrev::BitReversedMatrixView<RowMajorMatrix<KoalaBear>>) -> Vec<KoalaBear> {
    m.to_row_major_matrix().values
}

#[test]
fn patch_matches_cpu() {
    if tacit_ntt::device().is_none() { println!("no GPU"); return; }
    // NaiveDft is quadratic, so these heights stay small; the widths are the whole point (the uint4 reversal used to
    // drop the columns past the last multiple of four), and every width 1..=33 is covered at 2^10.
    let mut shapes: Vec<(usize, usize)> = (1..=33).map(|w| (10usize, w)).collect();
    shapes.extend([(12, 132), (12, 192), (13, 21), (13, 5)]);
    for &(log_h, w) in &shapes {
        let m = mat(1 << log_h, w, 0xabc ^ log_h as u64 ^ ((w as u64) << 9));
        // Radix2DitParallel is the patched crate, so its own CPU path is what the GPU replaces; compare against
        // NaiveDft instead, which shares no code with either and cannot hide a mistake in the patch itself.
        let gpu = flat(Radix2DitParallel::default().dft_batch(m.clone()));
        let cpu = NaiveDft.dft_batch(m.clone()).values;
        let bad = gpu.iter().zip(&cpu).filter(|(a, b)| a != b).count();
        println!("dft {log_h}x{w}: bad={bad}");
        assert_eq!(bad, 0, "dft {log_h}x{w} differs from the CPU at {bad} cells");
    }
}
