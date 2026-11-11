use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::AbstractField;
use p3_koala_bear::KoalaBear;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
mod common;
use common::gpu_dft;

fn rand_matrix(h: usize, w: usize, seed: u64) -> RowMajorMatrix<KoalaBear> {
    let mut s = seed | 1;
    let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s % 0x7f000001) as u32 };
    RowMajorMatrix::new((0..h * w).map(|_| KoalaBear::from_canonical_u32(next())).collect(), w)
}

#[test]
fn sizes() {
    if tacit_ntt::device().is_none() { return; }
    // The widths and heights the prover actually asks for, plus the small ones.
    for &(log_h, w) in &[
        (10usize, 32usize), (12, 64), (14, 96),
        (16, 4), (16, 5), (18, 4), (18, 5),
        (20, 4), (20, 5), (21, 132), (22, 32), (23, 4), (23, 5), (23, 21),
    ] {
        let h = 1usize << log_h;
        let m = rand_matrix(h, w, 0x1234 ^ (log_h as u64) ^ ((w as u64) << 8));
        let ev = Radix2DitParallel::default().dft_batch(m.clone());
        let want: Vec<KoalaBear> = (0..ev.height()).flat_map(|r| ev.row(r).collect::<Vec<_>>()).collect();
        let Some(got) = gpu_dft(&m) else { println!("{h}x{w}: DECLINED"); continue };
        let bad = common::gpu_rows(&got).iter().zip(&want).filter(|(a, b)| a != b).count();
        println!("{h}x{w}: bad={bad}");
    }
}
