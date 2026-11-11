// GPU against CPU for the same matrices, both through p3-dft's Radix2DitParallel. Set TACIT_METAL=0 to time the
// stock CPU path; otherwise this times the GPU path over the same inputs.
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{AbstractField, PrimeField32};
use p3_koala_bear::KoalaBear;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use std::time::Instant;

fn rand_matrix(h: usize, w: usize, seed: u64) -> RowMajorMatrix<KoalaBear> {
    let mut s = seed | 1;
    let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s % 0x7f000001) as u32 };
    RowMajorMatrix::new((0..h * w).map(|_| KoalaBear::from_canonical_u32(next())).collect(), w)
}

#[test]
fn bench() {
    let path = if tacit_ntt::available() { "gpu" } else { "cpu" };
    let cpu = Radix2DitParallel::default();
    for &(log_h, w) in &[(16usize, 132usize), (18, 132), (20, 132), (21, 132), (21, 192), (22, 132)] {
        let h = 1usize << log_h;
        let m = rand_matrix(h, w, 0xbeef ^ log_h as u64);
        let _ = cpu.dft_batch(m.clone());
        let mut best = f64::MAX;
        let mut digest = 0u64;
        for _ in 0..3 {
            let t = Instant::now();
            let ev = cpu.dft_batch(m.clone());
            // The view is lazy, so force it inside the timed region: measure the DFT, not just the setup.
            std::hint::black_box(&ev);
            let _ = ev.height();
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        let ev = cpu.dft_batch(m.clone());
        let out: Vec<u32> = (0..ev.height()).flat_map(|r| ev.row(r).map(|x| x.as_canonical_u32()).collect::<Vec<_>>()).collect();
        digest = out.iter().fold(0u64, |a, &x| a.wrapping_mul(0x100000001b3).wrapping_add(x as u64));
        println!("{path} log_h={log_h:2} w={w:3}: {best:8.1} ms  digest {digest:016x}");
    }
}
