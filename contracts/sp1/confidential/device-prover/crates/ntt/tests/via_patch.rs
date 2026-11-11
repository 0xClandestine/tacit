use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{AbstractField, PrimeField32};
use p3_koala_bear::KoalaBear;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use std::io::Write;
use std::fs::File;

#[test]
fn dump() {
    let (log_h, w) = (10usize, 32usize);
    let h = 1usize << log_h;
    let vals: Vec<KoalaBear> = (0..h * w).map(|i| KoalaBear::from_canonical_u32((i as u32 * 7 + 3) % 0x7f000001)).collect();
    let m = RowMajorMatrix::new(vals, w);
    let ev = Radix2DitParallel::default().dft_batch(m);
    let out: Vec<u32> = (0..ev.height()).flat_map(|r| ev.row(r).map(|x| x.as_canonical_u32()).collect::<Vec<_>>()).collect();
    let path = std::env::var("DUMP").unwrap_or_else(|_| "/tmp/gpu.bin".into());
    let mut f = File::create(path).unwrap();
    for x in &out { f.write_all(&x.to_le_bytes()).unwrap(); }
    println!("wrote {} values", out.len());
}
