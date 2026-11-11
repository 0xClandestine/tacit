use p3_field::{AbstractField, Field, PrimeField32, TwoAdicField};
use p3_koala_bear::KoalaBear;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_matrix::Matrix;
use p3_matrix::bitrev::BitReversableMatrix;
use p3_util::reverse_slice_index_bits;
use p3_matrix::util::reverse_matrix_index_bits;

fn dit_layer<F: Field>(m: &mut RowMajorMatrixViewMut<'_, F>, log_h: usize, layer: usize, tw: &[F], wd: usize) {
    let lr = log_h - 1 - layer;
    let hb = 1 << layer; let bs = hb * 2;
    let hgt = m.height();
    for start in (0..hgt).step_by(bs) {
        for i in 0..hb {
            let hi = start + i; let lo = hi + hb;
            let t = tw[i << lr];
            let (hc, lc) = m.row_pair_mut(hi, lo);
            for c in 0..wd {
                let a = hc[c]; let b = lc[c];
                hc[c] = a + b * t; lc[c] = a - b * t;
            }
        }
    }
}

/// dit_layer_rev: block pairing on the bit-reversed index, twiddle per block.
fn dit_layer_rev<F: Field>(m: &mut RowMajorMatrixViewMut<'_, F>, log_h: usize, layer: usize, tw_rev: &[F], wd: usize, rows: usize, chunk: usize, nchunk: usize) {
    let lr = log_h - 1 - layer;
    let hb = 1 << lr; let bs = hb * 2;
    // rows of chunk `chunk` are [chunk*rows, (chunk+1)*rows); `block` restarts at 0 within the chunk
    let lo_row = chunk * rows;
    for (block, start) in (0..rows).step_by(bs).enumerate() {
        let start = lo_row + start;
        let t = tw_rev[block];
        for i in 0..hb {
            let hi = start + i; let lo = hi + hb;
            let (hc, lc) = m.row_pair_mut(hi, lo);
            for c in 0..wd {
                let a = hc[c]; let b = lc[c];
                hc[c] = a + b * t; lc[c] = a - b * t;
            }
        }
    }
}

#[test]
fn replicate() {
  for &(log_h, w) in &[(4usize,1usize),(5,3),(6,4),(8,3),(10,5),(12,21),(13,32)] { rep(log_h,w); }
}

fn rep(log_h: usize, w: usize) {
    let h = 1usize << log_h;
    let root = KoalaBear::two_adic_generator(log_h);
    let mut tw: Vec<KoalaBear> = root.powers().take(h/2).collect();
    let m0 = RowMajorMatrix::new((0..h*w).map(|i| KoalaBear::from_canonical_u32((i as u32 * 7 + 1) % 0x7f000001)).collect(), w);
    let mid = log_h/2;
    let mut m = m0.clone();
    reverse_matrix_index_bits(&mut m);
    for layer in 0..mid { dit_layer(&mut m.as_view_mut(), log_h, layer, &tw, w); }
    reverse_matrix_index_bits(&mut m);   // p3 reverses the rows again between the halves
    reverse_slice_index_bits(&mut tw);
    // second half: for each chunk of 2^(log_h-mid) rows, run layers mid..log_h, block-twiddled
    let cs = 1usize << (log_h - mid);
    let nchunk = h / cs;
    for thread in 0..nchunk {
        for layer in mid..log_h {
            let fb = thread << (layer - mid);
            dit_layer_rev(&mut m.as_view_mut(), log_h, layer, &tw[fb.min(tw.len())..], w, cs, thread, nchunk);
        }
    }
    // p3's own answer, materialised, including its trailing bit_reverse_rows
    use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
    let want = Radix2DitParallel::default().dft_batch(m0.clone()).to_row_major_matrix().values;
    let got = m.bit_reverse_rows().to_row_major_matrix().values;
    let bad = got.iter().zip(&want).filter(|(a,b)| a!=b).count();
    println!("log_h={log_h} w={w}: bad={bad} of {}", got.len());
    assert_eq!(bad, 0, "log_h={log_h} w={w}");
}
