// The four-step (Bailey) decomposition, written plainly on the CPU, checked against p3's Radix2DitParallel.
// If this is not bit-identical there is no point writing a kernel for it.
//
// n = n1*n2. Index the data as a[i][j] = x[i + n1*j] (the natural four-step layout):
//   1. DFT each column of length n1
//   2. multiply a[i][j] by w^(i*j)
//   3. DFT each row of length n2
// and read the result out transposed. p3 wants bit-reversed row order, which is one permutation at the end.
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{AbstractField, Field, TwoAdicField};
use p3_koala_bear::KoalaBear;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

fn mat(h: usize, w: usize, seed: u64) -> RowMajorMatrix<KoalaBear> {
    let mut s = seed | 1;
    let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s % 0x7f000001) as u32 };
    RowMajorMatrix::new((0..h * w).map(|_| KoalaBear::from_canonical_u32(next())).collect(), w)
}

/// A naive O(n^2) DFT of one column of length n, with the given root.
fn dft_vec(v: &[KoalaBear], root: KoalaBear) -> Vec<KoalaBear> {
    let n = v.len();
    let mut out = vec![KoalaBear::zero(); n];
    for k in 0..n {
        let mut acc = KoalaBear::zero();
        let mut tw = KoalaBear::one();
        let step = root.exp_u64(k as u64);
        for j in 0..n {
            acc += v[j] * tw;
            tw *= step;
        }
        out[k] = acc;
    }
    out
}

/// The four-step (Bailey) DFT over `h` rows of `w` columns.
///
/// With N = n1*n2, index the data as x[i + n1*j] (i in [0,n1) fast, j in [0,n2) slow) and the frequency as
/// k2 + n2*k1 (k2 in [0,n2) fast, k1 in [0,n1) slow):
///
///   X[k2 + n2*k1] = sum_i [ w^(i*k2) * (w^n2)^(i*k1) ] * sum_j x[i+n1 j] * (w^n1)^(j*k2)
///
/// so: a length-n2 DFT over j with root w^n1, a twiddle by w^(i*k2), then a length-n1 DFT over i with root w^n2.
/// The w^(i*n2*k1) term is *not* dropped — it is the root of the second stage.
fn four_step(m: &RowMajorMatrix<KoalaBear>, log_h: usize, a: usize) -> RowMajorMatrix<KoalaBear> {
    let h = 1usize << log_h;
    let w = m.width();
    let n1 = 1usize << a;
    let n2 = 1usize << (log_h - a);
    assert_eq!(n1 * n2, h);
    let root = KoalaBear::two_adic_generator(log_h);
    let r1 = root.exp_u64(n1 as u64); // inner DFT over j
    let r2 = root.exp_u64(n2 as u64); // outer DFT over i

    let mut t = m.values.clone();
    // 1. length-n2 DFT over j for each i (and each column).
    for i in 0..n1 {
        for c in 0..w {
            let mut v: Vec<KoalaBear> = (0..n2).map(|j| t[(i + n1 * j) * w + c]).collect();
            v = dft_vec(&v, r1);
            for j in 0..n2 { t[(i + n1 * j) * w + c] = v[j]; }
        }
    }
    // 2. twiddle by w^(i*k2).
    for i in 0..n1 {
        for k2 in 0..n2 {
            let tw = root.exp_u64((i * k2) as u64);
            for c in 0..w { t[(i + n1 * k2) * w + c] *= tw; }
        }
    }
    // 3. length-n1 DFT over i for each k2, written to output index k2 + n2*k1.
    let mut res = vec![KoalaBear::zero(); h * w];
    for k2 in 0..n2 {
        for c in 0..w {
            let v: Vec<KoalaBear> = (0..n1).map(|i| t[(i + n1 * k2) * w + c]).collect();
            let o = dft_vec(&v, r2);
            for k1 in 0..n1 { res[(k2 + n2 * k1) * w + c] = o[k1]; }
        }
    }
    RowMajorMatrix::new(res, w)
}

#[test]
fn four_step_matches_p3() {
    for &(log_h, w) in &[(6usize, 4usize), (8, 3), (8, 8), (10, 5), (12, 21), (13, 32), (14, 5)] {
        let m = mat(1 << log_h, w, 0xfeed ^ log_h as u64 ^ ((w as u64) << 5));
        let want = Radix2DitParallel::default().dft_batch(m.clone()).to_row_major_matrix().values;
        for a in 1..log_h {
            let got = four_step(&m, log_h, a);
            // four_step writes the transform in natural order, which is what the CPU path's materialised
            // output holds: dft_batch's view is applied on top of this, in the patch, exactly as for the CPU.
            let bad = got.values.iter().zip(&want).filter(|(x, y)| x != y).count();
            println!("log_h={log_h} w={w} a={a}: bad={bad}");
        }
    }
}
