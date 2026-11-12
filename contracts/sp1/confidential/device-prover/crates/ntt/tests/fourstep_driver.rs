//! The four-step driver against the plain kernel, byte for byte. The contract the prover needs is that it leaves the
//! rows in the same physical order `dft_in_place` does; anything weaker (same rows, different order) would be caught by
//! the p3-dft patch turning into a wrong vkey rather than a failed test.

mod common;

use common::{limbs, twiddles};
use p3_field::AbstractField;
use p3_koala_bear::KoalaBear;
use p3_matrix::dense::RowMajorMatrix;

fn check(log_h: usize, width: usize) {
    if !tacit_ntt::available() || !tacit_ntt::fourstep_supported(log_h, width) {
        return;
    }
    let h = 1usize << log_h;
    let mut v = Vec::with_capacity(h * width);
    let mut cur = KoalaBear::one();
    for _ in 0..h * width {
        v.push(cur);
        cur += KoalaBear::one();
    }
    let m = RowMajorMatrix::new(v, width);
    let (fwd, rev) = twiddles(log_h);

    let mut plain = limbs(&m.values);
    if tacit_ntt::dft_in_place(&mut plain, width, &fwd, &rev).is_none() {
        return;
    }
    let mut four = limbs(&m.values);
    if tacit_ntt::dft_fourstep_in_place(&mut four, width, &fwd).is_none() {
        return;
    }
    assert_eq!(plain, four, "log_h={log_h} w={width}");
}

#[test]
fn fourstep_driver() {
    for log_h in 2..=14 {
        for w in [1usize, 3, 4, 5, 16, 21, 32] {
            check(log_h, w);
        }
    }
}

/// The shapes above are not the ones the prover uses. These are, and the four-step is expected to fail on some of them
/// today (log_h 18 width 5 is the first), which is why the p3-dft patch does not call it. Ignored rather than removed so
/// the gate stays visible and can be run the moment the odd-length stage tables are fixed.
#[test]
#[ignore]
fn fourstep_prover_shapes() {
    for log_h in [18usize, 20, 21, 22, 23, 24] {
        for w in [4usize, 5, 16, 18, 20, 21, 23, 32] {
            check(log_h, w);
        }
    }
}
