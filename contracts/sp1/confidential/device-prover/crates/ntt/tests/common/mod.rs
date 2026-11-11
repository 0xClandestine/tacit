// Shared helpers for the differential tests: turn KoalaBear matrices into the raw limbs the kernels work on, build
// the twiddle tables the way p3-dft's Radix2DitParallel does, and run the GPU path.
#![allow(dead_code)]

use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{AbstractField, Field, TwoAdicField};
use p3_koala_bear::KoalaBear;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::reverse_slice_index_bits;

pub fn limbs(v: &[KoalaBear]) -> Vec<u32> {
    v.iter().map(|x| unsafe { std::mem::transmute::<KoalaBear, u32>(*x) }).collect()
}

pub fn fields(v: &[u32]) -> Vec<KoalaBear> {
    v.iter().map(|&x| unsafe { std::mem::transmute::<u32, KoalaBear>(x) }).collect()
}

/// The twiddle tables Radix2DitParallel builds: the powers of the 2^log_h root, and the same list bit-reversed.
pub fn twiddles(log_h: usize) -> (Vec<u32>, Vec<u32>) {
    let root = KoalaBear::two_adic_generator(log_h);
    let mut tw = Vec::new();
    let mut cur = KoalaBear::one();
    for _ in 0..(1usize << log_h) / 2 {
        tw.push(cur);
        cur *= root;
    }
    let fwd = limbs(&tw);
    reverse_slice_index_bits(&mut tw);
    (fwd, limbs(&tw))
}

/// The GPU's dft_batch, in the layout p3-dft returns (bit-reversed rows).
pub fn gpu_dft(m: &RowMajorMatrix<KoalaBear>) -> Option<RowMajorMatrix<KoalaBear>> {
    let log_h = m.height().trailing_zeros() as usize;
    let (fwd, rev) = twiddles(log_h);
    let mut v = limbs(&m.values);
    tacit_ntt::dft_in_place(&mut v, m.width(), &fwd, &rev)?;
    Some(RowMajorMatrix::new(fields(&v), m.width()))
}

/// The CPU's dft_batch, materialised row by row.
pub fn cpu_dft(m: &RowMajorMatrix<KoalaBear>) -> Vec<KoalaBear> {
    let ev = Radix2DitParallel::default().dft_batch(m.clone());
    (0..ev.height()).flat_map(|r| ev.row(r).collect::<Vec<_>>()).collect()
}

/// The GPU output as flat rows, in the same order as [`cpu_dft`]: the CPU's view applies one more row bit-reversal.
pub fn gpu_rows(m: &RowMajorMatrix<KoalaBear>) -> Vec<KoalaBear> {
    p3_matrix::bitrev::BitReversableMatrix::bit_reverse_rows(m.clone()).to_row_major_matrix().values
}

/// The GPU's LDE (shift = 1), in the layout p3's coset_lde_batch returns.
pub fn gpu_lde(m: &RowMajorMatrix<KoalaBear>, added_bits: usize) -> Option<RowMajorMatrix<KoalaBear>> {
    let in_h = m.height();
    let w = m.width();
    let in_log = in_h.trailing_zeros() as usize;
    let log_h = (in_h << added_bits).trailing_zeros() as usize;

    let root_inv = KoalaBear::two_adic_generator(in_log).inverse();
    let mut twi = Vec::new();
    let mut cur = KoalaBear::one();
    for _ in 0..in_h / 2 {
        twi.push(cur);
        cur *= root_inv;
    }
    let tw_inv = limbs(&twi);
    reverse_slice_index_bits(&mut twi);
    let tw_inv_rev = limbs(&twi);
    let h_inv = KoalaBear::from_canonical_usize(in_h).inverse();
    let dfted = tacit_ntt::idft_scaled_u32(&limbs(&m.values), w, &tw_inv, &tw_inv_rev, limbs(&[h_inv])[0])?;

    let mut padded = vec![0u32; (in_h << added_bits) * w];
    for i in 0..in_h {
        let dst = i << added_bits;
        padded[dst * w..(dst + 1) * w].copy_from_slice(&dfted[i * w..(i + 1) * w]);
    }
    let (fwd, rev) = twiddles(log_h);
    let out = tacit_ntt::dft_u32(&padded, w, &fwd, &rev)?;
    Some(RowMajorMatrix::new(fields(&out), w))
}
