use alloc::vec::Vec;

use p3_field::{Field, Powers, TwoAdicField};
use p3_matrix::bitrev::{BitReversableMatrix, BitReversedMatrixView};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_matrix::util::reverse_matrix_index_bits;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_bits, reverse_slice_index_bits};
use tracing::instrument;

use crate::butterflies::{Butterfly, DitButterfly};

/// The GPU path, only for KoalaBear. `tacit_ntt` decides at run time whether this machine has an Apple-silicon GPU and
/// whether the shape suits its kernels, so this is a pure fast path over the code below.
///
/// The tables are built here, with the same field type and the same order as the CPU path (powers of the 2^log_h root,
/// then the same list bit-reversed), and handed over as raw limbs; the kernel does KoalaBear arithmetic itself.
fn gpu_dft<F: TwoAdicField + 'static>(
    mat: &RowMajorMatrix<F>,
) -> Option<BitReversedMatrixView<RowMajorMatrix<F>>> {
    if !is_koala_bear::<F>() {
        return None;
    }
    let h = mat.height();
    let log_h = log2_strict_usize(h);
    let root = F::two_adic_generator(log_h);
    let mut tw: Vec<F> = root.powers().take(h / 2).collect();
    let tw_fwd = limbs(&tw);
    reverse_slice_index_bits(&mut tw);
    let tw_rev = limbs(&tw);
    let mut values = limbs(&mat.values);
    // The four-step variant does the same transform in three passes over the matrix instead of log_h, which is most of
    // the reason a DFT this size is memory-bound. It declines shapes its tile cannot hold, and the plain path below
    // covers those. It emits the materialised row order, so it skips the bit_reverse_rows view the plain path needs.
    // The four-step variant (tacit_ntt::dft_fourstep_in_place) is not wired in: it matches this kernel for some
    // shapes but not the ones the prover uses (log_h 18, width 5 is the first that differs), so the plain network stays
    // the only path. crates/ntt's fs_vs_plain test is the gate it has to pass first.
    tacit_ntt::dft_in_place(&mut values, mat.width(), &tw_fwd, &tw_rev)?;
    // `dft_in_place` leaves the rows in the physical order the CPU matrix has before its trailing `bit_reverse_rows`
    // view, so the view is built here exactly as the CPU path does.
    Some(RowMajorMatrix::new(to_fields::<F>(values), mat.width()).bit_reverse_rows())
}

/// The LDE path: only KoalaBear, and only shift = 1, which is the plain LDE the prover uses. Other shifts keep the
/// CPU's coset weights.
fn gpu_lde<F: TwoAdicField + 'static>(
    mat: &RowMajorMatrix<F>,
    added_bits: usize,
    shift: F,
) -> Option<BitReversedMatrixView<RowMajorMatrix<F>>> {
    if shift != F::one() || !is_koala_bear::<F>() {
        return None;
    }
    let h = mat.height();
    let w = mat.width();
    let h_out = h << added_bits;
    let in_log = log2_strict_usize(h);
    let log_h = log2_strict_usize(h_out);

    // 1. the inverse DFT on the input's height, with inverse-root twiddles scaled by h^-1.
    let root_inv = F::two_adic_generator(in_log).inverse();
    let mut twi: Vec<F> = root_inv.powers().take(h / 2).collect();
    let tw_inv = limbs(&twi);
    reverse_slice_index_bits(&mut twi);
    let tw_inv_rev = limbs(&twi);
    let h_inv = F::from_canonical_usize(h).inverse();
    let h_inv_limb = limbs(core::slice::from_ref(&h_inv))[0];
    let dfted = tacit_ntt::idft_scaled_u32(&limbs(&mat.values), w, &tw_inv, &tw_inv_rev, h_inv_limb)?;

    // 2. zero-pad in bit-reversed row order: entry i of the result becomes row i << added_bits.
    let mut padded = alloc::vec![0u32; h_out * w];
    for i in 0..h {
        let dst = i << added_bits;
        padded[dst * w..(dst + 1) * w].copy_from_slice(&dfted[i * w..(i + 1) * w]);
    }

    // 3. the forward DFT on the padded height.
    let root = F::two_adic_generator(log_h);
    let mut tw: Vec<F> = root.powers().take(h_out / 2).collect();
    let tw_fwd = limbs(&tw);
    reverse_slice_index_bits(&mut tw);
    let tw_rev = limbs(&tw);
    let out = tacit_ntt::dft_u32(&padded, w, &tw_fwd, &tw_rev)?;
    Some(RowMajorMatrix::new(to_fields::<F>(out), w).bit_reverse_rows())
}

/// Whether `F` is KoalaBear, which the kernels' field arithmetic is written for: one u32 limb each, characteristic
/// 2^31 - 2^24 + 1. Checked from the field itself, so no dependency on p3-koala-bear (which depends on this crate).
fn is_koala_bear<F: Field>() -> bool {
    core::mem::size_of::<F>() == 4 && F::order() == 0x7f00_0001u64.into()
}

/// The raw limbs of a slice of field elements. KoalaBear and the other 32-bit fields store their Montgomery form in
/// one u32, which is what the kernel expects.
fn limbs<F: Field>(v: &[F]) -> Vec<u32> {
    // Only reached for KoalaBear, whose limbs are one u32; other fields never get here.
    debug_assert_eq!(core::mem::size_of::<F>(), 4);
    let mut out = Vec::with_capacity(v.len());
    for x in v {
        let mut limb = 0u32;
        // SAFETY: KoalaBear is one u32 and plain old data.
        unsafe { core::ptr::copy_nonoverlapping(x as *const F as *const u32, &mut limb as *mut u32, 1) };
        out.push(limb);
    }
    out
}

/// The inverse of [`limbs`]: raw limbs back into field elements.
fn to_fields<F: Field>(v: Vec<u32>) -> Vec<F> {
    let mut out = Vec::with_capacity(v.len());
    for limb in v {
        let mut x = core::mem::MaybeUninit::<F>::uninit();
        unsafe {
            core::ptr::copy_nonoverlapping(&limb as *const u32, x.as_mut_ptr() as *mut u32, 1);
            out.push(x.assume_init());
        }
    }
    out
}

use crate::TwoAdicSubgroupDft;

/// A parallel FFT algorithm which divides a butterfly network's layers into two halves.
///
/// For the first half, we apply a butterfly network with smaller blocks in earlier layers,
/// i.e. either DIT or Bowers G. Then we bit-reverse, and for the second half, we continue executing
/// the same network but in bit-reversed order. This way we're always working with small blocks,
/// so within each half, we can have a certain amount of parallelism with no cross-thread
/// communication.
#[derive(Default, Clone, Debug)]
pub struct Radix2DitParallel;

/// The textbook path, kept as a separate function so the GPU path can be checked against it in place.
fn dft_cpu<F: TwoAdicField>(mat: &RowMajorMatrix<F>) -> BitReversedMatrixView<RowMajorMatrix<F>> {
    let mut mat = mat.clone();
    let h = mat.height();
    let log_h = log2_strict_usize(h);
    let root = F::two_adic_generator(log_h);
    let mut twiddles: Vec<F> = root.powers().take(h / 2).collect();
    let mid = log_h / 2;
    reverse_matrix_index_bits(&mut mat);
    par_dit_layer(&mut mat, mid, &twiddles);
    reverse_matrix_index_bits(&mut mat);
    reverse_slice_index_bits(&mut twiddles);
    par_dit_layer_rev(&mut mat, mid, &twiddles);
    mat.bit_reverse_rows()
}

impl<F: TwoAdicField> TwoAdicSubgroupDft<F> for Radix2DitParallel {
    type Evaluations = BitReversedMatrixView<RowMajorMatrix<F>>;

    fn dft_batch(&self, mat: RowMajorMatrix<F>) -> Self::Evaluations {
        // On an Apple-silicon GPU the whole network runs as a few dispatches; the result is bit-for-bit what the loop
        // below computes. `try_gpu` returns None on every other machine, and for shapes the kernels do not take, so
        // the CPU path is always there to fall back on.
        if let Some(out) = gpu_dft(&mat) {
            return out;
        }
        dft_cpu(&mat)
    }

    #[instrument(skip_all, fields(dims = %mat.dimensions(), added_bits))]
    fn coset_lde_batch(
        &self,
        mat: RowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> Self::Evaluations {
        if let Some(out) = gpu_lde(&mat, added_bits, shift) {
            return out;
        }
        let mut mat = mat;
        let h = mat.height();
        let log_h = log2_strict_usize(h);
        let mid = log_h / 2;
        let h_inv = F::from_canonical_usize(h).inverse();

        let root = F::two_adic_generator(log_h);
        let root_inv = root.inverse();

        let mut twiddles_inv: Vec<F> = root_inv.powers().take(h / 2).collect();

        // The first half looks like a normal DIT.
        reverse_matrix_index_bits(&mut mat);
        par_dit_layer(&mut mat, mid, &twiddles_inv);

        // For the second half, we flip the DIT, working in bit-reversed order.
        reverse_matrix_index_bits(&mut mat);
        reverse_slice_index_bits(&mut twiddles_inv);
        par_dit_layer_rev(&mut mat, mid, &twiddles_inv);
        // We skip the final bit-reversal, since the next FFT expects bit-reversed input.

        // Rescale coefficients in two ways:
        // - divide by height (since we're doing an inverse DFT)
        // - multiply by powers of the coset shift (see default coset LDE impl for an explanation)
        let weights = Powers {
            base: shift,
            current: h_inv,
        }
        .take(h);
        for (row, weight) in weights.enumerate() {
            // reverse_bits because mat is encoded in bit-reversed order
            mat.scale_row(reverse_bits(row, h), weight);
        }

        mat = mat.bit_reversed_zero_pad(added_bits);

        let h = mat.height();
        let log_h = log2_strict_usize(h);
        let mid = log_h / 2;

        let root = F::two_adic_generator(log_h);

        let mut twiddles: Vec<F> = root.powers().take(h / 2).collect();

        // The first half looks like a normal DIT.
        par_dit_layer(&mut mat, mid, &twiddles);

        // For the second half, we flip the DIT, working in bit-reversed order.
        reverse_matrix_index_bits(&mut mat);
        reverse_slice_index_bits(&mut twiddles);
        par_dit_layer_rev(&mut mat, mid, &twiddles);

        mat.bit_reverse_rows()
    }
}

/// This can be used as the first half of a parallelized butterfly network.
fn par_dit_layer<F: Field>(mat: &mut RowMajorMatrix<F>, mid: usize, twiddles: &[F]) {
    let log_h = log2_strict_usize(mat.height());

    // max block size: 2^mid
    mat.par_row_chunks_exact_mut(1 << mid)
        .for_each(|mut submat| {
            for layer in 0..mid {
                dit_layer(&mut submat, log_h, layer, twiddles);
            }
        });
}

/// This can be used as the second half of a parallelized butterfly network.
fn par_dit_layer_rev<F: Field>(mat: &mut RowMajorMatrix<F>, mid: usize, twiddles_rev: &[F]) {
    let log_h = log2_strict_usize(mat.height());

    // max block size: 2^(log_h - mid)
    mat.par_row_chunks_exact_mut(1 << (log_h - mid))
        .enumerate()
        .for_each(|(thread, mut submat)| {
            for layer in mid..log_h {
                let first_block = thread << (layer - mid);
                dit_layer_rev(&mut submat, log_h, layer, &twiddles_rev[first_block..]);
            }
        });
}

/// One layer of a DIT butterfly network.
fn dit_layer<F: Field>(
    submat: &mut RowMajorMatrixViewMut<'_, F>,
    log_h: usize,
    layer: usize,
    twiddles: &[F],
) {
    let layer_rev = log_h - 1 - layer;

    let half_block_size = 1 << layer;
    let block_size = half_block_size * 2;
    debug_assert!(submat.height() >= block_size);

    for block_start in (0..submat.height()).step_by(block_size) {
        for i in 0..half_block_size {
            let hi = block_start + i;
            let lo = hi + half_block_size;
            let twiddle = twiddles[i << layer_rev];

            let (hi_chunk, lo_chunk) = submat.row_pair_mut(hi, lo);
            DitButterfly(twiddle).apply_to_rows(hi_chunk, lo_chunk);
        }
    }
}

/// Like `dit_layer`, except the matrix and twiddles are encoded in bit-reversed order.
/// This can also be viewed as a layer of the Bowers G^T network.
fn dit_layer_rev<F: Field>(
    submat: &mut RowMajorMatrixViewMut<'_, F>,
    log_h: usize,
    layer: usize,
    twiddles_rev: &[F],
) {
    let layer_rev = log_h - 1 - layer;

    let half_block_size = 1 << layer_rev;
    let block_size = half_block_size * 2;
    debug_assert!(submat.height() >= block_size);

    for (block, block_start) in (0..submat.height()).step_by(block_size).enumerate() {
        let twiddle = twiddles_rev[block];
        for i in 0..half_block_size {
            let hi = block_start + i;
            let lo = hi + half_block_size;
            let (hi_chunk, lo_chunk) = submat.row_pair_mut(hi, lo);
            DitButterfly(twiddle).apply_to_rows(hi_chunk, lo_chunk);
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_goldilocks::Goldilocks;

    use crate::testing::*;
    use crate::Radix2DitParallel;

    #[test]
    fn dft_matches_naive() {
        test_dft_matches_naive::<BabyBear, Radix2DitParallel>();
    }

    #[test]
    fn coset_dft_matches_naive() {
        test_coset_dft_matches_naive::<BabyBear, Radix2DitParallel>();
    }

    #[test]
    fn idft_matches_naive() {
        test_idft_matches_naive::<Goldilocks, Radix2DitParallel>();
    }

    #[test]
    fn coset_idft_matches_naive() {
        test_coset_idft_matches_naive::<BabyBear, Radix2DitParallel>();
        test_coset_idft_matches_naive::<Goldilocks, Radix2DitParallel>();
    }

    #[test]
    fn lde_matches_naive() {
        test_lde_matches_naive::<BabyBear, Radix2DitParallel>();
    }

    #[test]
    fn coset_lde_matches_naive() {
        test_coset_lde_matches_naive::<BabyBear, Radix2DitParallel>();
    }

    #[test]
    fn dft_idft_consistency() {
        test_dft_idft_consistency::<BabyBear, Radix2DitParallel>();
    }
}
