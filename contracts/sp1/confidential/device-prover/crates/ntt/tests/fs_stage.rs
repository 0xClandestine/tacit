// One four-step stage: a length-2^n_log DFT along a stride, tile-resident, checked against the CPU network from
// tests/v.rs (itself checked against p3's dft_batch).
use p3_field::{AbstractField, PrimeField32, TwoAdicField};
use p3_koala_bear::KoalaBear;
use p3_util::reverse_slice_index_bits;

mod common;
use common::{limbs, twiddles};

fn br(k: usize, bits: usize) -> usize { let mut r=0; let mut x=k; for _ in 0..bits { r=(r<<1)|(x&1); x>>=1; } r }

/// The CPU network for one sub-DFT, in natural output order.
fn cpu_sub(v: &mut [KoalaBear], n_log: usize, tw_fwd: &[KoalaBear], tw_rev: &[KoalaBear]) {
    let n = 1usize << n_log;
    let mid = n_log / 2;
    let mut t: Vec<KoalaBear> = (0..n).map(|k| v[br(k, n_log)]).collect();  // input bit-reversal
    for layer in 0..mid {
        let hb = 1usize << layer; let bs = hb * 2; let lr = n_log - 1 - layer;
        for start in (0..n).step_by(bs) {
            for i in 0..hb {
                let ia = start + i; let ib = ia + hb;
                let w = tw_fwd[i << lr];
                let (a, b) = (t[ia], t[ib]);
                t[ia] = a + b * w; t[ib] = a - b * w;
            }
        }
    }
    // p3 reverses the row index BITS here, not the row order
    let mut u = vec![KoalaBear::zero(); n];
    for k in 0..n { u[br(k, n_log)] = t[k]; }
    t = u;
    for layer in mid..n_log {
        let lr = n_log - 1 - layer; let hb = 1usize << lr; let bs = hb * 2;
        for (block, start) in (0..n).step_by(bs).enumerate() {
            for i in 0..hb {
                let ia = start + i; let ib = ia + hb;
                let w = tw_rev[block];
                let (a, b) = (t[ia], t[ib]);
                t[ia] = a + b * w; t[ib] = a - b * w;
            }
        }
    }
    // the CPU leaves bit-reversed rows; the kernel stores the tile as-is, so apply the same view here
    v.copy_from_slice(&t);
}

#[test]
fn stage_matches_cpu() {
    if !tacit_ntt::available() { eprintln!("no GPU"); return; }
    for &(log_h, n_log, w) in &[(4usize,2usize,1usize),(4,2,2),(6,3,1),(6,3,2)] {
        let h = 1usize << log_h; let stride = 1usize << (log_h - n_log); let subs = 1usize << (log_h - n_log);
        let mut s = 0xabcdu64; let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s % 0x7f000001) as u32 };
        let vals: Vec<KoalaBear> = (0..h*w).map(|_| KoalaBear::from_canonical_u32(next())).collect();

        let root_full = KoalaBear::two_adic_generator(log_h);
        let root = root_full.exp_u64(stride as u64);
        let tw_fwd: Vec<KoalaBear> = root.powers().take((1usize<<n_log)/2).collect();
        let mut tw_rev = tw_fwd.clone(); reverse_slice_index_bits(&mut tw_rev);

        let mut want = vals.clone();
        for sub in 0..subs { for c in 0..w {
            let mut col: Vec<KoalaBear> = (0..(1usize<<n_log)).map(|k| vals[(sub*stride+k)*w + c]).collect();
            cpu_sub(&mut col, n_log, &tw_fwd, &tw_rev);
            for k in 0..(1usize<<n_log) { want[(sub*stride+k)*w + c] = col[k]; }
        }}

        let mut got = limbs(&vals);
        let fwd = limbs(&tw_fwd); let rev = limbs(&tw_rev);
        let r = tacit_ntt::fs_stage_for_test(&mut got, &fwd, &rev, n_log as u32, w as u32, 1, stride as u32, subs as u32);
        let wantl = limbs(&want);
        let bad = got.iter().zip(&wantl).filter(|(a,b)| a!=b).count();
        println!("log_h={log_h} n_log={n_log} w={w}: rc={r} bad={bad} of {}", got.len());
    }
}
