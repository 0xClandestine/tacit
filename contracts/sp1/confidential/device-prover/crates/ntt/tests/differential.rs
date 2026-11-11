// The GPU NTT against p3-dft's own `Radix2DitParallel`, cell for cell, on random matrices of several shapes and on
// edge values (0, 1, p - 1) in every column.
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{AbstractField, Field};
use p3_koala_bear::KoalaBear;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
mod common;
use common::{cpu_dft, gpu_dft, gpu_lde};
use tacit_ntt::{available, device};

fn rand_matrix(h: usize, w: usize, seed: u64) -> RowMajorMatrix<KoalaBear> {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % 0x7f000001) as u32
    };
    let vals: Vec<KoalaBear> = (0..h * w).map(|_| KoalaBear::from_canonical_u32(next())).collect();
    RowMajorMatrix::new(vals, w)
}

#[test]
fn dft_matches_cpu() {
    let Some(dev) = device() else {
        eprintln!("no Metal device here; the CPU is the only path");
        return;
    };
    assert!(available());
    let cases = [
        (1usize << 10, 32usize),
        (1 << 12, 64),
        (1 << 14, 96),
        (1 << 16, 32),
        (1 << 13, 160),
    ];
    for (i, &(h, w)) in cases.iter().enumerate() {
        let m = rand_matrix(h, w, 0x9e37 ^ (i as u64));
        let want = cpu_dft(&m);
        let got = gpu_dft(&m).unwrap_or_else(|| panic!("GPU declined {h}x{w}"));
        assert_eq!(got.height(), h);
        assert_eq!(got.width(), w);
        let g = common::gpu_rows(&got);
        for (k, (a, b)) in g.iter().zip(&want).enumerate() {
            assert_eq!(a, b, "{dev}: {h}x{w} cell {k} (row {}, col {}) differs", k / w, k % w);
        }
        println!("{dev}: {h}x{w} matches the CPU");
    }
}

#[test]
fn dft_edge_values() {
    if device().is_none() {
        return;
    }
    // columns set to 0, 1, -1 and p - 1, the values a reduction can get wrong
    let (h, w) = (1usize << 12, 64usize);
    let edges = [
        KoalaBear::zero(),
        KoalaBear::one(),
        KoalaBear::zero() - KoalaBear::one(),
        KoalaBear::from_canonical_u32(0x7f000000),
    ];
    let mut vals = vec![KoalaBear::zero(); h * w];
    for r in 0..h {
        for c in 0..w {
            vals[r * w + c] = edges[c % edges.len()];
        }
    }
    let m = RowMajorMatrix::new(vals, w);
    let want = cpu_dft(&m);
    let got = gpu_dft(&m).expect("GPU declined");
    let g: Vec<KoalaBear> = (0..got.height()).flat_map(|r| got.row(r).collect::<Vec<_>>()).collect();
    assert_eq!(g, want, "edge values");

    // and an all -1 matrix, where every sum saturates
    let ones = RowMajorMatrix::new(vec![KoalaBear::zero() - KoalaBear::one(); h * w], w);
    let want = cpu_dft(&ones);
    let got = gpu_dft(&ones).expect("GPU declined");
    let g: Vec<KoalaBear> = (0..got.height()).flat_map(|r| got.row(r).collect::<Vec<_>>()).collect();
    assert_eq!(g, want, "all -1");
    println!("edge values match");
}

#[test]
fn dft_matches_cpu_differential_stress() {
    if device().is_none() {
        return;
    }
    for i in 0..30u64 {
        let log_h = 5 + (i % 10) as usize;
        let w = 32 * (1 + (i as usize % 6));
        let m = rand_matrix(1 << log_h, w, 0xdeadbeef + i);
        let want = cpu_dft(&m);
        let got = gpu_dft(&m).expect("GPU declined");
        assert_eq!(common::gpu_rows(&got), want, "{i}: {}x{}", 1 << log_h, w);
    }
    println!("30 random shapes match");
}

#[test]
fn lde_matches_cpu() {
    if device().is_none() {
        return;
    }
    let cpu = Radix2DitParallel::default();
    for &(log_h, w, bits) in &[(10usize, 32usize, 2usize), (12, 64, 1)] {
        let m = rand_matrix(1 << log_h, w, 0x5151 + log_h as u64 + bits as u64);
        let ev = cpu.coset_lde_batch(m.clone(), bits, KoalaBear::one());
        let want: Vec<KoalaBear> = (0..ev.height()).flat_map(|r| ev.row(r).collect::<Vec<_>>()).collect();
        let got = gpu_lde(&m, bits).expect("GPU declined");
        assert_eq!(common::gpu_rows(&got), want, "lde {log_h}x{w}+{bits}");
        println!("lde {log_h}x{w} +{bits} matches the CPU");
    }
}
