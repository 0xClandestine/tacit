// Two threads using the GPU at once, as SP1's parallel proving stages do. Each thread's result must match its own
// single-threaded result exactly, and neither may deadlock.
use std::sync::Arc;

#[test]
fn two_threads_at_once() {
    if tacit_ntt::device().is_none() { println!("no GPU"); return; }
    let (h, w) = (1usize << 15, 32usize);
    let a: Vec<u32> = (0..h * w).map(|i| (i as u32) % 0x7f000001).collect();
    let b: Vec<u32> = (0..h * w).map(|i| ((i as u32) * 7 + 11) % 0x7f000001).collect();
    let want_a = tacit_ntt::bitrev_rows_for_test(&a, h, w);
    let want_b = tacit_ntt::bitrev_rows_for_test(&b, h, w);
    let (a, b) = (Arc::new(a), Arc::new(b));
    let hs: Vec<_> = [(a.clone(), want_a), (b.clone(), want_b)].into_iter().map(|(src, want)| {
        std::thread::spawn(move || {
            for _ in 0..8 {
                let got = tacit_ntt::bitrev_rows_for_test(&src, 1 << 15, 32);
                assert!(got == want, "thread result drifted from its own single-threaded run");
            }
            true
        })
    }).collect();
    for t in hs { assert!(t.join().unwrap()); }
    println!("two threads, 8 rounds each: consistent, no deadlock");
}
