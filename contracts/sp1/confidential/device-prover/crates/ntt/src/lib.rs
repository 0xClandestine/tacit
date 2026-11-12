//! The radix-2 DIT NTT over KoalaBear rows on an Apple GPU, bit-identical to p3-dft's `Radix2DitParallel`.
//!
//! `Radix2DitParallel::dft_batch` (the DFT slop-basefold drives for every SP1 core, shrink and wrap commitment) is a
//! bit-reversal, a scattered DIT first half of `mid = log_h / 2` layers, a bit-reversal, a gathered DIT second half
//! of the rest, and a final bit-reversal. Each half is one dispatch here, with the same twiddle tables the CPU uses,
//! so the output is the same array of field elements the CPU produces.
//!
//! This crate works on raw `u32` Montgomery limbs and has no dependencies of its own, so the p3 crates can patch a
//! GPU path in without a dependency cycle. It does the dispatch scheduling and the memory; every field operation is
//! in `ntt.metal`, where KoalaBear's `monty_reduce` is reproduced exactly as p3-koala-bear computes it.
//!
//! Elsewhere than macOS, or with `TACIT_METAL=0`, the GPU is reported unavailable and callers keep their CPU path.

use std::sync::OnceLock;

/// The device the kernel runs on, if any.
pub fn device() -> Option<&'static str> {
    gpu().map(|g| g.name.as_str())
}

/// Whether the kernels can run here.
pub fn available() -> bool {
    gpu().is_some()
}

/// The narrowest row the kernels take: the bit-reversal moves 4 cells per thread.
pub const MIN_WIDTH: usize = 4;

/// One forward DFT of an `h x width` matrix of KoalaBear Montgomery limbs, in the order the CPU's
/// `Radix2DitParallel::dft_batch` leaves its matrix (bit-reversed rows included). `tw_fwd` and `tw_rev` are the powers
/// of the 2^log_h root of unity and the same table bit-reversed, as the CPU builds them. `None` when the GPU is
/// unavailable, the shape is one the kernels do not take, or a dispatch fails.
pub fn dft_in_place(
    values: &mut [u32],
    width: usize,
    tw_fwd: &[u32],
    tw_rev: &[u32],
) -> Option<()> {
    let h = if width == 0 { 0 } else { values.len() / width };
    if gpu().is_none() || width == 0 || values.len() % width != 0 {
        return None;
    }
    let log_h = h.checked_ilog2().filter(|_| h.is_power_of_two())? as usize;
    if h < 2 || width < MIN_WIDTH || (h as u64) * (width as u64) > u32::MAX as u64 {
        return None;
    }
    if tw_fwd.len() < h / 2 || tw_rev.len() < h / 2 {
        return None;
    }
    if std::env::var("TACIT_NTT_TRACE").is_ok() {
        eprintln!("tacit-ntt: dft {h}x{width} log_h={log_h}");
    }
    let t_dft = std::time::Instant::now();
    let mid = log_h / 2;

    let mut a = Buffer::from_u32s(values);
    let mut b = Buffer::zeros(a.len);
    bit_reverse_2d(&mut a, &mut b, h, width)?;

    run_half(&mut a, tw_fwd, log_h, 0, mid as u32, width as u32, 0)?;
    bit_reverse_2d(&mut a, &mut b, h, width)?;
    run_half(&mut a, tw_rev, log_h, mid as u32, log_h as u32, width as u32, 1)?;
    a.read_into(values);
    if std::env::var("TACIT_NTT_TRACE").is_ok() {
        eprintln!("tacit-ntt: dft {h}x{width} took {:.3}s", t_dft.elapsed().as_secs_f64());
    }
    Some(())
}

/// A forward DFT of `h x width` limbs, returning the new buffer.
pub fn dft_u32(values: &[u32], width: usize, tw_fwd: &[u32], tw_rev: &[u32]) -> Option<Vec<u32>> {
    let mut v = values.to_vec();
    dft_in_place(&mut v, width, tw_fwd, tw_rev)?;
    Some(v)
}

/// An inverse DFT of an `h x width` matrix, scaled by `h^-1`, leaving the rows physically bit-reversed (the CPU's
/// `coset_lde_batch` state before padding). `h_inv` is the field inverse of `h` in Montgomery form.
pub fn idft_scaled_u32(
    values: &[u32],
    width: usize,
    tw_inv: &[u32],
    tw_inv_rev: &[u32],
    h_inv: u32,
) -> Option<Vec<u32>> {
    let h = if width == 0 { 0 } else { values.len() / width };
    if gpu().is_none() || width == 0 || values.len() % width != 0 || width < MIN_WIDTH {
        return None;
    }
    let log_h = h.checked_ilog2().filter(|_| h.is_power_of_two())? as usize;
    if h < 2 || (h as u64) * (width as u64) > u32::MAX as u64 {
        return None;
    }
    if tw_inv.len() < h / 2 || tw_inv_rev.len() < h / 2 {
        return None;
    }
    let mid = log_h / 2;
    if std::env::var("TACIT_NTT_TRACE").is_ok() {
        eprintln!("tacit-ntt: idft {h}x{width} log_h={log_h}");
    }
    let mut a = Buffer::from_u32s(values);
    let mut b = Buffer::zeros(a.len);
    bit_reverse_2d(&mut a, &mut b, h, width)?;
    run_half(&mut a, tw_inv, log_h, 0, mid as u32, width as u32, 0)?;
    bit_reverse_2d(&mut a, &mut b, h, width)?;
    run_half(&mut a, tw_inv_rev, log_h, mid as u32, log_h as u32, width as u32, 1)?;
    scale_rows(&mut a, h_inv);
    Some(a.to_u32s())
}

/// Multiplies every cell by `k` in the field, on the CPU: one pass, cheaper than a dispatch.
fn scale_rows(b: &mut Buffer, k: u32) {
    const P: u64 = 0x7f00_0001;
    const MU: u64 = 0x8100_0001;
    let v = b.as_u32s_mut();
    for x in v.iter_mut() {
        // monty_reduce(x * k), the same reduction the kernel does.
        let z = (*x as u64) * (k as u64);
        let t = (z.wrapping_mul(MU)) & 0xffff_ffff;
        let u = t.wrapping_mul(P);
        let hi = (z.wrapping_sub(u) >> 32) as u32;
        *x = if z < u { hi.wrapping_add(P as u32) } else { hi };
    }
}

/// Layers `first..last` of one half, in place on `buf`.
fn run_half(
    buf: &mut Buffer,
    tw: &[u32],
    log_h: usize,
    first: u32,
    last: u32,
    width: u32,
    gathered: u32,
) -> Option<()> {
    if first >= last {
        return Some(());
    }
    let t = Buffer::from_u32s(tw);
    let rc = unsafe { tacit_ntt_half(buf.ptr, t.ptr, log_h as u32, first, last, width, gathered) };
    if rc != 0 {
        eprintln!("tacit-ntt: half dispatch returned {rc}");
    }
    (rc == 0).then_some(())
}

/// Bit-reverses the row order of an `h x width` buffer, from `a` into `b`, and leaves the result in `a`.
fn bit_reverse_2d(a: &mut Buffer, b: &mut Buffer, h: usize, width: usize) -> Option<()> {
    if h < 2 {
        return Some(());
    }
    let log_h = h.trailing_zeros();
    debug_assert_eq!(1usize << log_h, h);
    let rc = unsafe { tacit_ntt_bitrev(a.ptr, b.ptr, log_h, width as u32) };
    if rc != 0 {
        return None;
    }
    std::mem::swap(a, b);
    Some(())
}

/// A page-aligned `u32` buffer handed to the GPU in place (unified memory, no copy).
struct Buffer {
    ptr: *mut u32,
    len: usize,
}

// The pointer is only ever dereferenced while a dispatch holds it, and each dispatch is synchronous.
unsafe impl Send for Buffer {}

impl Buffer {
    fn zeros(len: usize) -> Self {
        let bytes = len.max(1) * 4;
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        // 16 KiB: a page on Apple silicon, and the GPU's preferred offset alignment.
        let rc = unsafe { posix_memalign(&mut p, 1 << 14, bytes) };
        assert_eq!(rc, 0, "posix_memalign failed");
        unsafe { std::ptr::write_bytes(p as *mut u8, 0, bytes) };
        Self { ptr: p as *mut u32, len }
    }

    fn from_u32s(v: &[u32]) -> Self {
        let mut b = Self::zeros(v.len());
        b.as_u32s_mut().copy_from_slice(v);
        b
    }

    fn as_u32s_mut(&mut self) -> &mut [u32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    fn to_u32s(&self) -> Vec<u32> {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }.to_vec()
    }

    fn read_into(&self, out: &mut [u32]) {
        out.copy_from_slice(unsafe { std::slice::from_raw_parts(self.ptr, self.len) });
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { free(self.ptr as *mut std::ffi::c_void) };
    }
}

struct Gpu {
    name: String,
}

fn gpu() -> Option<&'static Gpu> {
    static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
    GPU.get_or_init(init).as_ref()
}

#[cfg(target_os = "macos")]
extern "C" {
    fn tacit_ntt_init(msl: *const u8, name: *mut u8, name_len: usize) -> i32;
    fn tacit_ntt_bitrev(src: *mut u32, dst: *mut u32, log_h: u32, width: u32) -> i32;
    /// A whole half of the network in one dispatch, in place on `buf`.
    fn tacit_ntt_half(
        buf: *mut u32,
        tw: *const u32,
        log_h: u32,
        first: u32,
        last: u32,
        width: u32,
        gathered: u32,
    ) -> i32;
}

#[cfg(target_os = "macos")]
fn init() -> Option<Gpu> {
    if std::env::var("TACIT_METAL").as_deref() == Ok("0") || std::env::var("TACIT_NTT").as_deref() == Ok("0") {
        return None;
    }
    let msl: Vec<u8> = match std::env::var("TACIT_NTT_MSL") {
        // Only for the differential tests: run an alternative kernel source without rebuilding.
        Ok(p) => {
            let mut v = std::fs::read(&p).expect("TACIT_NTT_MSL file");
            v.push(0);
            v
        }
        Err(_) => concat!(include_str!("ntt.metal"), "\0").as_bytes().to_vec(),
    };
    let mut name = [0u8; 128];
    let rc = unsafe { tacit_ntt_init(msl.as_ptr(), name.as_mut_ptr(), name.len()) };
    if rc != 0 {
        return None;
    }
    let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
    Some(Gpu { name: String::from_utf8_lossy(&name[..end]).into_owned() })
}

#[cfg(not(target_os = "macos"))]
fn init() -> Option<Gpu> {
    None
}

// Non-macOS builds still need the symbols for the tests above to compile.
#[cfg(not(target_os = "macos"))]
extern "C" {
    #[allow(dead_code)]
    fn tacit_ntt_init(msl: *const u8, name: *mut u8, name_len: usize) -> i32;
    #[allow(dead_code)]
    fn tacit_ntt_bitrev(src: *mut u32, dst: *mut u32, log_h: u32, width: u32) -> i32;
    #[allow(dead_code)]
    fn tacit_ntt_half(
        buf: *mut u32,
        tw: *const u32,
        log_h: u32,
        first: u32,
        last: u32,
        width: u32,
        gathered: u32,
    ) -> i32;
}

extern "C" {
    fn tacit_ntt_fs_stage(
        buf: *mut u32,
        tw: *const u32,
        tw_rev: *const u32,
        n_log: u32,
        width: u32,
        step: u32,
        base: u32,
        subs: u32,
    ) -> i32;
}

extern "C" {
    fn free(p: *mut std::ffi::c_void);
    fn posix_memalign(p: *mut *mut std::ffi::c_void, align: usize, size: usize) -> i32;
}

/// Test hook: one four-step stage exactly as the driver calls it.
pub fn fs_stage_for_test(buf: &mut [u32], tw: &[u32], tw_rev: &[u32], n_log: u32, width: u32, step: u32, base: u32, subs: u32) -> i32 {
    if gpu().is_none() { return -1; }
    unsafe { tacit_ntt_fs_stage(buf.as_mut_ptr(), tw.as_ptr(), tw_rev.as_ptr(), n_log, width, step, base, subs) }
}

/// Test hook: the GPU's row bit-reversal applied to an `h x width` matrix.
pub fn bitrev_rows_for_test(v: &[u32], h: usize, width: usize) -> Vec<u32> {
    let mut a = Buffer::from_u32s(v);
    let mut b = Buffer::zeros(v.len());
    assert!(bit_reverse_2d(&mut a, &mut b, h, width).is_some());
    a.to_u32s()
}

/// Test hook: one layer of the network over `src`, without the bit-reversals around it.
pub fn debug_layer(
    src: &[u32],
    out: &mut [u32],
    tw: &[u32],
    log_h: usize,
    layer: u32,
    width: usize,
    gathered: u32,
) -> bool {
    if gpu().is_none() || src.len() != out.len() || src.len() != (1 << log_h) * width {
        return false;
    }
    let mut a = Buffer::from_u32s(src);
    let ok = run_half(&mut a, tw, log_h, layer, layer + 1, width as u32, gathered).is_some();
    a.read_into(out);
    ok
}

// ---- four-step (Bailey) NTT --------------------------------------------------
//
// The layer-by-layer kernel reads and writes the whole matrix once per layer. The four-step splits N = n1*n2 so each
// sub-DFT is short enough to hold in threadgroup memory and is done in one pass: three passes instead of log_h.
//
//   with x[i + n1*j] and output index k2 + n2*k1,
//   X[k2 + n2*k1] = sum_i [ w^(i*k2) * (w^n2)^(i*k1) ] * sum_j x[i+n1 j] * (w^n1)^(j*k2)
//
// stage 1: length-n2 DFT along j (stride n1), root w^n1
// stage 2: multiply (i,k2) by w^(i*k2)
// stage 3: length-n1 DFT along i (stride 1), root w^n2, read out to index k2 + n2*k1
//
// Each sub-DFT is the network of `kb_fs_tile`, which is p3's Radix2DitParallel: input bit-reversal, dit_layer for
// layers 0..mid, the row reversal, then dit_layer_rev for layers mid..n_log with reversed twiddles.

/// Whether the four-step kernel can do this shape. Even `log_h` only: the two sub-DFTs are then the same length, and
/// an odd one needs the stage tables of a different length each time, which is not right yet (log_h 15, width 21).
pub fn fourstep_supported(log_h: usize, width: usize) -> bool {
    available() && (2..=24).contains(&log_h) && width > 0 && log_h % 2 == 0 && log_h / 2 <= 12
}

/// A four-step DFT of an `h x width` matrix, in place, in the same physical row order [`dft_in_place`] leaves.
pub fn dft_fourstep_in_place(values: &mut [u32], width: usize, tw_fwd: &[u32]) -> Option<()> {
    let h = if width == 0 { 0 } else { values.len() / width };
    if width == 0 || h == 0 || !h.is_power_of_two() {
        return None;
    }
    let log_h = h.trailing_zeros() as usize;
    if !fourstep_supported(log_h, width) {
        return None;
    }
    let n1_log = log_h / 2;
    let n2_log = log_h - n1_log;
    let n1 = 1usize << n1_log;
    let _ = gpu()?;
    let n2 = 1usize << n2_log;

    // Stage 1: length-n2 DFT along j (the slow index, stride n1), root w^n1.
    let (tw1, tw1r) = stage_tables(tw_fwd, log_h, n2_log);
    let r = unsafe {
        tacit_ntt_fs_stage(values.as_mut_ptr(), tw1.as_ptr(), tw1r.as_ptr(), n2_log as u32, width as u32, n1 as u32, 1, n1 as u32)
    };
    if r != 0 {
        if std::env::var("TACIT_NTT_DEBUG").is_ok() { eprintln!("fourstep: stage 1 rc={r}"); }
        return None;
    }
    // Stage 2: twiddle (i,k2) by w^(i*k2); the full table holds w^k at [k].
    // w^(i*k2): reduce the exponent mod h and look it up in the full table, which is the half table followed by
    // its negation. The limbs are Montgomery form and negation there is p - x, same as in the canonical domain.
    let half_h = h / 2;
    let mut tw_full: Vec<u32> = Vec::with_capacity(h);
    tw_full.extend_from_slice(&tw_fwd[..half_h]);
    for k in 0..half_h {
        tw_full.push(neg_cpu(tw_fwd[k]));
    }
    for i in 0..n1 {
        for k2 in 0..n2 {
            let tw = tw_full[(i * k2) % h];
            // stage 1 left frequency k2 at physical row br(k2); scale it there, in place
            let row = (i + n1 * rev_bits(k2, n2_log)) * width;
            for c in 0..width {
                values[row + c] = mul_cpu(values[row + c], tw);
            }
        }
    }
    // Stage 3: length-n1 DFT along i (stride 1), root w^n2, then read out to k2 + n2*k1.
    let (tw3, tw3r) = stage_tables(tw_fwd, log_h, n1_log);
    let r = unsafe {
        tacit_ntt_fs_stage(values.as_mut_ptr(), tw3.as_ptr(), tw3r.as_ptr(), n1_log as u32, width as u32, 1, n1 as u32, n2 as u32)
    };
    if r != 0 {
        if std::env::var("TACIT_NTT_DEBUG").is_ok() { eprintln!("fourstep: stage 3 rc={r}"); }
        return None;
    }
    // Stage 3 already leaves the rows in the physical order `dft_in_place` does: measured against it directly at
    // log_h 4 with width 16, every row matched with no readout at all. Nothing to permute here; the values and their
    // placement were both right once the twiddles were, and a spurious reversal of my own is what hid that.
    Some(())
}

/// The forward and reversed twiddle tables for a sub-DFT: w_sub = w^(2^(log_h - n_log)).
fn stage_tables(tw_fwd: &[u32], log_h: usize, n_log: usize) -> (Vec<u32>, Vec<u32>) {
    let stride = 1usize << (log_h - n_log);
    let fwd: Vec<u32> = (0..(1usize << n_log) / 2).map(|i| tw_fwd[i * stride]).collect();
    // p3 builds the second half's table by reversing the INDEX BITS, not by reversing the list.
    let mut rev = vec![0u32; fwd.len()];
    for (i, &v) in fwd.iter().enumerate() {
        rev[rev_bits(i, n_log - 1)] = v;
    }
    (fwd, rev)
}

/// Reverse the low `bits` bits of `x`.
fn rev_bits(x: usize, bits: usize) -> usize {
    let mut r = 0;
    let mut v = x;
    for _ in 0..bits { r = (r << 1) | (v & 1); v >>= 1; }
    r
}

/// -x mod p, for turning w^(k+h/2) = -w^k back into a table lookup.
fn neg_cpu(a: u32) -> u32 {
    if a == 0 { 0 } else { 0x7f000001u32.wrapping_sub(a) }
}

/// One KoalaBear multiply in Montgomery form, for the twiddle pass between stages.
fn mul_cpu(a: u32, b: u32) -> u32 {
    const P: u64 = 0x7f000001;
    const MU: u64 = 0x81000001;
    let x = (a as u64) * (b as u64);
    let t = x.wrapping_mul(MU) & 0xffff_ffff;
    let u = t.wrapping_mul(P);
    let hi = (x.wrapping_sub(u) >> 32) as u32;
    if x < u { hi.wrapping_add(P as u32) } else { hi }
}
