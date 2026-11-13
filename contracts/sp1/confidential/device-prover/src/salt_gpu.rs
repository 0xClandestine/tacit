// Salt grinding on the GPU (Metal). Verified bit-exact against salt.rs address() for every nonce tried, and about 5x
// the throughput of the sixteen CPU cores (measured ~150 M attempts/s vs ~28 M/s). The grind is a pure fast path: if the
// device will not come up, or the shape is not one this kernel handles, the caller keeps the CPU threads and nothing changes.
//
// The three Keccak-256 hashes per attempt, exactly as address() does them:
//   guarded = keccak(00*12 || DEPLOYER(20) || salt(32))                 64 bytes, one block
//   proxy   = keccak(0xff || CREATEX(20) || guarded(32) || proxy(32))   85 bytes, two blocks
//   addr    = keccak(0xd6 0x94 || proxy[12..32) || 0x01)               23 bytes, one block
// Only salt[24..32) varies per thread, so the host builds the fixed bytes once and the kernel writes the counter.

#![cfg(target_os = "macos")]

use std::sync::atomic::{AtomicBool, Ordering};

// What one grind lane needs to know, mirrored from salt.rs. Kept here so the GPU crate stays free of field types.
pub struct Shape {
    pub g: [u8; 64],        // 00*12 || DEPLOYER || salt[0..32), with salt[24..32) zeroed
    pub createx: [u8; 20],
    pub proxy_hash: [u8; 32],
}

extern "C" {
    fn tacit_salt_gpu_available() -> i32;
    fn tacit_salt_gpu_grind(shape: *const u8, base: u64, target: u32, out: *mut u64, max: u32) -> i32;
    #[cfg(test)]
    fn tacit_salt_gpu_dump(shape: *const u8, base: u64, n: u32, out: *mut u8) -> i32;
}

static PROBED: AtomicBool = AtomicBool::new(false);
static AVAILABLE: AtomicBool = AtomicBool::new(false);

/// Whether the device came up and the MSL compiled. Probed once.
pub fn available() -> bool {
    if !PROBED.swap(true, Ordering::Relaxed) {
        let ok = unsafe { tacit_salt_gpu_available() } == 1;
        AVAILABLE.store(ok, Ordering::Relaxed);
        if ok { eprintln!("tacit-prover: grinding salts on the GPU"); }
    }
    AVAILABLE.load(Ordering::Relaxed)
}

/// Grind one batch and collect the nonces whose address has at least `target` leading zero bits.
/// Returns false only if the GPU could not run, so the caller can fall back to the CPU threads.
pub fn grind(shape: &Shape, base: u64, target: u32, out: &mut Vec<u64>, max: u32) -> bool {
    // The C side reads one flat buffer: g(64) || createx(20) || proxy_hash(32).
    let mut flat = [0u8; 64 + 20 + 32];
    flat[..64].copy_from_slice(&shape.g);
    flat[64..84].copy_from_slice(&shape.createx);
    flat[84..116].copy_from_slice(&shape.proxy_hash);
    let mut buf = vec![0u64; max as usize];
    let n = unsafe { tacit_salt_gpu_grind(flat.as_ptr(), base, target, buf.as_mut_ptr(), max) };
    if n < 0 { return false; }
    out.extend_from_slice(&buf[..n as usize]);
    true
}

pub const BATCH: u64 = 1 << 22; // attempts per dispatch: 4.2M, about 28 ms at 150 M/s

#[cfg(test)]
mod tests {
    use super::*;

    fn shape_for(day: u16) -> (Shape, [u8; 32]) {
        let proxy_hash = crate::salt::keccak_test(&[&crate::salt::PROXY_INIT_TEST]);
        let mut salt = [0u8; 32];
        salt[..20].copy_from_slice(&crate::salt::DEPLOYER_TEST);
        salt[21..23].copy_from_slice(&day.to_be_bytes());
        salt[23] = 0x01;
        let mut g = [0u8; 64];
        g[12..32].copy_from_slice(&crate::salt::DEPLOYER_TEST);
        g[32..64].copy_from_slice(&salt);
        (Shape { g, createx: crate::salt::CREATEX_TEST, proxy_hash }, salt)
    }

    // The whole point: the GPU's addresses must equal salt.rs address() byte for byte.
    #[test]
    fn gpu_address_matches_the_cpu() {
        if !available() { eprintln!("no Metal device; skipping"); return; }
        let (shape, salt) = shape_for(20770);
        let mut flat = [0u8; 116];
        flat[..64].copy_from_slice(&shape.g);
        flat[64..84].copy_from_slice(&shape.createx);
        flat[84..116].copy_from_slice(&shape.proxy_hash);

        let n = 64u32;
        let mut out = vec![0u8; (20 * n) as usize];
        let rc = unsafe { tacit_salt_gpu_dump(flat.as_ptr(), 0, n, out.as_mut_ptr()) };
        assert_eq!(rc, 0, "dump failed");

        for i in 0..n as u64 {
            let mut s = salt;
            s[24..32].copy_from_slice(&i.to_be_bytes());
            let (want, _) = crate::salt::address(&s, &shape.proxy_hash);
            let got = &out[(i as usize) * 20..(i as usize) * 20 + 20];
            assert_eq!(got, &want[..], "nonce {i}: gpu {got:02x?} != cpu {want:02x?}");
        }
        eprintln!("64/{n} nonces agree");
    }
}
