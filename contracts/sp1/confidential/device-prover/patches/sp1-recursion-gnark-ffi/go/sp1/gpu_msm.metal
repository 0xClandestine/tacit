// BN254 G1 bucket sums for the Groth16 step's multi-exponentiations (gpu_msm_darwin.go): Pippenger with 16-bit windows, the
// points sorted into their buckets on the CPU, one GPU thread per (window, bucket) adding its points. Fp in Montgomery form (R = 2^256) on eight little-endian 32-bit
// limbs, byte-identical to gnark-crypto's fp.Element; points are gnark's G1Affine (x, y; (0, 0) is infinity), sums are
// Jacobian (x = X/Z², y = Y/Z³; Z = 0 is infinity), as gnark's G1Jac.
#include <metal_stdlib>
using namespace metal;

struct F { uint l[8]; };
struct J { F x, y, z; };

constant uint P[8] = {0xd87cfd47u, 0x3c208c16u, 0x6871ca8du, 0x97816a91u, 0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u};
constant uint PINV = 0xe4866389u; // -P^-1 mod 2^32
constant uint ONE[8] = {0xc58f0d9du, 0xd35d438du, 0xf5c70b3du, 0x0a78eb28u, 0x7879462cu, 0x666ea36fu, 0x9a07df2fu, 0x0e0a77c1u};

inline F one() { F o; for (int i = 0; i < 8; i++) o.l[i] = ONE[i]; return o; }
inline F zero() { F o; for (int i = 0; i < 8; i++) o.l[i] = 0; return o; }
inline bool is_zero(F x) { uint a = 0; for (int i = 0; i < 8; i++) a |= x.l[i]; return a == 0; }

inline F sub_p_if_ge(F t) {
    F r; uint borrow = 0;
    for (int i = 0; i < 8; i++) {
        ulong d = (ulong)t.l[i] - (ulong)P[i] - (ulong)borrow;
        r.l[i] = (uint)d;
        borrow = (uint)(d >> 63);
    }
    return borrow ? t : r;
}

// CIOS Montgomery multiplication; P < 2^254 leaves two spare bits, so the top word needs no extra carry.
inline F mul(F x, F y) {
    uint t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 8; i++) {
        ulong s = (ulong)t[0] + (ulong)x.l[0] * y.l[i];
        uint t0 = (uint)s; uint a = (uint)(s >> 32);
        uint m = t0 * PINV;
        ulong u = (ulong)t0 + (ulong)m * P[0];
        uint c = (uint)(u >> 32);
        for (int j = 1; j < 8; j++) {
            ulong s2 = (ulong)t[j] + (ulong)x.l[j] * y.l[i] + a;
            a = (uint)(s2 >> 32);
            ulong u2 = (ulong)(uint)s2 + (ulong)m * P[j] + c;
            c = (uint)(u2 >> 32);
            t[j - 1] = (uint)u2;
        }
        t[7] = c + a;
    }
    F o; for (int i = 0; i < 8; i++) o.l[i] = t[i];
    return sub_p_if_ge(o);
}

inline F add(F x, F y) {
    F o; uint carry = 0;
    for (int i = 0; i < 8; i++) {
        ulong s = (ulong)x.l[i] + (ulong)y.l[i] + carry;
        o.l[i] = (uint)s; carry = (uint)(s >> 32);
    }
    return sub_p_if_ge(o);
}

inline F sub(F x, F y) {
    F o; uint borrow = 0;
    for (int i = 0; i < 8; i++) {
        ulong d = (ulong)x.l[i] - (ulong)y.l[i] - (ulong)borrow;
        o.l[i] = (uint)d; borrow = (uint)(d >> 63);
    }
    if (borrow) {
        uint carry = 0;
        for (int i = 0; i < 8; i++) {
            ulong s = (ulong)o.l[i] + (ulong)P[i] + carry;
            o.l[i] = (uint)s; carry = (uint)(s >> 32);
        }
    }
    return o;
}

inline J infinity() { J r; r.x = one(); r.y = one(); r.z = zero(); return r; }

// dbl-2009-l (a = 0)
inline J jdouble(J p) {
    if (is_zero(p.z)) return p;
    F A = mul(p.x, p.x), B = mul(p.y, p.y), C = mul(B, B);
    F t = add(p.x, B);
    F D = sub(sub(mul(t, t), A), C); D = add(D, D);
    F E = add(add(A, A), A), G = mul(E, E);
    J r;
    r.x = sub(G, add(D, D));
    F C8 = add(C, C); C8 = add(C8, C8); C8 = add(C8, C8);
    r.y = sub(mul(E, sub(D, r.x)), C8);
    F yz = mul(p.y, p.z); r.z = add(yz, yz);
    return r;
}

// add-2007-bl: the sum of two Jacobian points, for folding a bucket's partial sums.
inline J jadd(J p, J q) {
    if (is_zero(p.z)) return q;
    if (is_zero(q.z)) return p;
    F Z1Z1 = mul(p.z, p.z), Z2Z2 = mul(q.z, q.z);
    F U1 = mul(p.x, Z2Z2), U2 = mul(q.x, Z1Z1);
    F S1 = mul(mul(p.y, q.z), Z2Z2), S2 = mul(mul(q.y, p.z), Z1Z1);
    F H = sub(U2, U1), rr = sub(S2, S1);
    if (is_zero(H)) return is_zero(rr) ? jdouble(p) : infinity();
    F HH = mul(H, H);
    F I = add(HH, HH); I = add(I, I);
    F Jv = mul(H, I);
    rr = add(rr, rr);
    F V = mul(U1, I);
    J r;
    r.x = sub(sub(mul(rr, rr), Jv), add(V, V));
    r.y = sub(mul(rr, sub(V, r.x)), mul(S1, add(Jv, Jv)));
    // add-2007-bl's z. The trailing *H is required here: madd's z-formula is for a mixed add (an affine second
    // operand), and copying its shape for Jacobian+Jacobian gives a correct-looking but wrong point.
    F zh = add(p.z, q.z);
    r.z = mul(sub(sub(mul(zh, zh), Z1Z1), Z2Z2), H);
    return r;
}

// madd-2007-bl: p + (x2, y2), the affine point (0, 0) being infinity.
inline J madd(J p, F x2, F y2) {
    if (is_zero(x2) && is_zero(y2)) return p;
    if (is_zero(p.z)) { J r; r.x = x2; r.y = y2; r.z = one(); return r; }
    F Z1Z1 = mul(p.z, p.z);
    F U2 = mul(x2, Z1Z1);
    F S2 = mul(mul(y2, p.z), Z1Z1);
    F H = sub(U2, p.x);
    F rr = sub(S2, p.y);
    if (is_zero(H)) return is_zero(rr) ? jdouble(p) : infinity();
    F HH = mul(H, H);
    F I = add(HH, HH); I = add(I, I);
    F Jv = mul(H, I);
    rr = add(rr, rr);
    F V = mul(p.x, I);
    J r;
    r.x = sub(sub(mul(rr, rr), Jv), add(V, V));
    F y1j = mul(p.y, Jv); y1j = add(y1j, y1j);
    r.y = sub(mul(rr, sub(V, r.x)), y1j);
    F zh = add(p.z, H);
    r.z = sub(sub(mul(zh, zh), Z1Z1), HH);
    return r;
}

// Adds the points whose digit in that window is bucket + 1 to the bucket's sum. A bucket's points are split across
// several threads (`sbase`/`scount` say how many, and where that bucket's slots begin), each taking a contiguous run
// and writing to `part`; bucket_fold then adds them up. One thread per bucket was the first version, and the witness
// scalars are nothing like uniform: in the lowest window one digit holds half of a 2M block, so a single thread walked
// a million points while the rest of the GPU sat idle, and the whole Groth16 step took 32 s against the CPU's 2 s.
kernel void bucket_sum(device const F *pts [[buffer(0)]],
                       device const uint *order [[buffer(1)]],
                       device const uint *start [[buffer(2)]],
                       device J *out [[buffer(3)]],
                       device J *part [[buffer(8)]],
                       constant uint &nb [[buffer(4)]],
                       constant uint &n [[buffer(5)]],
                       constant uint &windows [[buffer(6)]],
                       constant uint &first [[buffer(7)]],
                       constant uint &winslots [[buffer(9)]],
                       device const uint *sbase [[buffer(10)]],
                       device const uint *scount [[buffer(11)]],
                       uint gid [[thread_position_in_grid]]) {
    uint w = gid / winslots, slot = gid % winslots;
    if (w >= windows) return;
    device const uint *st = start + (ulong)w * (nb + 1);
    device const uint *ord = order + (ulong)w * n;
    device const uint *bs = sbase + (ulong)w * nb, *cs = scount + (ulong)w * nb;
    // slots past the last bucket's are padding from the fixed dispatch width; they must not run, or every one of
    // them would binary-search to the final bucket and overwrite the partials the fold is about to read.
    if (slot >= bs[nb - 1] + cs[nb - 1]) return;
    // find the bucket that owns this slot. Buckets are few relative to slots, so a scan from a running hint is not
    // worth it: binary search the prefix over per-bucket slot counts, which is monotone in b.
    uint lo_b = 0, hi_b = nb;
    while (lo_b + 1 < hi_b) {
        uint mid = (lo_b + hi_b) / 2;
        if (bs[mid] <= slot) lo_b = mid; else hi_b = mid;
    }
    uint b = lo_b;
    uint sp = slot - bs[b], nsp = cs[b];
    uint a0 = st[b], e0 = st[b + 1], len = e0 - a0;
    uint a = a0 + (uint)(((ulong)len * sp) / nsp);
    uint e = a0 + (uint)(((ulong)len * (sp + 1)) / nsp);
    J acc = infinity();
    for (uint k = a; k < e; k++) {
        uint i = ord[k];
        acc = madd(acc, pts[2 * (ulong)i], pts[2 * (ulong)i + 1]);
    }
    part[gid] = acc;
}

// Folds each bucket's `splits` partial sums into its running sum, carrying over the block before unless `first`.
kernel void bucket_fold(device J *out [[buffer(0)]],
                        device const J *part [[buffer(1)]],
                        constant uint &nb [[buffer(2)]],
                        constant uint &windows [[buffer(3)]],
                        constant uint &first [[buffer(4)]],
                        constant uint &winslots [[buffer(5)]],
                        device const uint *sbase [[buffer(6)]],
                        device const uint *scount [[buffer(7)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= nb * windows) return;
    uint w = gid / nb, b = gid % nb;
    uint slot = sbase[(ulong)w * nb + b], nsp = scount[(ulong)w * nb + b];
    J acc = first ? infinity() : out[gid];
    for (uint sp = 0; sp < nsp; sp++) {
        acc = jadd(acc, part[(ulong)w * winslots + slot + sp]);
    }
    out[gid] = acc;
}
