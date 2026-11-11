// Gemm-like radix-2 DIT layer over KoalaBear, matching p3-dft's Radix2DitParallel bit for bit.
//
// Elements are p3-koala-bear's Montgomery form (x·2^32 mod p), exactly as they sit in memory, so there is no
// conversion at either end: p3's monty_reduce is reproduced here the same way crates/leafhash does it.

#include <metal_stdlib>
using namespace metal;

constant uint P = 0x7f000001u;      // the KoalaBear prime
constant uint MU = 0x81000001u;     // p^-1 mod 2^32, p3's convention (no negation)

// p3-koala-bear's monty_reduce for x < p·2^32.
inline uint red(ulong x) {
    ulong t = (x * (ulong)MU) & 0xffffffffu;
    ulong u = t * (ulong)P;
    uint hi = (uint)((x - u) >> 32);
    return x < u ? hi + P : hi;
}

inline uint mul(uint a, uint b) { return red((ulong)a * (ulong)b); }
inline uint add(uint a, uint b) { uint s = a + b; return s >= P ? s - P : s; }
inline uint sub(uint a, uint b) { return a >= b ? a - b : a - b + P; }

// Butterfly: (a + b·w, a - b·w).
inline void butterfly(uint a, uint b, uint w, thread uint &lo, thread uint &hi) {
    uint t = mul(b, w);
    lo = add(a, t);
    hi = sub(a, t);
}

// Reverse the low `bits` bits of x.
inline uint revbits(uint x, uint bits) {
    uint r = 0u;
    for (uint b = 0u; b < bits; b++) { r = (r << 1) | (x & 1u); x >>= 1; }
    return r;
}

struct ChunkParams {
    uint log_h;
    uint width;
    uint gathered;
    uint chunk;    // rows per chunk
    uint first;    // first layer of this half
    uint last;     // one past the last
    uint pad[2];
};

struct LayerParams {
    uint log_h;     // log2 of the height (rows)
    uint layer;     // layer index l: the butterfly pairs rows 2^l apart
    uint width;     // cells per row (a multiple of 32)
    uint gathered;  // 0: twiddle per block (first half), 1: per block, bit-reversed twiddles (second half)
    uint pad[4];    // fixed-size struct so both sides of the boundary agree
};

// One layer, one thread per (butterfly, cell), matching p3-dft's par_dit_layer / par_dit_layer_rev exactly.
//
// The CPU splits the matrix into chunks of 2^mid rows and runs layers 0..mid inside each chunk (par_dit_layer), then
// bit-reverses and runs layers mid..log_h inside chunks of 2^(log_h-mid) rows (par_dit_layer_rev). Within a chunk,
// layer l pairs rows 2^l apart, counting from the chunk's own start; the twiddle index and the block size are what
// differ between the two halves.
kernel void kb_ntt_layer(device const uint *__restrict__ src [[buffer(0)]],
                         device uint *__restrict__ dst [[buffer(1)]],
                         device const uint *__restrict__ tw [[buffer(2)]],
                         constant LayerParams &p [[buffer(3)]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint tid [[thread_position_in_grid]]) {
    uint h = 1u << p.log_h;
    uint mid = p.log_h >> 1;
    uint half_block, chunk, tw_stride;
    if (p.gathered == 0u) {
        // par_dit_layer: chunks of 2^mid rows, layer l pairs rows 2^l apart, twiddle step 2^(log_h-1-l).
        half_block = 1u << p.layer;
        chunk = 1u << mid;
        tw_stride = 1u << (p.log_h - 1u - p.layer);
    } else {
        // par_dit_layer_rev: chunks of 2^(log_h-mid) rows, layer l pairs rows 2^(log_h-1-l) apart, one twiddle per block.
        half_block = 1u << (p.log_h - 1u - p.layer);
        chunk = h >> mid;
        tw_stride = 0u;
    }
    uint block = half_block << 1;

    // One simdgroup covers 32 cells of one butterfly; the last one may be partial when the width is not a multiple
    // of 32 (the prover's widths are not), so threads past the row's end do nothing.
    uint sgs_per_butterfly = (p.width + 31u) >> 5;
    uint butterflies = h >> 1;
    if (tid >= butterflies * sgs_per_butterfly * 32u) return;

    uint sg = tid >> 5;
    uint bf = sg / sgs_per_butterfly;
    uint cell = (sg % sgs_per_butterfly) << 5;
    if (cell + lane >= p.width) return;
    if (bf >= butterflies) return;

    // Which chunk and which block inside it.
    uint blocks_per_chunk = chunk / block;
    uint chunk_index = bf / (blocks_per_chunk * half_block);
    uint in_chunk = bf % (blocks_per_chunk * half_block);
    uint block_index = in_chunk / half_block;
    uint i = in_chunk % half_block;

    uint block_start = chunk_index * chunk + block_index * block;
    uint hi_row = block_start + i;
    uint lo_row = hi_row + half_block;

    // par_dit_layer strides the whole table by 2^(log_h-1-l); par_dit_layer_rev hands each chunk its own slice and
    // passes one twiddle per block, so the index restarts in every chunk.
    // In the reversed half each chunk gets twiddles_rev[first_block..] where first_block = thread * 2^(layer-mid).
    uint first_block = chunk_index << (p.layer - mid);
    uint twiddle = (p.gathered == 0u) ? tw[i * tw_stride] : tw[first_block + block_index];

    uint oa = hi_row * p.width + cell + lane;
    uint ob = lo_row * p.width + cell + lane;

    uint lo, hi;
    butterfly(src[oa], src[ob], twiddle, lo, hi);

    // Both halves write in place; the bit-reversals between them reorder the rows.
    dst[oa] = lo;
    dst[ob] = hi;
}

// Reorder the rows into bit-reversed order. One thread per row: a thread copies its whole row, so any width works,
// including widths that are not a multiple of four (the prover uses 5 and 21). The row is copied as uint4s where it
// divides by four and one uint at a time for the remainder, so the vectorised case stays vectorised.
kernel void kb_bitrev(device const uint *__restrict__ src [[buffer(0)]],
                      device uint *__restrict__ dst [[buffer(1)]],
                      constant uint &log_h [[buffer(2)]],
                      constant uint &width [[buffer(3)]],
                      uint row [[thread_position_in_grid]]) {
    if (row >= (1u << log_h)) return;
    uint r = row, rev = 0u;
    for (uint b = 0u; b < log_h; b++) { rev = (rev << 1) | (r & 1u); r >>= 1; }
    uint so = rev * width, dof = row * width;
    uint w4 = width >> 2;
    uint i = 0u;
    for (; i < w4; i++) {
        uint a = so + (i << 2), b = dof + (i << 2);
        *((device uint4 *)(dst + b)) = *((const device uint4 *)(src + a));
    }
    for (uint c = (w4 << 2); c < width; c++) dst[dof + c] = src[so + c];
}

// A whole half of the network in one dispatch: one threadgroup per chunk, running layers `first..last` over that
// chunk in registers-to-memory order. This is the CPU's par_dit_layer loop, with the chunk held on the GPU instead of
// reading and writing the whole matrix once per layer.
//
// Each thread owns one column of the chunk's rows and walks all of the chunk's butterflies, so a layer needs no
// cross-thread communication and no barrier between layers.
kernel void kb_ntt_chunk(device uint *__restrict__ buf [[buffer(0)]],
                         device const uint *__restrict__ tw [[buffer(1)]],
                         constant ChunkParams &p [[buffer(2)]],
                         uint lane [[thread_index_in_threadgroup]],
                         uint tgid [[threadgroup_position_in_grid]]) {
    // Columns are interleaved across a 32-wide simdgroup; each threadgroup of 32 threads covers 32 columns.
    uint cols = (p.width + 31u) >> 5;          // threadgroups per chunk
    uint chunk_of_tg = tgid / cols;
    uint col0 = (tgid % cols) << 5;

    uint chunk_start = chunk_of_tg * p.chunk;
    for (uint layer = p.first; layer < p.last; layer++) {
        uint half_block, tw_stride;
        if (p.gathered == 0u) {
            half_block = 1u << layer;
            tw_stride = 1u << (p.log_h - 1u - layer);
        } else {
            half_block = 1u << (p.log_h - 1u - layer);
            tw_stride = 0u;
        }
        uint block = half_block << 1;
        for (uint bs = 0; bs < p.chunk; bs += block) {
            for (uint i = 0; i < half_block; i++) {
                uint gi = chunk_start + bs + i;
                uint gl = gi + half_block;
                if (col0 + lane >= p.width) continue;
                uint twiddle;
                if (p.gathered == 0u) {
                    twiddle = tw[i * tw_stride];
                } else {
                    uint first_block = chunk_of_tg << (layer - (p.log_h >> 1));
                    twiddle = tw[first_block + (bs / block)];
                }
                uint oa = gi * p.width + col0 + lane;
                uint ob = gl * p.width + col0 + lane;
                uint lo, hi;
                butterfly(buf[oa], buf[ob], twiddle, lo, hi);
                buf[oa] = lo;
                buf[ob] = hi;
            }
        }
    }
}

