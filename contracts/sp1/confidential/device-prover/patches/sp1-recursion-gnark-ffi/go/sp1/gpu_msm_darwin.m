// Metal host for gpu_msm.metal: the pipeline and a queue built once; two sets of block buffers (points, per-window order and
// starts) and the bucket sums, made when a Groth16 step first needs them and freed after it (tacit_msm_release), so they do
// not sit through the other stages. While the GPU adds one block the CPU writes the next into the other set; the queue runs
// the blocks in order, and the bucket sums carry from one to the next. One caller at a time (gpu_msm_darwin.go locks).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <string.h>

static id<MTLDevice> g_dev;
static id<MTLComputePipelineState> g_pso, g_fold, g_pso2, g_fold2;
static id<MTLCommandQueue> g_queue;
// One set for G1 and one for G2 (g2 == 1). The two run one after the other in a Groth16 proof, and keeping them apart
// keeps the per-curve element sizes (64/96 against 128/192 bytes) out of every dispatch.
static id<MTLBuffer> g_pts[2][2], g_order[2][2], g_start[2][2], g_out[2], g_part[2][2], g_sbase[2][2], g_scount[2][2];
// How many points one bucket's thread is allowed to walk. The bucket sizes are wildly uneven (one digit of the
// lowest window holds half a block), so a bucket gets ceil(count/CAP) threads rather than a fixed number.
#define MSM_PER_THREAD 1024u
static uint32_t g_winslots = 1;
static id<MTLCommandBuffer> g_cb[2];
static uint32_t g_cap, g_nb, g_windows;

int tacit_msm_init(const char *msl, uint32_t cap, uint32_t nb, uint32_t windows, char *name, size_t name_len) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
            if (all.count) dev = all[0];
        }
        if (!dev) return -1;
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:msl] options:nil error:&err];
        if (!lib) return -2;
        id<MTLFunction> fn = [lib newFunctionWithName:@"bucket_sum"];
        id<MTLFunction> ff = [lib newFunctionWithName:@"bucket_fold"];
        id<MTLFunction> fn2 = [lib newFunctionWithName:@"bucket_sum_g2"];
        id<MTLFunction> ff2 = [lib newFunctionWithName:@"bucket_fold_g2"];
        id<MTLComputePipelineState> pso = fn ? [dev newComputePipelineStateWithFunction:fn error:&err] : nil;
        id<MTLComputePipelineState> fold = ff ? [dev newComputePipelineStateWithFunction:ff error:&err] : nil;
        id<MTLComputePipelineState> pso2 = fn2 ? [dev newComputePipelineStateWithFunction:fn2 error:&err] : nil;
        id<MTLComputePipelineState> fold2 = ff2 ? [dev newComputePipelineStateWithFunction:ff2 error:&err] : nil;
        // the G2 pipelines are optional: without them the G2 sum stays on the CPU
        if (!pso || !fold) return -3;
        g_queue = [dev newCommandQueue];
        if (!g_queue) return -4;
        g_dev = dev;
        g_pso = pso;
        g_fold = fold;
        g_pso2 = pso2;
        g_fold2 = fold2;
        g_cap = cap;
        g_nb = nb;
        g_windows = windows;
        strncpy(name, dev.name.UTF8String, name_len - 1);
        name[name_len - 1] = 0;
        return 0;
    }
}

// The buffers, made if they are not there. The G2 ones are 128- and 192-byte elements against G1's 64 and 96.
int tacit_msm_alloc(int g2) {
    if (g_pso2 == nil) g2 = 0;
    if (g_out[g2]) return 0;
    size_t affine = g2 ? 128 : 64, jac = g2 ? 192 : 96;
    for (int s = 0; s < 2; s++) {
        g_pts[g2][s] = [g_dev newBufferWithLength:(size_t)g_cap * affine options:MTLResourceStorageModeShared];
        g_order[g2][s] = [g_dev newBufferWithLength:(size_t)g_cap * g_windows * 4 options:MTLResourceStorageModeShared];
        g_start[g2][s] = [g_dev newBufferWithLength:(size_t)(g_nb + 1) * g_windows * 4 options:MTLResourceStorageModeShared];
        if (!g_pts[g2][s] || !g_order[g2][s] || !g_start[g2][s]) return -4;
    }
    g_out[g2] = [g_dev newBufferWithLength:(size_t)g_nb * g_windows * jac options:MTLResourceStorageModeShared];
    // At most cap/MSM_PER_THREAD + g_nb slots per window (every bucket gets at least one).
    g_winslots = g_cap / MSM_PER_THREAD + 2u * g_nb;
    // per set: the CPU fills set 1 for the next block while the GPU is still running set 0's fold
    for (int s = 0; s < 2; s++) {
        g_part[g2][s] = [g_dev newBufferWithLength:(size_t)g_winslots * g_windows * jac options:MTLResourceStorageModeShared];
        g_sbase[g2][s] = [g_dev newBufferWithLength:(size_t)g_nb * g_windows * 4 options:MTLResourceStorageModeShared];
        g_scount[g2][s] = [g_dev newBufferWithLength:(size_t)g_nb * g_windows * 4 options:MTLResourceStorageModeShared];
        if (!g_part[g2][s] || !g_sbase[g2][s] || !g_scount[g2][s]) return -4;
    }
    return g_out[g2] ? 0 : -4;
}

int tacit_msm_have_g2(void) { return g_pso2 != nil && g_fold2 != nil; }

// Frees one curve's buffers (g2 = -1 frees both), so the G1 sums' buffers do not sit through the G2 one or the stages after.
void tacit_msm_release(int g2) {
    for (int c = 0; c < 2; c++) {
        if (g2 >= 0 && c != g2) continue;
        for (int s = 0; s < 2; s++) {
            if (g_cb[s]) { [g_cb[s] waitUntilCompleted]; g_cb[s] = nil; }
            g_pts[c][s] = g_order[c][s] = g_start[c][s] = nil;
            g_part[c][s] = g_sbase[c][s] = g_scount[c][s] = nil;
        }
        g_out[c] = nil;
    }
}

// Where the caller writes set s's points, per-window order and starts, and reads the bucket sums.
void *tacit_msm_pts(int s, int g2) { return g_pts[g2][s].contents; }
void *tacit_msm_order(int s, int g2) { return g_order[g2][s].contents; }
void *tacit_msm_start(int s, int g2) { return g_start[g2][s].contents; }
void *tacit_msm_sbase(int s, int g2) { return g_sbase[g2][s].contents; }
void *tacit_msm_scount(int s, int g2) { return g_scount[g2][s].contents; }
uint32_t tacit_msm_winslots(void) { return g_winslots; }
void *tacit_msm_out(int g2) { return g_out[g2].contents; }

// Waits for the block last sent from set s; nonzero if it failed.
int tacit_msm_wait(int s) {
    if (!g_cb[s]) return 0;
    [g_cb[s] waitUntilCompleted];
    int rc = g_cb[s].error ? -11 : 0;
    g_cb[s] = nil;
    return rc;
}

// Sends set s's n points to be added into the bucket sums (from infinity when `first`), without waiting. g2 picks the
// curve: same kernels and same layout, over F2 rather than F.
int tacit_msm_send(int s, int g2, uint32_t n, uint32_t first) {
    @autoreleasepool {
        if (n > g_cap) return -10;
        if (g2 && (g_pso2 == nil || g_fold2 == nil || g_out[1] == nil)) return -12;
        if (!g2 && (g_pso == nil || g_fold == nil || g_out[0] == nil)) return -13;
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g2 ? g_pso2 : g_pso];
        [enc setBuffer:g_pts[g2][s] offset:0 atIndex:0];
        [enc setBuffer:g_order[g2][s] offset:0 atIndex:1];
        [enc setBuffer:g_start[g2][s] offset:0 atIndex:2];
        [enc setBuffer:g_out[g2] offset:0 atIndex:3];
        [enc setBytes:&g_nb length:4 atIndex:4];
        [enc setBytes:&n length:4 atIndex:5];
        [enc setBytes:&g_windows length:4 atIndex:6];
        [enc setBytes:&first length:4 atIndex:7];
        [enc setBuffer:g_part[g2][s] offset:0 atIndex:8];
        [enc setBytes:&g_winslots length:4 atIndex:9];
        [enc setBuffer:g_sbase[g2][s] offset:0 atIndex:10];
        [enc setBuffer:g_scount[g2][s] offset:0 atIndex:11];
        NSUInteger total = (NSUInteger)g_winslots * g_windows;
        NSUInteger tg = (g2 ? g_pso2 : g_pso).maxTotalThreadsPerThreadgroup;
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        // fold this block's per-bucket partials into the running bucket sums
        id<MTLComputeCommandEncoder> fe = [cb computeCommandEncoder];
        [fe setComputePipelineState:g2 ? g_fold2 : g_fold];
        [fe setBuffer:g_out[g2] offset:0 atIndex:0];
        [fe setBuffer:g_part[g2][s] offset:0 atIndex:1];
        [fe setBytes:&g_nb length:4 atIndex:2];
        [fe setBytes:&g_windows length:4 atIndex:3];
        [fe setBytes:&first length:4 atIndex:4];
        [fe setBytes:&g_winslots length:4 atIndex:5];
        [fe setBuffer:g_sbase[g2][s] offset:0 atIndex:6];
        [fe setBuffer:g_scount[g2][s] offset:0 atIndex:7];
        NSUInteger ftotal = (NSUInteger)g_nb * g_windows;
        NSUInteger ftg = (g2 ? g_fold2 : g_fold).maxTotalThreadsPerThreadgroup;
        if (ftg > 256) ftg = 256;
        [fe dispatchThreads:MTLSizeMake(ftotal, 1, 1) threadsPerThreadgroup:MTLSizeMake(ftg, 1, 1)];
        [fe endEncoding];
        [cb commit];
        g_cb[s] = cb;
        return 0;
    }
}
