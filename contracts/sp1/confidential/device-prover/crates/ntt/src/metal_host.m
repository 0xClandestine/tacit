// The Metal host for crates/ntt: compiles the MSL at first use and runs NTT layers or a bit-reversal over a buffer.
//
// Buffers are page-aligned and handed to the GPU where they lie (unified memory, no copy). One pipeline state per kernel,
// created once; a command queue per thread (see thread_queue below).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

static pthread_mutex_t g_init_lock = PTHREAD_MUTEX_INITIALIZER;
static id<MTLDevice> g_dev = nil;
static id<MTLComputePipelineState> g_layer = nil;
static id<MTLComputePipelineState> g_bitrev = nil;
static id<MTLComputePipelineState> g_chunk = nil;
static char g_name[128];

// A command queue per thread. SP1 proves several stages at once, and a queue is not safe to share without waiting on it;
// one per thread means no thread ever waits for another, and nothing is serialised that need not be. Metal queues are
// cheap, and a thread's queue stays with the thread for as long as it lives.
static pthread_key_t g_queue_key;
static pthread_once_t g_queue_once = PTHREAD_ONCE_INIT;

static void queue_destroy(void *q) {
    id<MTLCommandQueue> cq = (__bridge_transfer id<MTLCommandQueue>)q;
    (void)cq; // ARC releases it here
}

static void queue_key_init(void) {
    pthread_key_create(&g_queue_key, queue_destroy);
}

static id<MTLCommandQueue> thread_queue(void) {
    pthread_once(&g_queue_once, queue_key_init);
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)pthread_getspecific(g_queue_key);
    if (!q) {
        id<MTLDevice> dev = g_dev ? g_dev : MTLCreateSystemDefaultDevice();
        q = [dev newCommandQueue];
        pthread_setspecific(g_queue_key, (__bridge_retained void *)q);
    }
    return q;
}

// One-time init. `msl` is the kernel source (NUL-terminated). Returns 0 on success and writes the device name.
int32_t tacit_ntt_init(const char *msl, char *name, size_t name_len) {
    @autoreleasepool {
        pthread_mutex_lock(&g_init_lock);
        if (g_chunk != nil) {  // already built, e.g. a second call from another thread
            if (name) { strncpy(name, g_name, name_len - 1); name[name_len - 1] = 0; }
            pthread_mutex_unlock(&g_init_lock);
            return 0;
        }
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) { pthread_mutex_unlock(&g_init_lock); return -1; }
        NSError *err = nil;
        id<MTLLibrary> lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:msl] options:nil error:&err];
        if (!lib) {
            fprintf(stderr, "tacit-ntt: MSL did not compile: %s\n", err ? err.localizedDescription.UTF8String : "no message");
            pthread_mutex_unlock(&g_init_lock); return -3;
        }
        id<MTLFunction> fl = [lib newFunctionWithName:@"kb_ntt_layer"];
        id<MTLFunction> fb = [lib newFunctionWithName:@"kb_bitrev"];
        id<MTLFunction> fc = [lib newFunctionWithName:@"kb_ntt_chunk"];
        if (!fl || !fb || !fc) {
            fprintf(stderr, "tacit-ntt: kernel missing (layer=%d bitrev=%d chunk=%d)\n", fl != nil, fb != nil, fc != nil);
            pthread_mutex_unlock(&g_init_lock); return -6;
        }
        g_layer = [g_dev newComputePipelineStateWithFunction:fl error:&err];
        if (!g_layer) {
            fprintf(stderr, "tacit-ntt: layer pipeline: %s\n", err ? err.localizedDescription.UTF8String : "no message");
            pthread_mutex_unlock(&g_init_lock); return -4;
        }
        g_bitrev = [g_dev newComputePipelineStateWithFunction:fb error:&err];
        if (!g_bitrev) {
            fprintf(stderr, "tacit-ntt: bitrev pipeline: %s\n", err ? err.localizedDescription.UTF8String : "no message");
            pthread_mutex_unlock(&g_init_lock); return -5;
        }
        g_chunk = [g_dev newComputePipelineStateWithFunction:fc error:&err];
        if (!g_chunk) {
            fprintf(stderr, "tacit-ntt: chunk pipeline: %s\n", err ? err.localizedDescription.UTF8String : "no message");
            pthread_mutex_unlock(&g_init_lock); return -7;
        }
        const char *n = g_dev.name.UTF8String;
        strncpy(g_name, n ? n : "Apple GPU", name_len - 1);
        if (name) {
            strncpy(name, g_name, name_len - 1);
            name[name_len - 1] = 0;
        }
        pthread_mutex_unlock(&g_init_lock);
        return 0;
    }
}

// The layer's parameters, padded so both sides of the boundary pass the same bytes.
struct LayerParams {
    uint32_t log_h;
    uint32_t layer;
    uint32_t width;
    uint32_t gathered;
    uint32_t pad[4];
};

static id<MTLBuffer> wrap(uint32_t *p, uint64_t bytes) {
    return [g_dev newBufferWithBytesNoCopy:p length:bytes options:MTLResourceStorageModeShared deallocator:nil];
}

// One DIT layer over `h` rows of `width` u32s: buf0 -> buf1. `tw` is the layer's twiddle table.
int32_t tacit_ntt_layer(uint32_t *buf0, uint32_t *buf1, const uint32_t *tw, uint32_t log_h, uint32_t layer,
                        uint32_t width, uint32_t gathered) {
    @autoreleasepool {
        if (!g_layer) return -1;
        uint64_t cells = ((uint64_t)1 << log_h) * width;
        uint64_t tw_n = ((uint64_t)1 << log_h) / 2;
        id<MTLBuffer> b0 = wrap(buf0, cells * 4);
        id<MTLBuffer> b1 = wrap(buf1, cells * 4);
        id<MTLBuffer> bt = wrap((uint32_t *)tw, tw_n * 4);
        if (!b0 || !b1 || !bt) return -2;

        struct LayerParams p = { log_h, layer, width, gathered, { 0, 0, 0, 0 } };
        // One simdgroup per 32 cells of each butterfly, rounded up: the last one is masked off when width % 32 != 0.
        uint64_t threads = (((uint64_t)1 << log_h) / 2) * (((uint64_t)width + 31) / 32) * 32;
        NSUInteger per = 128;
        NSUInteger groups = (NSUInteger)((threads + per - 1) / per);
        (void)threads;

        id<MTLCommandBuffer> cb = [thread_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_layer];
        [enc setBuffer:b0 offset:0 atIndex:0];
        [enc setBuffer:b1 offset:0 atIndex:1];
        [enc setBuffer:bt offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(per, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "tacit-ntt: layer command buffer status %ld\n", (long)cb.status);
            return -4;
        }
        return 0;
    }
}

// A whole half of the network in one dispatch, one threadgroup per 32 columns of each chunk.
struct ChunkParams {
    uint32_t log_h, width, gathered, chunk, first, last, pad[2];
};

int32_t tacit_ntt_half(uint32_t *buf, const uint32_t *tw, uint32_t log_h, uint32_t first, uint32_t last,
                       uint32_t width, uint32_t gathered) {
    @autoreleasepool {
        if (!g_chunk) return -1;
        uint64_t cells = ((uint64_t)1 << log_h) * width;
        uint64_t tw_n = ((uint64_t)1 << log_h) / 2;
        id<MTLBuffer> b = wrap(buf, cells * 4);
        id<MTLBuffer> bt = wrap((uint32_t *)tw, tw_n * 4);
        if (!b || !bt) return -2;
        uint32_t mid = log_h >> 1;
        struct ChunkParams p = {
            log_h, width, gathered,
            (gathered == 0u) ? (1u << mid) : ((1u << log_h) >> mid),
            first, last, { 0, 0 }
        };
        NSUInteger cols = ((NSUInteger)width + 31) / 32;
        NSUInteger groups = (NSUInteger)(((uint64_t)1 << log_h) / p.chunk) * cols;
        id<MTLCommandBuffer> cb = [thread_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_chunk];
        [enc setBuffer:b offset:0 atIndex:0];
        [enc setBuffer:bt offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "tacit-ntt: half command buffer status %ld\n", (long)cb.status);
            return -4;
        }
        return 0;
    }
}

// Layers `first..last` of one half, in place on `buf`, all encoded into one command buffer. The layers are ordered:
// each reads what the previous one wrote, so a memory barrier goes between dispatches instead of a round-trip to the
// CPU.
int32_t tacit_ntt_layers(uint32_t *buf, const uint32_t *tw, uint32_t log_h, uint32_t first, uint32_t last,
                         uint32_t width, uint32_t gathered) {
    @autoreleasepool {
        if (!g_layer) return -1;
        uint64_t cells = ((uint64_t)1 << log_h) * width;
        uint64_t tw_n = ((uint64_t)1 << log_h) / 2;
        id<MTLBuffer> b = wrap(buf, cells * 4);
        id<MTLBuffer> bt = wrap((uint32_t *)tw, tw_n * 4);
        if (!b || !bt) return -2;

        NSUInteger per = 128;
        id<MTLCommandBuffer> cb = [thread_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_layer];
        [enc setBuffer:b offset:0 atIndex:0];
        [enc setBuffer:b offset:0 atIndex:1];
        [enc setBuffer:bt offset:0 atIndex:2];
        for (uint32_t layer = first; layer < last; layer++) {
            struct LayerParams p = { log_h, layer, width, gathered, { 0, 0, 0, 0 } };
            uint64_t threads = (((uint64_t)1 << log_h) / 2) * (((uint64_t)width + 31) / 32) * 32;
            NSUInteger groups = (NSUInteger)((threads + per - 1) / per);
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(per, 1, 1)];
            if (layer + 1 < last) {
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "tacit-ntt: layers command buffer status %ld\n", (long)cb.status); return -4;
        }
        return 0;
    }
}

// Bit-reversal of the row order: `log_h` rows of `width` u32s, from `src` to `dst` (they must differ, the mapping is
// not its own inverse for every row pair at every size).
int32_t tacit_ntt_bitrev(uint32_t *src, uint32_t *dst, uint32_t log_h, uint32_t width) {
    @autoreleasepool {
        if (!g_bitrev) return -1;
        uint64_t cells = ((uint64_t)1 << log_h) * width;
        id<MTLBuffer> b = wrap(src, cells * 4);
        id<MTLBuffer> o = wrap(dst, cells * 4);
        if (!b || !o) return -2;
        // One thread per row: the kernel copies a whole row, so widths that are not a multiple of four are exact.
        NSUInteger rows = (NSUInteger)1 << log_h;
        NSUInteger per = 1;
        NSUInteger groups = rows;
        id<MTLCommandBuffer> cb = [thread_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_bitrev];
        [enc setBuffer:b offset:0 atIndex:0];
        [enc setBuffer:o offset:0 atIndex:1];
        [enc setBytes:&log_h length:sizeof(log_h) atIndex:2];
        [enc setBytes:&width length:sizeof(width) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(per, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return cb.status == MTLCommandBufferStatusCompleted ? 0 : -3;
    }
}
