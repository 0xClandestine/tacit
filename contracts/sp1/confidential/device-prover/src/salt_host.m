// The Metal host for the salt grind: compiles salt.metal at first use and grinds a batch of counters.
// Kept next to salt.rs so the whole fast path is one file of MSL plus this. Any failure returns -1 and the
// caller keeps the CPU threads it already has.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "salt_msl.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
    id<MTLComputePipelineState> pso;
    id<MTLBuffer> bG, bC, bP, bBase, bTgt, bFound, bNonce;
    int ok;
} SaltGPU;

static pthread_key_t g_key;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static void make_key(void) { pthread_key_create(&g_key, NULL); }

// ARC owns the Objective-C objects; only the C struct is freed here.
static void free_gpu(void *p) { free(p); }

// One device, one queue, one pipeline per thread: a shared command queue serialises dispatches across threads.
static SaltGPU *gpu(void) {
    pthread_once(&g_once, make_key);
    SaltGPU *g = (SaltGPU *)pthread_getspecific(g_key);
    if (g) return g;

    g = (SaltGPU *)calloc(1, sizeof(SaltGPU));
    if (!g) return NULL;
    @autoreleasepool {
        g->dev = MTLCreateSystemDefaultDevice();
        if (!g->dev) { free(g); return NULL; }
        g->q = [g->dev newCommandQueue];
        if (!g->q) { free_gpu(g); pthread_setspecific(g_key, NULL); return NULL; }
        NSError *err = nil;
        id<MTLLibrary> lib = [g->dev newLibraryWithSource:[NSString stringWithUTF8String:MSL] options:nil error:&err];
        if (!lib) {
            fprintf(stderr, "tacit-prover: salt MSL did not compile: %s\n",
                    err ? err.localizedDescription.UTF8String : "no message");
            free_gpu(g); return NULL;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"grind"];
        if (!fn) { fprintf(stderr, "tacit-prover: salt MSL has no grind\n"); free_gpu(g); return NULL; }
        g->pso = [g->dev newComputePipelineStateWithFunction:fn error:&err];
        if (!g->pso) {
            fprintf(stderr, "tacit-prover: salt pipeline: %s\n", err ? err.localizedDescription.UTF8String : "no message");
            free_gpu(g); return NULL;
        }
        g->bG    = [g->dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
        g->bC    = [g->dev newBufferWithLength:20 options:MTLResourceStorageModeShared];
        g->bP    = [g->dev newBufferWithLength:32 options:MTLResourceStorageModeShared];
        g->bBase = [g->dev newBufferWithLength:8  options:MTLResourceStorageModeShared];
        g->bTgt  = [g->dev newBufferWithLength:4  options:MTLResourceStorageModeShared];
        g->bFound= [g->dev newBufferWithLength:4  options:MTLResourceStorageModeShared];
        g->bNonce= [g->dev newBufferWithLength:256*8 options:MTLResourceStorageModeShared];
        if (!g->bG || !g->bC || !g->bP || !g->bBase || !g->bTgt || !g->bFound || !g->bNonce) {
            free_gpu(g); return NULL;
        }
        g->ok = 1;
    }
    pthread_setspecific(g_key, g);
    return g;
}

// Test only: the 20-byte address for each of `n` consecutive nonces from `base`.
int tacit_salt_gpu_dump(const unsigned char *shape, uint64_t base, unsigned int n, unsigned char *out) {
    SaltGPU *g = gpu();
    if (!g || !g->ok) return -1;
    int rc = -1;
    @autoreleasepool {
        memcpy([g->bG contents], shape, 64);
        memcpy([g->bC contents], shape + 64, 20);
        memcpy([g->bP contents], shape + 84, 32);
        *(uint64_t *)[g->bBase contents] = base;

        NSError *err = nil;
        id<MTLLibrary> lib = [g->dev newLibraryWithSource:[NSString stringWithUTF8String:MSL] options:nil error:&err];
        if (!lib) return -1;
        id<MTLFunction> fn = [lib newFunctionWithName:@"dump"];
        if (!fn) return -1;
        id<MTLComputePipelineState> pso = [g->dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) return -1;
        id<MTLBuffer> bo = [g->dev newBufferWithLength:20*n options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [g->q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:g->bG offset:0 atIndex:0];
        [e setBuffer:g->bC offset:0 atIndex:1];
        [e setBuffer:g->bP offset:0 atIndex:2];
        [e setBuffer:g->bBase offset:0 atIndex:3];
        [e setBuffer:bo offset:0 atIndex:7];
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(n,1,1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (!cb.error) { memcpy(out, [bo contents], 20*n); rc = 0; }
    }
    return rc;
}

int tacit_salt_gpu_available(void) {
    SaltGPU *g = gpu();
    return g && g->ok ? 1 : 0;
}

// shape: 64 bytes G || 20 CREATEX || 32 proxy_hash. Grinds [base, base+n) where n comes from the grid size
// the caller asks for through `max` (max nonces to record; the grid is BATCH threads).
int tacit_salt_gpu_grind(const unsigned char *shape, uint64_t base, uint32_t target,
                         unsigned long long *out, uint32_t max) {
    static const uint64_t BATCH = 1u << 22;
    SaltGPU *g = gpu();
    if (!g || !g->ok) return -1;

    int n = -1;
    @autoreleasepool {
        memcpy([g->bG contents], shape, 64);
        memcpy([g->bC contents], shape + 64, 20);
        memcpy([g->bP contents], shape + 84, 32);
        *(uint64_t *)[g->bBase contents] = base;
        *(uint32_t *)[g->bTgt contents] = target;
        *(uint32_t *)[g->bFound contents] = 0;

        NSUInteger tg = g->pso.maxTotalThreadsPerThreadgroup;
        if (tg > 256) tg = 256;
        NSUInteger grid = BATCH / tg;

        id<MTLCommandBuffer> cb = [g->q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g->pso];
        [e setBuffer:g->bG offset:0 atIndex:0];
        [e setBuffer:g->bC offset:0 atIndex:1];
        [e setBuffer:g->bP offset:0 atIndex:2];
        [e setBuffer:g->bBase offset:0 atIndex:3];
        [e setBuffer:g->bTgt offset:0 atIndex:4];
        [e setBuffer:g->bFound offset:0 atIndex:5];
        [e setBuffer:g->bNonce offset:0 atIndex:6];
        [e dispatchThreadgroups:MTLSizeMake(grid, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) return -1;

        uint32_t f = *(uint32_t *)[g->bFound contents];
        if (f > 256) f = 256;
        if (f > max) f = max;
        if (f) memcpy(out, [g->bNonce contents], f * sizeof(unsigned long long));
        n = (int)f;
    }
    return n;
}
