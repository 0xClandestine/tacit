#include <metal_stdlib>
using namespace metal;
constant uint64_t RC[24]={0x1,0x8082,0x800000000000808aULL,0x8000000080008000ULL,0x808b,0x80000001,0x8000000080008081ULL,0x8000000000008009ULL,0x8a,0x88,0x80008009,0x8000000a,0x8000808b,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,0x8000000000008002ULL,0x8000000000000080ULL,0x800a,0x800000008000000aULL,0x8000000080008081ULL,0x8000000000008080ULL,0x80000001,0x8000000080008008ULL};
constant int RHO[24]={1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44};
constant int PI[24]={10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1};
static inline uint64_t rotl(uint64_t x,int n){return (x<<n)|(x>>(64-n));}

static inline void kf(thread uint64_t st[25]){for(int r=0;r<24;r++){uint64_t bc[5];
 for(int i=0;i<5;i++)bc[i]=st[i]^st[i+5]^st[i+10]^st[i+15]^st[i+20];
 for(int i=0;i<5;i++){uint64_t t=bc[(i+4)%5]^rotl(bc[(i+1)%5],1);for(int j=0;j<25;j+=5)st[j+i]^=t;}
 uint64_t t=st[1];for(int i=0;i<24;i++){int j=PI[i];uint64_t tmp=st[j];st[j]=rotl(t,RHO[i]);t=tmp;}
 for(int j=0;j<25;j+=5){for(int i=0;i<5;i++)bc[i]=st[j+i];for(int i=0;i<5;i++)st[j+i]^=(~bc[(i+1)%5])&bc[(i+2)%5];}
 st[0]^=RC[r];}}
// one-shot keccak256 over n<=136 bytes held in a scratch thread buffer
static inline void keccak256(thread uint64_t st[25], thread uchar *m, int n){
  int rem = n; 
  thread uchar last[136];
  for(int i=0;i<136;i++) last[i]=0;
  for(int i=0;i<rem;i++) last[i]=m[i];
  last[rem]^=0x01; last[135]^=0x80;
  for(int i=0;i<17;i++){uint64_t v=0;for(int b=7;b>=0;b--)v=(v<<8)|(uint64_t)last[i*8+b];st[i]^=v;}
  kf(st);
}

// Layout, all prebuilt by the host except salt[24..32):
//   G  : uchar[64]  = 00*12 || DEPLOYER(20) || salt(32)   (salt[24..32) = counter, patched here)
//   P  : uchar[64]  = 0xff || CREATEX(20)                 (guarded(32) || proxy_hash(32) appended here)
static inline uint count_bits(thread const uchar*a){ uint bits=0;
  for(int i=12;i<32;i++){ uchar c=a[i]; if(c==0){bits+=8;continue;} bits+=clz((uint)c)-24; break; } return bits; }

kernel void grind(device const uchar *G [[buffer(0)]],
                  device const uchar *CREATEXB [[buffer(1)]],
                  device const uchar *PROXYH [[buffer(2)]],
                  constant uint64_t &base [[buffer(3)]],
                  constant uint &target [[buffer(4)]],
                  device atomic_uint *found [[buffer(5)]],
                  device uint *foundNonce [[buffer(6)]],
                  uint gid [[thread_position_in_grid]]) {
  uint64_t n = base + gid;

  // block 1: guarded = keccak(00*12 || DEPLOYER || salt)   (76 bytes)
  thread uchar m1[136]; for(int i=0;i<136;i++) m1[i]=0;
  for(int i=0;i<64;i++) m1[i]=G[i];
  for(int b=0;b<8;b++) m1[56+b]=(uchar)(n>>(8*(7-b)));   // salt[24..32) sits at 12+20+24=56
  uint64_t st[25]; for(int i=0;i<25;i++) st[i]=0;
  keccak256(st, m1, 64);
  thread uchar guarded[32];
  for(int i=0;i<4;i++){uint64_t v=st[i];for(int b=0;b<8;b++)guarded[i*8+b]=(uchar)(v>>(8*b));}

  // block 2: proxy = keccak(0xff || CREATEX || guarded || proxy_hash)  (85 bytes)
  thread uchar m2[136]; for(int i=0;i<136;i++) m2[i]=0;
  m2[0]=0xff; for(int i=0;i<20;i++) m2[1+i]=CREATEXB[i];
  for(int i=0;i<32;i++) m2[21+i]=guarded[i];
  for(int i=0;i<32;i++) m2[53+i]=PROXYH[i];
  for(int i=0;i<25;i++) st[i]=0;
  keccak256(st, m2, 85);
  thread uchar proxy[32];
  for(int i=0;i<4;i++){uint64_t v=st[i];for(int b=0;b<8;b++)proxy[i*8+b]=(uchar)(v>>(8*b));}

  // block 3: addr = keccak(0xd6 0x94 || proxy[12..32) || 0x01)  (23 bytes)
  thread uchar m3[136]; for(int i=0;i<136;i++) m3[i]=0;
  m3[0]=0xd6; m3[1]=0x94;
  for(int i=0;i<20;i++) m3[2+i]=proxy[12+i];
  m3[22]=0x01;
  for(int i=0;i<25;i++) st[i]=0;
  keccak256(st, m3, 23);
  thread uchar a[32];
  for(int i=0;i<4;i++){uint64_t v=st[i];for(int b=0;b<8;b++)a[i*8+b]=(uchar)(v>>(8*b));}

  if (count_bits(a) >= target) {
    uint slot = atomic_fetch_add_explicit(found, 1u, memory_order_relaxed);
    if (slot < 256) foundNonce[slot] = (uint)n;
  }
}

// Test only: write the 20-byte address for gid, so the probe can compare with salt.rs address().
kernel void dump(device const uchar *G [[buffer(0)]],
                 device const uchar *CREATEXB [[buffer(1)]],
                 device const uchar *PROXYH [[buffer(2)]],
                 constant uint64_t &base [[buffer(3)]],
                 device uchar *outAddr [[buffer(7)]],
                 uint gid [[thread_position_in_grid]]) {
  uint64_t n = base + gid;
  thread uchar m1[136]; for(int i=0;i<136;i++) m1[i]=0;
  for(int i=0;i<64;i++) m1[i]=G[i];
  for(int b=0;b<8;b++) m1[56+b]=(uchar)(n>>(8*(7-b)));
  uint64_t st[25]; for(int i=0;i<25;i++) st[i]=0;
  keccak256(st,m1,64);
  thread uchar guarded[32];
  for(int i=0;i<4;i++){uint64_t v=st[i];for(int b=0;b<8;b++)guarded[i*8+b]=(uchar)(v>>(8*b));}
  thread uchar m2[136]; for(int i=0;i<136;i++) m2[i]=0;
  m2[0]=0xff; for(int i=0;i<20;i++) m2[1+i]=CREATEXB[i];
  for(int i=0;i<32;i++) m2[21+i]=guarded[i];
  for(int i=0;i<32;i++) m2[53+i]=PROXYH[i];
  for(int i=0;i<25;i++) st[i]=0;
  keccak256(st,m2,85);
  thread uchar proxy[32];
  for(int i=0;i<4;i++){uint64_t v=st[i];for(int b=0;b<8;b++)proxy[i*8+b]=(uchar)(v>>(8*b));}
  thread uchar m3[136]; for(int i=0;i<136;i++) m3[i]=0;
  m3[0]=0xd6; m3[1]=0x94;
  for(int i=0;i<20;i++) m3[2+i]=proxy[12+i];
  m3[22]=0x01;
  for(int i=0;i<25;i++) st[i]=0;
  keccak256(st,m3,23);
  thread uchar a[32];
  for(int i=0;i<4;i++){uint64_t v=st[i];for(int b=0;b<8;b++)a[i*8+b]=(uchar)(v>>(8*b));}
  for(int i=0;i<20;i++) outAddr[gid*20+i]=a[12+i];
}
