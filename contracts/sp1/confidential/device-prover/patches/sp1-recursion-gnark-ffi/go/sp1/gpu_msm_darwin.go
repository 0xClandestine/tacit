//go:build darwin && !icicle

package sp1

// The Groth16 step's G1 multi-exponentiations on the Mac's GPU (Metal): Pippenger with 16 windows of 16 bits. Per block of
// bases the CPU writes the canonical scalars' digits as, for each window, the point indices sorted into their buckets, while
// the GPU adds the block before's (gpu_msm.metal; two sets of buffers), the bucket sums carrying from block to block; at the
// end the CPU folds the buckets (Σ b·B_b by running sums) and the windows. The CPU's part needs its cores: the G2 sum runs
// after, not beside it (prove_lean.go). The result is the same point gnark's
// MultiExp gives. Before the GPU is first used its sum over 4096 of the key's own bases is compared with gnark's, and any
// mismatch turns it off; a proof made with it is also verified before it is returned (prove_lean.go).
// Off unless SP1_GNARK_GPU=1: on a machine that was busy the whole step ran slower this way than on the lean CPU path (440 s against
// 259 s), so it stays a choice until it is timed on an idle one.

/*
#cgo CFLAGS: -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation
#include <stdint.h>
#include <stdlib.h>
int tacit_msm_init(const char *msl, uint32_t cap, uint32_t nb, uint32_t windows, char *name, size_t name_len);
int tacit_msm_alloc(void);
void tacit_msm_release(void);
void *tacit_msm_pts(int s);
void *tacit_msm_order(int s);
void *tacit_msm_start(int s);
void *tacit_msm_out(void);
void *tacit_msm_sbase(int s);
void *tacit_msm_scount(int s);
uint32_t tacit_msm_winslots(void);
int tacit_msm_wait(int s);
int tacit_msm_send(int s, uint32_t n, uint32_t first);
*/
import "C"

import (
	_ "embed"
	"fmt"
	"os"
	"sync"
	"unsafe"

	"github.com/consensys/gnark-crypto/ecc"
	curve "github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

//go:embed gpu_msm.metal
var gpuMsmSource string

const (
	gpuWindowBits = 16
	gpuWindows    = 16 // 16 windows of 16 bits hold a canonical BN254 scalar
	gpuBuckets    = 1<<gpuWindowBits - 1
	gpuBlock      = 1 << 21 // bases per block: 128 MB of points and 128 MB of order, twice over
	msmPerThread  = 8192    // a bucket gets ceil(count/this) threads, so no one thread walks half a block
)

var (
	gpuMu    sync.Mutex
	gpuOnce  sync.Once
	gpuName  string
	gpuState = 0 // 0 unknown, 1 on, -1 off
)

// gpuMsmReady says whether the GPU path may be used, setting it up once.
func gpuMsmReady() bool {
	if os.Getenv("SP1_GNARK_GPU") != "1" {
		return false
	}
	gpuOnce.Do(func() {
		src := C.CString(gpuMsmSource)
		defer C.free(unsafe.Pointer(src))
		var name [128]C.char
		if rc := C.tacit_msm_init(src, C.uint32_t(gpuBlock), C.uint32_t(gpuBuckets), C.uint32_t(gpuWindows), &name[0], 128); rc != 0 {
			gpuState = -1
			return
		}
		gpuName = C.GoString(&name[0])
		gpuState = 1
	})
	gpuMu.Lock()
	defer gpuMu.Unlock()
	return gpuState == 1
}

// gpuMsmRelease frees the GPU's block buffers until the next multi-exponentiation needs them.
func gpuMsmRelease() {
	gpuMu.Lock()
	C.tacit_msm_release()
	gpuMu.Unlock()
}

// gpuMsmOff turns the GPU path off for the rest of the process.
func gpuMsmOff(why string) {
	gpuMu.Lock()
	gpuState = -1
	gpuMu.Unlock()
	fmt.Printf("groth16: G1 multi-exponentiation back on the CPU: %s\n", why)
}

// gpuMsmCanary compares the GPU's sum over the first 4096 of `bases` with gnark's, once per process.
var canaryOnce sync.Once
var canaryOK bool

func gpuMsmCanary(bases []curve.G1Affine) bool {
	canaryOnce.Do(func() {
		n := min(4096, len(bases))
		s := make([]fr.Element, n)
		for i := range s {
			s[i].SetRandom()
		}
		var want, got curve.G1Jac
		if _, err := want.MultiExp(bases[:n], s, ecc.MultiExpConfig{}); err != nil {
			return
		}
		err := gpuMsmG1(&got, bases[:n], s)
		gpuMsmRelease()
		canaryOK = err == nil && got.Equal(&want)
	})
	if !canaryOK {
		gpuMsmOff("its sum over the key's bases differs from the CPU's")
	}
	return canaryOK
}

// gpuMsmG1 sets dst to Σ scalars[i]·bases[i] on the GPU.
func gpuMsmG1(dst *curve.G1Jac, bases []curve.G1Affine, scalars []fr.Element) error {
	if len(bases) != len(scalars) {
		return fmt.Errorf("msm: %d bases for %d scalars", len(bases), len(scalars))
	}
	if len(bases) == 0 {
		dst.Set(&curve.G1Jac{})
		dst.X.SetOne()
		dst.Y.SetOne()
		return nil
	}
	gpuMu.Lock()
	defer gpuMu.Unlock()
	if gpuState != 1 {
		return fmt.Errorf("the GPU path is off")
	}
	if rc := C.tacit_msm_alloc(); rc != 0 {
		return fmt.Errorf("the GPU's buffers could not be made (%d)", rc)
	}
	canon := make([][4]uint64, min(gpuBlock, len(bases)))
	// Set k%2 takes block k; before it is written again the GPU must be done with the block it last held.
	k := 0
	for lo := 0; lo < len(bases); lo, k = lo+gpuBlock, k+1 {
		set := C.int(k % 2)
		if rc := C.tacit_msm_wait(set); rc != 0 {
			return fmt.Errorf("the GPU's bucket sums failed (%d)", rc)
		}
		ptsBuf := unsafe.Slice((*curve.G1Affine)(C.tacit_msm_pts(set)), gpuBlock)
		order := unsafe.Slice((*uint32)(C.tacit_msm_order(set)), gpuBlock*gpuWindows)
		start := unsafe.Slice((*uint32)(C.tacit_msm_start(set)), (gpuBuckets+1)*gpuWindows)
		nb := gpuBuckets
		sbase := unsafe.Slice((*uint32)(C.tacit_msm_sbase(set)), nb*gpuWindows)
		scount := unsafe.Slice((*uint32)(C.tacit_msm_scount(set)), nb*gpuWindows)
		hi := min(lo+gpuBlock, len(bases))
		n := hi - lo
		copy(ptsBuf[:n], bases[lo:hi])
		parallelize(n, func(a, b int) {
			for i := a; i < b; i++ {
				canon[i] = scalars[lo+i].Bits()
			}
		})
		// Per window, the indices of the block's points sorted by digit (a counting sort); digit 0 adds nothing.
		var wg sync.WaitGroup
		for w := 0; w < gpuWindows; w++ {
			wg.Add(1)
			go func(w int) {
				defer wg.Done()
				limb, sh := w/4, uint(w%4)*gpuWindowBits
				cnt := make([]uint32, gpuBuckets+1)
				for i := 0; i < n; i++ {
					cnt[uint32(canon[i][limb]>>sh)&gpuBuckets]++
				}
				st := start[w*(gpuBuckets+1) : (w+1)*(gpuBuckets+1)]
				acc := uint32(0)
				for b := 0; b < gpuBuckets; b++ {
					st[b] = acc
					acc += cnt[b+1]
				}
				st[gpuBuckets] = acc
				// how many threads this window's bucket gets, and where its slots start
				slot := uint32(0)
				for b := 0; b < gpuBuckets; b++ {
					need := (cnt[b+1] + msmPerThread - 1) / msmPerThread
					if need == 0 {
						need = 1 // keep every bucket's slots non-empty so the fold has something to read
					}
					sbase[w*nb+b] = slot
					scount[w*nb+b] = need
					slot += need
				}
				pos := cnt[:gpuBuckets]
				copy(pos, st[:gpuBuckets])
				ord := order[w*n : (w+1)*n]
				for i := 0; i < n; i++ {
					if d := uint32(canon[i][limb]>>sh) & gpuBuckets; d != 0 {
						ord[pos[d-1]] = uint32(i)
						pos[d-1]++
					}
				}
			}(w)
		}
		wg.Wait()
		first := C.uint32_t(0)
		if lo == 0 {
			first = 1
		}
		if rc := C.tacit_msm_send(set, C.uint32_t(n), first); rc != 0 {
			return fmt.Errorf("the GPU's bucket sums failed (%d)", rc)
		}
	}
	for set := C.int(0); set < 2; set++ {
		if rc := C.tacit_msm_wait(set); rc != 0 {
			return fmt.Errorf("the GPU's bucket sums failed (%d)", rc)
		}
	}
	// Each window's Σ b·B_b by running sums (windows in parallel), then the windows by doubling.
	out := unsafe.Slice((*curve.G1Jac)(C.tacit_msm_out()), gpuBuckets*gpuWindows)
	wsum := make([]curve.G1Jac, gpuWindows)
	var wg sync.WaitGroup
	for w := 0; w < gpuWindows; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			var run, tot curve.G1Jac
			B := out[w*gpuBuckets : (w+1)*gpuBuckets]
			for b := gpuBuckets - 1; b >= 0; b-- {
				run.AddAssign(&B[b])
				tot.AddAssign(&run)
			}
			wsum[w] = tot
		}(w)
	}
	wg.Wait()
	res := wsum[gpuWindows-1]
	for w := gpuWindows - 2; w >= 0; w-- {
		for k := 0; k < gpuWindowBits; k++ {
			res.DoubleAssign()
		}
		res.AddAssign(&wsum[w])
	}
	dst.Set(&res)
	return nil
}
