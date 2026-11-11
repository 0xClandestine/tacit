//go:build !icicle

package sp1

// The Groth16 prover for a machine with less memory than the proving key: the arithmetic of gnark's groth16/bn254 Prove
// (the version this module pins), scheduled so the working set stays small. gnark runs its five multi-exponentiations
// at once, and each one scans its bases once per window (about 17 times): with 5.9 GB of mapped bases on an 8 GB machine
// those scans evict one another and the proof waits on the disk. Here
//   - H is computed on a domain without FFT tables (2 GB at this size; each transform builds the little it needs), padding
//     A, B and C one at a time;
//   - the multi-exponentiations run one after another, each on every core, and each walks its bases in blocks small
//     enough to stay in memory across the windows (SP1_GNARK_MSM_BLOCK points; 0 = one block), so a base page is read
//     from the disk about once per proof; the block sums are added up, so the result is the same point;
//   - every vector is dropped as soon as its last product is taken;
//   - on a Mac the four G1 multi-exponentiations run on the GPU, the CPU sorting each block for it (gpu_msm_darwin.go).
// The proof is the same kind of proof (random r and s, as gnark's), and the caller verifies it against the verifying key.

import (
	"fmt"
	"math/big"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"sync"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	curve "github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/hash_to_field"
	groth16_bn254 "github.com/consensys/gnark/backend/groth16/bn254"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
	cs "github.com/consensys/gnark/constraint/bn254"
	"github.com/consensys/gnark/constraint/solver"
	fcs "github.com/consensys/gnark/frontend/cs"
	"github.com/consensys/gnark/logger"
)

// msmBlock is how many bases one block of a multi-exponentiation scans: 4M points (256 MB of G1, 512 MB of G2) by default.
func msmBlock() int {
	if v, err := strconv.Atoi(os.Getenv("SP1_GNARK_MSM_BLOCK")); err == nil && v >= 0 {
		return v
	}
	return 1 << 22
}

// leanTimes prints each phase's time when SP1_GNARK_TIMES is set.
func leanPhase(name string, t time.Time) {
	if os.Getenv("SP1_GNARK_TIMES") != "" {
		fmt.Printf("groth16 %s %.1f s\n", name, time.Since(t).Seconds())
	}
}

// check, when given, verifies a proof (prove_groth16.go: against groth16_vk.bin); it is run on proofs the GPU helped make.
func proveLean(r1cs *cs.R1CS, pk *groth16_bn254.ProvingKey, fullWitness witness.Witness, check func(*groth16_bn254.Proof) error) (*groth16_bn254.Proof, error) {
	log := logger.Logger().With().Str("curve", "bn254").Str("acceleration", "lean").Int("nbConstraints", r1cs.GetNbConstraints()).Str("backend", "groth16").Logger()
	commitmentInfo := r1cs.CommitmentInfo.(constraint.Groth16Commitments)
	proof := &groth16_bn254.Proof{Commitments: make([]curve.G1Affine, len(commitmentInfo))}
	htf := hash_to_field.New([]byte(constraint.CommitmentDst))
	privateCommittedValues := make([][]fr.Element, len(commitmentInfo))

	// The commitment hint, as gnark's prover overrides it.
	bsb22ID := solver.GetHintID(fcs.Bsb22CommitmentComputePlaceholder)
	solverOpts := []solver.Option{solver.OverrideHint(bsb22ID, func(_ *big.Int, in []*big.Int, out []*big.Int) error {
		i := int(in[0].Int64())
		in = in[1:]
		privateCommittedValues[i] = make([]fr.Element, len(commitmentInfo[i].PrivateCommitted))
		hashed := in[:len(commitmentInfo[i].PublicAndCommitmentCommitted)]
		committed := in[+len(hashed):]
		for j, inJ := range committed {
			privateCommittedValues[i][j].SetBigInt(inJ)
		}
		var err error
		if proof.Commitments[i], err = pk.CommitmentKeys[i].Commit(privateCommittedValues[i]); err != nil {
			return err
		}
		htf.Write(constraint.SerializeCommitment(proof.Commitments[i].Marshal(), hashed, (fr.Bits-1)/8+1))
		hashBts := htf.Sum(nil)
		htf.Reset()
		nbBuf := fr.Bytes
		if htf.Size() < fr.Bytes {
			nbBuf = htf.Size()
		}
		var res fr.Element
		res.SetBytes(hashBts[:nbBuf])
		res.BigInt(out[0])
		return nil
	})}

	t := time.Now()
	_solution, err := r1cs.Solve(fullWitness, solverOpts...)
	if err != nil {
		return nil, err
	}
	solution := _solution.(*cs.R1CSSolution)
	wireValues := []fr.Element(solution.W)
	leanPhase("solve", t)

	start := time.Now()
	poks := make([]curve.G1Affine, len(pk.CommitmentKeys))
	for i := range pk.CommitmentKeys {
		if poks[i], err = pk.CommitmentKeys[i].ProveKnowledge(privateCommittedValues[i]); err != nil {
			return nil, err
		}
	}
	commitmentsSerialized := make([]byte, fr.Bytes*len(commitmentInfo))
	for i := range commitmentInfo {
		copy(commitmentsSerialized[fr.Bytes*i:], wireValues[commitmentInfo[i].CommitmentIndex].Marshal())
	}
	challenge, err := fr.Hash(commitmentsSerialized, []byte("G16-BSB22"), 1)
	if err != nil {
		return nil, err
	}
	if _, err = proof.CommitmentPok.Fold(poks, challenge[0], ecc.MultiExpConfig{NbTasks: 1}); err != nil {
		return nil, err
	}

	t = time.Now()
	h := computeHLean(solution, &pk.Domain)
	leanPhase("H", t)

	// r, s and the deltas, as gnark draws them.
	var r, s big.Int
	var _r, _s, _kr fr.Element
	if _, err := _r.SetRandom(); err != nil {
		return nil, err
	}
	if _, err := _s.SetRandom(); err != nil {
		return nil, err
	}
	_kr.Mul(&_r, &_s).Neg(&_kr)
	_r.BigInt(&r)
	_s.BigInt(&s)
	deltas := curve.BatchScalarMultiplicationG1(&pk.G1.Delta, []fr.Element{_r, _s, _kr})

	block := msmBlock()

	// The scalars: the wire values without those whose A or B base is the point at infinity, the private wires outside the
	// commitments for K, and H for Z. Then the wire values are dropped.
	wA := dropInfinity(wireValues, pk.InfinityA, int(pk.NbInfinityA))
	wB := dropInfinity(wireValues, pk.InfinityB, int(pk.NbInfinityB))
	toRemove := commitmentInfo.GetPrivateCommitted()
	toRemove = append(toRemove, commitmentInfo.CommitmentIndexes())
	nbPublic := r1cs.GetNbPublicVariables()
	wK := filterHeap(wireValues[nbPublic:], nbPublic, concatAll(toRemove...))
	wireValues, solution = nil, nil
	sizeH := int(pk.Domain.Cardinality - 1)

	// The four G1 sums, with one multi-exponentiation function or another.
	var ar, bs1, krs, krs2 curve.G1Jac
	g1 := func(where string, msm func(dst *curve.G1Jac, bases []curve.G1Affine, scalars []fr.Element) error) error {
		for _, m := range []struct {
			name  string
			dst   *curve.G1Jac
			bases []curve.G1Affine
			sc    []fr.Element
		}{{"A", &ar, pk.G1.A, wA}, {"B1", &bs1, pk.G1.B, wB}, {"K", &krs, pk.G1.K, wK}, {"Z", &krs2, pk.G1.Z, h[:sizeH]}} {
			t := time.Now()
			if err := msm(m.dst, m.bases, m.sc); err != nil {
				return err
			}
			leanPhase(fmt.Sprintf("msm %s %d (%s)", m.name, len(m.bases), where), t)
		}
		return nil
	}
	cpuG1 := func(dst *curve.G1Jac, bases []curve.G1Affine, scalars []fr.Element) error {
		return msmG1(dst, bases, scalars, block)
	}
	// The G2 sum can go to the GPU too, but it needs its own canary: the G2 kernels are a separate pair, over F2.
	g2 := func(where string, f func(*curve.G2Jac, []curve.G2Affine, []fr.Element) error) (curve.G2Jac, error) {
		t := time.Now()
		var bs curve.G2Jac
		err := f(&bs, pk.G2.B, wB)
		leanPhase(fmt.Sprintf("msm B2 %d (%s)", len(pk.G2.B), where), t)
		return bs, err
	}
	cpuG2 := func(dst *curve.G2Jac, bases []curve.G2Affine, scalars []fr.Element) error {
		return msmG2(dst, bases, scalars, block)
	}

	// On a Mac the G1 sums run on the GPU, then the G2 one on the CPU: the GPU's sums need the CPU's cores to sort each block
	// for them, so the two do not run side by side. Elsewhere, or with the GPU off, all on the CPU.
	useGPU := gpuMsmReady() && gpuMsmCanary(pk.G1.A)
	var bs curve.G2Jac
	var g1err, g2err error
	if useGPU {
		g1err = g1("GPU", gpuMsmG1)
		gpuMsmReleaseG1()
		if g1err != nil {
			gpuMsmOff(g1err.Error())
			useGPU = false
			g1err = g1("CPU", cpuG1)
		}
	} else {
		g1err = g1("CPU", cpuG1)
	}
	if g1err == nil {
		if useGPU && gpuMsmCanaryG2(pk.G2.B) {
			bs, g2err = g2("GPU", gpuMsmG2)
			gpuMsmRelease()
			if g2err != nil {
				gpuMsmOff(g2err.Error())
				bs, g2err = g2("CPU", cpuG2)
			}
		} else {
			bs, g2err = g2("CPU", cpuG2)
		}
	}
	if g1err != nil {
		return nil, g1err
	}
	if g2err != nil {
		return nil, g2err
	}
	wA, wB, wK, h = nil, nil, nil, nil

	ar.AddMixed(&pk.G1.Alpha)
	ar.AddMixed(&deltas[0])
	proof.Ar.FromJacobian(&ar)
	bs1.AddMixed(&pk.G1.Beta)
	bs1.AddMixed(&deltas[1])
	var deltaS curve.G2Jac
	deltaS.FromAffine(&pk.G2.Delta)
	deltaS.ScalarMultiplication(&deltaS, &s)
	bs.AddAssign(&deltaS)
	bs.AddMixed(&pk.G2.Beta)
	proof.Bs.FromJacobian(&bs)
	// [C]₁ = K sum + Z sum − rs·δ + s·A + r·B₁
	var p1 curve.G1Jac
	krs.AddMixed(&deltas[2])
	krs.AddAssign(&krs2)
	p1.ScalarMultiplication(&ar, &s)
	krs.AddAssign(&p1)
	p1.ScalarMultiplication(&bs1, &r)
	krs.AddAssign(&p1)
	proof.Krs.FromJacobian(&krs)

	// A proof the GPU took part in is verified before it leaves; one that does not verify turns the GPU off and is made again.
	if useGPU && check != nil {
		if err := check(proof); err != nil {
			gpuMsmOff("a proof made with it did not verify")
			return proveLean(r1cs, pk, fullWitness, check)
		}
	}

	log.Debug().Dur("took", time.Since(start)).Msg("prover done")
	return proof, nil
}

// computeHLean is gnark's computeH (h = (A·B − C)/Z on the coset, back to coefficients) on the key's domain, which has no
// FFT tables (lean_load.go): each transform builds what it needs (a 32 MB twiddle table, and the 0.5 GB coset table for
// a coset FFT) and drops it, where gnark keeps 2 GB of tables. A, B and C are padded and transformed in turn.
func computeHLean(sol *cs.R1CSSolution, domain *fft.Domain) []fr.Element {
	card := domain.Cardinality
	toCoset := func(v []fr.Element) []fr.Element {
		p := make([]fr.Element, card)
		copy(p, v)
		domain.FFTInverse(p, fft.DIF)
		domain.FFT(p, fft.DIT, fft.OnCoset())
		return p
	}
	a := toCoset(sol.A)
	sol.A = nil
	b := toCoset(sol.B)
	sol.B = nil
	parallelize(len(a), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			a[i].Mul(&a[i], &b[i])
		}
	})
	b = nil
	c := toCoset(sol.C)
	sol.C = nil
	var den, one fr.Element
	one.SetOne()
	den.Exp(domain.FrMultiplicativeGen, big.NewInt(int64(card)))
	den.Sub(&den, &one).Inverse(&den)
	parallelize(len(a), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			a[i].Sub(&a[i], &c[i]).Mul(&a[i], &den)
		}
	})
	c = nil
	domain.FFTInverse(a, fft.DIF, fft.OnCoset())
	runtime.GC()
	debug.FreeOSMemory()
	return a
}

// msmG1 sets dst to Σ scalars[i]·bases[i], in blocks of `block` bases (0: one block), each on every core.
func msmG1(dst *curve.G1Jac, bases []curve.G1Affine, scalars []fr.Element, block int) error {
	if len(bases) != len(scalars) {
		return fmt.Errorf("msm: %d bases for %d scalars", len(bases), len(scalars))
	}
	cfg := ecc.MultiExpConfig{NbTasks: runtime.NumCPU()}
	if block <= 0 || len(bases) <= block {
		_, err := dst.MultiExp(bases, scalars, cfg)
		return err
	}
	var acc, part curve.G1Jac
	for i := 0; i < len(bases); i += block {
		j := min(i+block, len(bases))
		if _, err := part.MultiExp(bases[i:j], scalars[i:j], cfg); err != nil {
			return err
		}
		acc.AddAssign(&part)
	}
	dst.Set(&acc)
	return nil
}

// msmG2 is msmG1 on G2, with twice the tasks, as gnark runs its G2 multi-exponentiation.
func msmG2(dst *curve.G2Jac, bases []curve.G2Affine, scalars []fr.Element, block int) error {
	if len(bases) != len(scalars) {
		return fmt.Errorf("msm: %d bases for %d scalars", len(bases), len(scalars))
	}
	cfg := ecc.MultiExpConfig{NbTasks: 2 * runtime.NumCPU()}
	if block <= 0 || len(bases) <= block {
		_, err := dst.MultiExp(bases, scalars, cfg)
		return err
	}
	var acc, part curve.G2Jac
	for i := 0; i < len(bases); i += block {
		j := min(i+block, len(bases))
		if _, err := part.MultiExp(bases[i:j], scalars[i:j], cfg); err != nil {
			return err
		}
		acc.AddAssign(&part)
	}
	dst.Set(&acc)
	return nil
}

// dropInfinity returns the wire values without those whose base is the point at infinity, as gnark filters them.
func dropInfinity(w []fr.Element, infinity []bool, nbInfinity int) []fr.Element {
	out := make([]fr.Element, len(w)-nbInfinity)
	for i, j := 0, 0; j < len(out); i++ {
		if infinity[i] {
			continue
		}
		out[j] = w[i]
		j++
	}
	return out
}

// filterHeap, concatAll and parallelize: gnark's internal helpers of the same names (groth16/bn254, groth16/internal,
// internal/utils), which another module cannot import.

func filterHeap(slice []fr.Element, sliceFirstIndex int, toRemove []int) (r []fr.Element) {
	if len(toRemove) == 0 {
		return slice
	}
	heap := intHeap(toRemove)
	heap.heapify()
	r = make([]fr.Element, 0, len(slice))
	for i := 0; i < len(slice); i++ {
		if len(heap) > 0 && i+sliceFirstIndex == heap[0] {
			for len(heap) > 0 && i+sliceFirstIndex == heap[0] {
				heap.pop()
			}
			continue
		}
		r = append(r, slice[i])
	}
	return
}

func concatAll(slices ...[]int) []int {
	n := 0
	for _, s := range slices {
		n += len(s)
	}
	out := make([]int, 0, n)
	for _, s := range slices {
		out = append(out, s...)
	}
	return out
}

type intHeap []int

func (h *intHeap) heapify() {
	n := len(*h)
	for i := n/2 - 1; i >= 0; i-- {
		h.down(i, n)
	}
}

func (h *intHeap) pop() {
	n := len(*h) - 1
	(*h)[0], (*h)[n] = (*h)[n], (*h)[0]
	h.down(0, n)
	*h = (*h)[:n]
}

func (h *intHeap) down(i0, n int) {
	i := i0
	for {
		j1 := 2*i + 1
		if j1 >= n || j1 < 0 {
			break
		}
		j := j1
		if j2 := j1 + 1; j2 < n && (*h)[j2] < (*h)[j1] {
			j = j2
		}
		if !((*h)[j] < (*h)[i]) {
			break
		}
		(*h)[i], (*h)[j] = (*h)[j], (*h)[i]
		i = j
	}
}

func parallelize(n int, work func(lo, hi int)) {
	tasks := runtime.NumCPU()
	if tasks > n {
		tasks = 1
	}
	per := (n + tasks - 1) / tasks
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += per {
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			work(lo, hi)
		}(lo, min(lo+per, n))
	}
	wg.Wait()
}
