# MPFR `Sandpile_Model_Seq` with full mu matrix and fpylll bridge

This package rewrites **only the non-logging `Sandpile_Model_Seq`** from your uploaded C++ code. It preserves its *sequential index scheduler, adjacent-swap formulas, and stochastic resampling of precisely three nearby subdiagonal mu values*. The only algorithmic change is a selectable Lovasz / Siegel swap condition. The **complete lower triangular mu matrix** is retained for later DeepLLL work; off-subdiagonal entries are not recomputed or sampled in this model.

No NTL dependency. All simulator arithmetic and sampling are done in native MPFR. The bridge from an fpylll **MPFR** GSO uses Cython's direct access to the C++ `FP_NR<mpfr_t>` values: no Python `float`, `get_d`, or NumPy float64 intermediary.

## Build

Use the Python interpreter in the **same Sage/fpylll environment** as your notebook:

```bash
# Ubuntu: requires the development packages for fplll, mpfr, gmp
# sudo apt install libfplll-dev libmpfr-dev libgmp-dev
python -m pip install 'Cython>=3.0,<3.2' setuptools
cd sandpile_mpfr_bridge
python setup.py build_ext --inplace
```

If your Sage kernel is at a custom path, run its Python instead of a system Python. Linker/include paths may need `CPATH`, `LIBRARY_PATH`, and `LD_LIBRARY_PATH` when fplll is installed outside `/usr`.

**Validation status:** code is written against the published fpylll 0.6.x Cython declarations, but this container does not have fpylll/fplll/MPFR headers. Therefore it could not be compiled here; run `test_sandpile_mpfr.py` in your actual Sage environment. This is a source package, not a prebuilt `.so`.

## A. Two simulations from the same fpylll `GSO.Mat`

```python
from fpylll import IntegerMatrix, GSO, FPLLL
from sandpile_mpfr import SandpileSeq

FPLLL.set_precision(212)    # BEFORE creating MPFR GSO
B = IntegerMatrix.random(96, 'qary', bits=14, k=48)
M = GSO.Mat(B, float_type='mpfr')
M.update_gso()

# The two simulators independently COPY the initial r and FULL mu state.
l = SandpileSeq(M, condition='lovasz', delta='0.749',
                mubnd='0.5001', seed=42)
s = SandpileSeq(M, condition='siegel', delta='0.749',
                mubnd='0.5001', seed=42)

l.run(max_swaps=1000000)
s.run(max_swaps=1000000)

print('Lovasz:', l.swaps, 'Siegel:', s.swaps)
print('initial mu matrix maintained in copies:', len(l.mu_matrix()))
```

These are *independent copies*: running `l` never changes `s`, `M`, or `B`. You may choose different seeds for independent random streams. With identical seeds, the random draws still diverge if different conditions cause different sequences of swaps.

For a native-precision copy, use `GSO.Mat(..., float_type='mpfr')`. A `dd`, `qd` or `double` source is also accepted but **explicitly warns** and falls back to binary64 `get_r`/`get_mu` before storing in MPFR (53-bit significant precision only). Pass `allow_lossy_gso=False` to reject such inputs. Do not change fpylll's global MPFR precision while the source GSO is alive.

## B. Synthetic Python-side initialization at chosen precision

```python
from sandpile_mpfr import SandpileSeq
from sage.all import RealField, matrix

R = RealField(212)
r = [R(100), R(70), R(120)]  # SQUARED GSO lengths, not log lengths
mu = matrix(R, 3, 3, 0)
mu[1, 0] = R('0.5')
mu[2, 0] = R('0.125')  # retained, unused by the sequential model
mu[2, 1] = R('-0.375')

sim = SandpileSeq((r, mu), precision=212,
                  condition='siegel', delta='0.749',
                  mubnd='0.5001', seed=42)
sim.run()
result = sim.snapshot()

# Recover Sage MPFR values without converting through double:
R = RealField(result['precision'])
r_out = [R(x) for x in result['r']]
mu_out = matrix(R, [[R(x) for x in row] for row in result['mu']])
```

Python floats are accepted (their *already limited* binary64 precision is only cast to MPFR). To retain high precision, supply Sage `RealField` values or decimal strings with sufficient digits. `mu=None` explicitly makes the matrix all zero.

## C. Mathematical convention

With `r[i] = ||b_i^*||^2` and `mu[i+1,i]`:

- Lovasz swaps when `delta*r[i] > r[i+1] + mu[i+1,i]**2*r[i]`.
- Siegel swaps when `delta*r[i] > r[i+1]`.

Here `delta` is your **uploaded source code's** coefficient, not `4/3 + delta**2` from the discussion. For a uniform potential-decrease guarantee in the Siegel variant, a sufficient condition is `delta + mubnd**2 < 1`; your defaults satisfy this. `max_swaps` protects against numerical or modeling nontermination.

The simulation keeps the *full* coefficient matrix but, just like the original, it **only re-samples the swap index's subdiagonal mu and its immediate neighbors** after each swap. It does not make the full matrix a mathematically consistent GSO of an evolving lattice. This is intentional: it preserves the sandpile approximation while keeping state available for future work on DeepLLL.

## Tests

```bash
python test_sandpile_mpfr.py
```

## Build troubleshooting (v0.2.1)

If compilation ends with a Cython internal exception such as
`AttributeError: 'NoneType' object has no attribute 'is_builtin_type'`,
**do not interpret the subsequent `ModuleNotFoundError` as a second defect**:
there simply is no `.so` to import until compilation succeeds.

The new `setup.py` checks *before* running Cython that the installed fpylll
contains the required `.pxd` declarations. If they are missing,
it prints a concrete instruction to download the source package matching your
installed version and use `FPYLLL_SOURCE` to point at its `src` folder. The build script
reconstructs `config.pxi` locally from your installed `fpylll.config` flags,
so a generated `config.pxi` need not be present in the source archive. **This
does not reinstall or replace fpylll.** If the declarations are present but
Cython still crashes, please capture the **first error or compiler traceback**
(not just the final `CompileError`) along with `python -m pip show Cython fpylll`.

This revision also replaces the `fp_nr_t` aggregate temporary in the native
GSO bridge with `FP_NR[mpfr_t]`, directly matching the MPFR specialization of
fplll's `get_r()` and `get_mu()` interfaces.

The `.pyx` file was translated to C++ with Cython 3.2.4 against a local
minimal interface stub. Full compilation/loading with the actual fpylll
binary is **not verified** in this environment, which has no fpylll install.
