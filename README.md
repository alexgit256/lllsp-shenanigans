# Paired LLL / Siegel–LLL / MPFR sandpile benchmark (phases 1–3)

This extends the **working `sandpile_mpfr_bridge`**, retaining the full
lower-triangular `mu` matrix and original sequential sandpile update formulas.
Only the swap predicate changes according to fpylll's convention:

- Lovász: `delta*r[i] > r[i+1] + mu[i+1,i]^2*r[i]`.
- Siegel: `(delta-eta^2)*r[i] > r[i+1]`.

`eta` also replaces the old sampler width `mubnd`: after a swap the existing
model independently samples the **three neighboring subdiagonal** coefficients
uniformly from `(-eta,eta)`. Off-subdiagonal entries remain stored unchanged.
Future TODO: sample the new `mu` **conditionally on the pending valid swap**.
Do not confuse the simulator's full stored matrix with an exact evolving GSO.

## Requirements and building (Sage / WSL)

Use the Python belonging to your existing Sage/fpylll environment. The project
reuses `setup.py` and `sandpile_mpfr.pyx` and expects the same fpylll `0.6.4`
Cython source declarations as the working earlier bridge. For example, if you
unzip this folder next to your earlier `sandpile_mpfr_bridge`:

```bash
cd /mnt/c/Users/treme/ubuntu/git/lllsim/lllsp_benchmark
export FPYLLL_SOURCE="$(realpath ../sandpile_mpfr_bridge/fpylll-0.6.4/src)"
python setup.py build_ext --inplace --force
python test_sandpile_mpfr.py
```

You can rebuild after Cython edits with `python setup.py build_ext --inplace`.
If imported in Jupyter, **restart the kernel** after rebuilding.

Run generation and workers via a terminal (`python bench_cli.py ...`), not
inside a notebook; process workers use the `spawn` start method.

## Quick start

Start with the intentionally **small** four-family configuration:

```bash
python bench_cli.py all --config example_config.json --output results --workers 4
```

Or execute each phase separately:

```bash
python bench_cli.py generate --config example_config.json --output results
python bench_cli.py run --output results --workers 4
python bench_cli.py summarize --output results
python bench_cli.py plot --output results
```

Each family has its own `count`. Configuration allows arrays of variants, for
example separate NTRU degrees or q-ary dimensions under the same family. To
repeat generation intentionally, use `generate --overwrite`. To repeat reduction
jobs, use `run --rerun`.

## How real LLL is compared

For each generated lattice:

1. **Preprocess once:** use `LLL.Reduction(M).size_reduction(0,n)` on the basis
   (not full LLL); check `|mu[i,j]| <= eta + 1e-6` in the accessible diagnostic.
   Store the preprocessed basis and the initial MPFR GSO.
2. Initialize the independent simulators from this *same* MPFR `GSO.Mat`.
3. Run native fpylll `LLL.Reduction` with `LLL.DEFAULT` and `LLL.SIEGEL`,
   both with the same `(delta,eta)`.
4. Run the matching `SandpileSeq` for each condition and each simulator seed.
5. Separately replay the corresponding classical **adjacent-swap LLL** using
   fpylll's size-reduction and `MatGSO` operations. Immediately before each
   valid swap, read `mu[k,k-1]` **directly through the native MPFR Cython
   bridge**. This replay is *not a callback into the native fplll reducer*:
   native fplll can have a different insertion/scheduling path. Both its swap
   count and profile distance to native LLL are retained for auditing.

The real LLL statistics always come from **native fpylll**, not from the replay.
The real **pre-valid-swap μ** histogram comes from the instrumented replay;
real **terminal μ** come from the native reducer. The simulator also stores both
its pre-valid-swap μ samples and its final subdiagonal μ.

The fpylll GSO → sandpile initial copy and Cython pre-swap μ read preserve
native **MPFR** precision. The Python-supplied `(profile, mu)` constructor is
still supported. fpylll reduction parameter `delta` and `eta` are binary64 at
the public API boundary, and the benchmark uses identical parameter doubles in
both programs.

## Lattice families

- **`ntru`**: faithful Python translation of your `demo.ipynb` NTRU generator:
  short Gaussian `f,g`, `h=g/f mod q` in `(Z/qZ)[x]/(x^n-1)`, row basis
  `[[0,qI],[I,H]]` with centered coefficients. Configuration `degree` is the
  polynomial degree; lattice *dimension* is `2*degree`. Adjustable `sigma`,
  `sigma_f`, `sigma_g`, `q`. Secrets stored in instance metadata.
- **`qary`**: `IntegerMatrix.random(d, 'qary', bits=..., k=...)` (or `q=...`).
- **`intrel`**: fpylll integer-relation generator. Shape `d × (d+1)`; rank is
  `d`. RHF normalization uses the **lattice covolume in its row span**, not
  the determinant of a nonexistent square matrix.
- **`coppersmith`**: configurable monic univariate modular small-root shift
  lattice with coefficient scaling by `X^j`. Parameters `degree,m,t,N-bits,X`.
  A modular root is planted; this is a standard illustrative *family* of
  polynomial-shift lattices, not an asserted benchmark for every Coppersmith
  application.

NTRU reference: https://github.com/alexgit256/lllsp-shenanigans/blob/main/demo.ipynb

## Output structure

```text
results/
  manifest.json                      # family counts, config, deterministic seeds
  instances/<id>.json.gz             # generated raw integer basis and parameters
  runs/<id>/size_reduced_input.json.gz
  runs/<id>/initial_gso.json.gz
  runs/<id>/lovasz_real.json.gz
  runs/<id>/siegel_real.json.gz
  runs/<id>/lovasz_sim_000.json.gz ...
  runs/<id>/siegel_sim_000.json.gz ...
  run_report.json
  aggregate.json                     # groups + descriptive statistics
  summary.csv                        # means, stds, paired bias, RMSE
  paired.csv                         # one row per lattice per condition
  profiles_summary.json.gz           # mean/SD log2 profiles for plotting
  plots/<family>_v<...>_profiles.png
  plots/<family>_v<...>_before_valid_swap.png
  plots/<family>_v<...>_final_mu.png
```

Each run's `.json.gz` contains **full** final lower-triangular μ matrix,
**full-precision decimal strings** for squared GSO profile, RHF, slope,
log-potential, swap counters and time. `*_real.json.gz` also contains:

- `instrumented_trace.valid_swap_mu`: all reduced μ values immediately before
  successful swaps in the instrumented **real** LLL replay;
- `final_subdiagonal_mu`: **native** fpylll terminal μ values;
- `instrumented_trace.profile_discrepancy`: distance of replay versus native;
- both replay and native swap counts.

`*_sim_*.json.gz` includes `valid_swap_mu` and `final_subdiagonal_mu` from
**SandpileSeq**. All these distributions are saved, not only visualized.

### Metric convention

Let `p_i = log2(r_ii)`, squared GSO lengths, dimension `d`:

- `slope`: least-squares regression slope of `p_i` against `i`.
- `rhf`: `(||b_1|| / covol(L)^(1/d))^(1/d)`.
- `log2_potential`: `sum_i (d-1-i)*p_i`.
- `profile_mae/rmse`: paired differences in log2 squared GSO lengths.
- `swaps`: native `LLL.Reduction.nswaps` versus sandpile swap count.
- `runtime_sec`: wall clock, excluding input generation and plotting.

Summary groups use **(family, variant, dimension, condition)** and compute
mean, sample SD, median, 5/25/75/95% quantiles, a normal-approximate CI for
the mean, plus paired bias and RMSE. If multiple simulator seeds are used for
one lattice, their metrics and profiles are **averaged within that lattice**
first; extra seeds do not falsely increase the independent lattice count.
Plots show the mean±SD profile, an event-weighted pre-swap μ histogram, and a
terminal μ histogram. The distribution plots deliberately pool events and are
not unbiased per-lattice distributional estimates.

## Deferred / limitations

- `bench_invariants.py`: TODO hooks for full determinant conservation, terminal
  condition, potential monotonicity and cross-precision validation. These are
  **deliberately not executed**, as requested.
- No intermediate trajectories or checkpoint alignment (deferred).
- TODO: sample **conditional** `mu` before a pending valid simulated swap;
  for now uniform post-swap resampling remains as in the source.
- fpylll does not expose an exact native pre-swap callback. The separately
  instrumented adjacent-swap replay is useful for distribution measurement but
  may not match the native reducer's insertion sequence. Compare its swap
  count and stored profile discrepancy before drawing conclusions.
- `SandpileSeq` keeps full μ storage, but only the three neighboring
  subdiagonal entries are updated after swaps. Not exact GSO propagation.
- The benchmark assumes successful finite/rank-sufficient MPFR GSO. Partial
  worker failures are saved as `error.json` or recorded in the result.
- Tests of summary/plots can run locally without Sage; Cython/MPFR and Sage
  integration require the user's environment. **End-to-end compilation and
  evaluation were not possible in the authoring container**.

## Basic tests

```bash
python test_sandpile_mpfr.py     # needs built extension + Sage/fpylll
python test_analysis_smoke.py    # summary & plotting; no fpylll required
```

## Additional plots

Besides mean±SD profiles and both μ histograms, each group has four
per-lattice scatter plots comparing native versus simulated-mean **RHF, slope,
log-potential and swap count** against the `y=x` reference diagonal.
