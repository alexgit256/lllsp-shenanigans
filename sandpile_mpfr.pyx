# distutils: language = c++
# cython: language_level=3
"""MPFR rewrite of the original Sandpile_Model_Seq (sequential only).

The only algorithmic switch is the swap predicate.  The entire lower-triangular
mu matrix is kept, but EXACTLY as in the original C++ only the three adjacent
subdiagonal coefficients are randomly resampled after a swap.  This is not an
exact lattice GSO transformation.

Inputs:
  * fpylll.GSO.Mat(..., float_type='mpfr') -- direct Cython C++/MPFR copy
  * (squared_profile, full_mu_matrix) -- Python scalars -> MPFR at chosen bits

The original fpylll GSO and basis are never mutated by the simulator, except
for computing previously uncomputed GSO coefficients with update_gso().
"""

from libc.stdlib cimport malloc, free
from libc.stddef cimport size_t

from fpylll.fplll.gso cimport MatGSO
from fpylll.fplll.decl cimport mat_gso_mpz_mpfr, mat_gso_long_mpfr
from fpylll.fplll.fplll cimport FP_NR
from fpylll.mpfr.mpfr cimport (
    mpfr_t, mp_prec_t, mp_exp_t, MPFR_RNDN,
    mpfr_init2, mpfr_clear, mpfr_set, mpfr_set_d, mpfr_set_ui, mpfr_set_str,
    mpfr_sqr, mpfr_mul, mpfr_mul_ui, mpfr_add, mpfr_sub,
    mpfr_div, mpfr_div_ui, mpfr_cmp, mpfr_cmp_ui,
    mpfr_number_p, mpfr_get_str, mpfr_free_str, mpfr_urandomb,
)
from fpylll.gmp.types cimport gmp_randstate_t
from fpylll.gmp.random cimport gmp_randinit_default, gmp_randseed_ui, gmp_randclear


cdef str _decimal(mpfr_t x):
    """Exact round-trip decimal text for this MPFR value (no double conversion)."""
    cdef mp_exp_t exponent
    cdef char* ptr = mpfr_get_str(NULL, &exponent, 10, 0, x, MPFR_RNDN)
    cdef str digits
    if ptr == NULL:
        raise MemoryError("mpfr_get_str failed")
    try:
        digits = (<bytes>ptr).decode('ascii')
    finally:
        mpfr_free_str(ptr)
    if digits.startswith('-'):
        return '-0.' + digits[1:] + 'e' + str(exponent)
    return '0.' + digits + 'e' + str(exponent)


def _matrix_get(m, Py_ssize_t i, Py_ssize_t j):
    """Supports Sage matrices, NumPy arrays, or row-indexed nested lists."""
    try:
        return m[i, j]
    except (TypeError, IndexError):
        return m[i][j]


cdef class SandpileSeq:
    """Independent stochastic MPFR simulation with a retained n-by-n mu matrix.

    `condition`: 'lovasz' or 'siegel' in *fpylll's* convention:
        Lovasz: delta*r[i] > r[i+1] + mu[i+1,i]**2*r[i]
        Siegel: (delta - eta**2)*r[i] > r[i+1]

    `eta` is both the fpylll size-reduction bound and the width of the
    *current* uniform sampler (alternative sampler is a future TODO).
    `run` can be called repeatedly. `snapshot()` exports full-precision strings.
    """
    cdef mpfr_t* _r
    cdef mpfr_t* _mu          # n*n matrix: row-major, including upper zeros
    cdef Py_ssize_t _n, _r_init, _mu_init
    cdef long _prec
    cdef mpfr_t _delta, _eta, _siegel_coeff, _half, _lhs, _rhs, _tmp, _old_r, _new_r
    cdef bint _scalar_init, _rng_init, _siegel
    cdef gmp_randstate_t _rng
    cdef public unsigned long long swaps, swaps_last_block, tests
    cdef object _valid_swap_mu

    def __cinit__(self):
        self._r = NULL
        self._mu = NULL
        self._n = 0
        self._r_init = 0
        self._mu_init = 0
        self._scalar_init = False
        self._rng_init = False
        self.swaps = self.swaps_last_block = 0
        self.tests = 0
        self._valid_swap_mu = []

    def __dealloc__(self):
        cdef Py_ssize_t j
        if self._r != NULL:
            for j in range(self._r_init):
                mpfr_clear(self._r[j])
            free(self._r)
        if self._mu != NULL:
            for j in range(self._mu_init):
                mpfr_clear(self._mu[j])
            free(self._mu)
        if self._scalar_init:
            mpfr_clear(self._delta)
            mpfr_clear(self._eta)
            mpfr_clear(self._siegel_coeff)
            mpfr_clear(self._half)
            mpfr_clear(self._lhs)
            mpfr_clear(self._rhs)
            mpfr_clear(self._tmp)
            mpfr_clear(self._old_r)
            mpfr_clear(self._new_r)
        if self._rng_init:
            gmp_randclear(self._rng)

    def __init__(self, source, *, condition='lovasz', delta=0.99,
                 eta=0.51, precision=None, seed=42,
                 allow_lossy_gso=True):
        """`source` is either an MPFR MatGSO or `(profile, full_mu_matrix)`.

        `profile[i]` must contain SQUARED GS lengths. Full mu matrix entries
        with j<i are retained; the upper triangle and diagonal are zeroed.
        Python floats are accepted but retain only their original 53 bits;
        use decimal strings or Sage MPFR values to preserve extra precision.
        """
        from fpylll import FPLLL
        cdef Py_ssize_t n, i
        cdef bint from_gso = isinstance(source, MatGSO)
        cdef long source_prec
        cdef bint direct_mpfr = False

        if condition not in ('lovasz', 'siegel'):
            raise ValueError("condition must be 'lovasz' or 'siegel'")
        self._siegel = condition == 'siegel'
        if from_gso:
            direct_mpfr = ((<MatGSO>source)._type == mat_gso_mpz_mpfr or
                           (<MatGSO>source)._type == mat_gso_long_mpfr)
            if not direct_mpfr and not allow_lossy_gso:
                raise TypeError('Non-MPFR GSO requires allow_lossy_gso=True')
            if not direct_mpfr:
                import warnings
                warnings.warn('Non-MPFR fpylll GSO: copying get_r/get_mu via '
                              'binary64 before casting to MPFR. dd/qd precision '
                              'is NOT preserved. Prefer float_type="mpfr".',
                              RuntimeWarning, stacklevel=2)
            n = (<MatGSO>source).B.nrows
            # MPFR precision is global in fplll: use the precision actually
            # configured when constructing the source, and do not change it.
            source_prec = int(FPLLL.get_precision('mpfr'))
            if precision is None:
                precision = source_prec if direct_mpfr else 212
            elif direct_mpfr and int(precision) < source_prec:
                raise ValueError('target precision smaller than fplll MPFR precision')
        else:
            if not isinstance(source, (tuple, list)) or len(source) != 2:
                raise TypeError('source must be an MPFR GSO.Mat or (squared_profile, full_mu_matrix)')
            n = len(source[0])
            if precision is None:
                precision = 212
        if n < 2:
            raise ValueError('dimension must be >= 2')
        if int(precision) < 53:
            raise ValueError('precision must be >= 53 bits')
        if int(seed) < 0:
            raise ValueError('seed must be >= 0')
        self._n = n
        self._prec = int(precision)
        self._r = <mpfr_t*>malloc(n * sizeof(mpfr_t))
        self._mu = <mpfr_t*>malloc(n*n * sizeof(mpfr_t))
        if self._r == NULL or self._mu == NULL:
            raise MemoryError()
        for i in range(n):
            mpfr_init2(self._r[i], <mp_prec_t>self._prec)
            self._r_init += 1
        for i in range(n*n):
            mpfr_init2(self._mu[i], <mp_prec_t>self._prec)
            self._mu_init += 1
            mpfr_set_ui(self._mu[i], 0, MPFR_RNDN)
        mpfr_init2(self._delta, self._prec)
        mpfr_init2(self._eta, self._prec)
        mpfr_init2(self._siegel_coeff, self._prec)
        mpfr_init2(self._half, self._prec)
        mpfr_init2(self._lhs, self._prec)
        mpfr_init2(self._rhs, self._prec)
        mpfr_init2(self._tmp, self._prec)
        mpfr_init2(self._old_r, self._prec)
        mpfr_init2(self._new_r, self._prec)
        self._scalar_init = True
        self._parse(self._delta, delta)
        self._parse(self._eta, eta)
        if mpfr_cmp_ui(self._delta, 0) <= 0 or mpfr_cmp_ui(self._delta, 1) > 0:
            raise ValueError('delta must be in (0, 1]')
        if mpfr_cmp_ui(self._eta, 0) <= 0:
            raise ValueError('eta must be positive')
        mpfr_sqr(self._tmp, self._eta, MPFR_RNDN)
        mpfr_sub(self._siegel_coeff, self._delta, self._tmp, MPFR_RNDN)
        if mpfr_cmp_ui(self._siegel_coeff, 0) <= 0:
            raise ValueError('eta**2 must be smaller than delta')
        mpfr_set_ui(self._half, 1, MPFR_RNDN)
        mpfr_div_ui(self._half, self._half, 2, MPFR_RNDN)
        gmp_randinit_default(self._rng)
        self._rng_init = True
        gmp_randseed_ui(self._rng, <unsigned long>int(seed))

        if from_gso:
            if direct_mpfr:
                self._load_gso(<MatGSO>source)
            else:
                self._load_gso_lossy(<MatGSO>source)
        else:
            self._load_arrays(source[0], source[1])
        for i in range(n):
            if not mpfr_number_p(self._r[i]) or mpfr_cmp_ui(self._r[i], 0) <= 0:
                raise ArithmeticError(f'nonpositive or nonfinite squared GSO norm at {i}')
        for i in range(n*n):
            if not mpfr_number_p(self._mu[i]):
                raise ArithmeticError(f'nonfinite mu matrix entry at {i//n}, {i%n}')

    cdef void _parse(self, mpfr_t out, object value) except *:
        # Avoid float() on full-precision input; cast using its printed value.
        # To preserve Sage values, str(value) must show enough digits, as Sage
        # RealNumber normally does; callers may always pass explicit strings.
        cdef bytes b
        if isinstance(value, float):
            # Exact promotion of the binary64 value rather than reinterpreting
            # its shortest decimal repr as a different real number.
            mpfr_set_d(out, <double>value, MPFR_RNDN)
            if not mpfr_number_p(out):
                raise ValueError(f'nonfinite real {value!r}')
            return
        b = str(value).encode('ascii')
        if mpfr_set_str(out, b, 10, MPFR_RNDN) != 0:
            raise ValueError(f'cannot parse decimal real {value!r}')
        if not mpfr_number_p(out):
            raise ValueError(f'nonfinite decimal real {value!r}')

    cdef void _load_arrays(self, object profile, object mu_matrix) except *:
        cdef Py_ssize_t i, j, n = self._n
        for i in range(n):
            self._parse(self._r[i], profile[i])
        if mu_matrix is None:
            return  # explicit all-zero mu matrix
        for i in range(1, n):
            for j in range(i):
                self._parse(self._mu[i*n+j], _matrix_get(mu_matrix, i, j))

    cdef void _load_gso(self, MatGSO M) except *:
        """Copy directly from the C++ FP_NR<mpfr_t>, avoiding get_d()."""
        cdef Py_ssize_t i, j, n = self._n
        # Use the single concrete fplll floating-point type instead of the
        # fp_nr_t aggregate of all possible backends.  That aggregate is
        # unnecessary and can interact poorly with Cython's C++ type handling.
        cdef FP_NR[mpfr_t] t
        if not M.update_gso():
            raise ArithmeticError('MatGSO.update_gso() failed')
        for i in range(n):
            if M._type == mat_gso_mpz_mpfr:
                M._core.mpz_mpfr.get_r(t, <int>i, <int>i)
            else:
                M._core.long_mpfr.get_r(t, <int>i, <int>i)
            mpfr_set(self._r[i], t.get_data(), MPFR_RNDN)
            for j in range(i):
                if M._type == mat_gso_mpz_mpfr:
                    M._core.mpz_mpfr.get_mu(t, <int>i, <int>j)
                else:
                    M._core.long_mpfr.get_mu(t, <int>i, <int>j)
                mpfr_set(self._mu[i*n+j], t.get_data(), MPFR_RNDN)

    cdef void _load_gso_lossy(self, MatGSO M) except *:
        """Fallback for dd/qd/double: convert binary64 fpylll accessors to MPFR."""
        cdef Py_ssize_t i, j, n = self._n
        cdef double d
        if not M.update_gso():
            raise ArithmeticError('MatGSO.update_gso() failed')
        for i in range(n):
            d = M.get_r(i, i)
            mpfr_set_d(self._r[i], d, MPFR_RNDN)
            for j in range(i):
                d = M.get_mu(i, j)
                mpfr_set_d(self._mu[i*n+j], d, MPFR_RNDN)

    cdef void _sample_mu_entry(self, Py_ssize_t row, Py_ssize_t col) noexcept:
        """Uniform (-eta, eta), sampled directly into a native MPFR entry."""
        cdef mpfr_t* x = &self._mu[row*self._n+col]
        mpfr_urandomb(x[0], self._rng)
        mpfr_sub(x[0], x[0], self._half, MPFR_RNDN)
        mpfr_mul(x[0], x[0], self._eta, MPFR_RNDN)
        mpfr_mul_ui(x[0], x[0], 2, MPFR_RNDN)

    cdef void _sample_subdiag(self, Py_ssize_t i) noexcept:
        """Resample mu[i+1,i] (caller validates that it is in the window)."""
        self._sample_mu_entry(i+1, i)

    def resample_external_mu(self, Py_ssize_t start=0, kappa=None):
        """Separately resample mu[row,col] for row>=kappa, start<=col<kappa.

        This is an independent stochastic approximation to downstream GSO
        coefficient propagation, not an exact basis/GSO transformation.
        Neither profile nor any other mu coefficient is altered.
        """
        cdef Py_ssize_t stop, row, col
        if kappa is None:
            stop = self._n
        else:
            stop = <Py_ssize_t>kappa
        if not (0 <= start < stop <= self._n):
            raise ValueError('require 0 <= start < kappa <= n')
        for row in range(stop, self._n):
            for col in range(start, stop):
                self._sample_mu_entry(row, col)
        return self

    def run(self, unsigned long long max_swaps=1000000, *,
            Py_ssize_t start=0, kappa=None):
        """Simulate only [start,kappa); never mutate coefficients outside.

        swaps is cumulative; swaps_last_block counts the current invocation.
        max_swaps applies per invocation (even across repeated block calls).
        """
        cdef Py_ssize_t stop, i, n = self._n
        if kappa is None:
            stop = n
        else:
            stop = <Py_ssize_t>kappa
        if not (0 <= start < stop <= n):
            raise ValueError('require 0 <= start < kappa <= n')
        self.swaps_last_block = 0
        i = start
        while i < stop-1:
            self.tests += 1
            if self._siegel:
                mpfr_mul(self._lhs, self._siegel_coeff, self._r[i], MPFR_RNDN)
                mpfr_set(self._rhs, self._r[i+1], MPFR_RNDN)
            else:
                mpfr_mul(self._lhs, self._delta, self._r[i], MPFR_RNDN)
                mpfr_sqr(self._tmp, self._mu[(i+1)*n+i], MPFR_RNDN)
                mpfr_mul(self._rhs, self._tmp, self._r[i], MPFR_RNDN)
                mpfr_add(self._rhs, self._rhs, self._r[i+1], MPFR_RNDN)
            if mpfr_cmp(self._lhs, self._rhs) <= 0:
                i += 1
                continue
            if self.swaps_last_block >= max_swaps:
                raise RuntimeError(f'max_swaps={max_swaps} reached; state preserved')

            # Collect the size-reduced mu IMMEDIATELY before a valid swap.
            # TODO: replace uniform re-sampling with a distribution conditional
            # on the event that the subsequent swap is valid (user-requested).
            self._valid_swap_mu.append(_decimal(self._mu[(i+1)*n+i]))

            # Exact same algebra as Sandpile_Model_Seq in uploaded C++:
            #   temp = r[i+1]
            #   r[i+1] = r[i]
            #   r[i] = temp + mu[i+1,i]^2 * r[i]
            #   r[i+1] = (old r[i])*temp / new r[i].
            mpfr_set(self._old_r, self._r[i], MPFR_RNDN)
            mpfr_sqr(self._tmp, self._mu[(i+1)*n+i], MPFR_RNDN)
            mpfr_mul(self._new_r, self._tmp, self._old_r, MPFR_RNDN)
            mpfr_add(self._new_r, self._new_r, self._r[i+1], MPFR_RNDN)
            if not mpfr_number_p(self._new_r) or mpfr_cmp_ui(self._new_r, 0) <= 0:
                raise ArithmeticError(f'bad GSO diagonal after swap at {i}')
            mpfr_mul(self._tmp, self._old_r, self._r[i+1], MPFR_RNDN)
            mpfr_div(self._r[i+1], self._tmp, self._new_r, MPFR_RNDN)
            mpfr_set(self._r[i], self._new_r, MPFR_RNDN)
            self.swaps += 1
            self.swaps_last_block += 1

            # Original stochastic rule, unchanged; full mu matrix is RETAINED.
            self._sample_subdiag(i)
            if i > start:
                self._sample_subdiag(i-1)
            if i+1 < stop-1:
                self._sample_subdiag(i+1)
            if i > start:
                i -= 1
        return self

    @property
    def n(self):
        return self._n

    @property
    def precision(self):
        return self._prec

    @property
    def condition(self):
        return 'siegel' if self._siegel else 'lovasz'

    def profile(self):
        """Squared GS profile, as exact MPFR round-trip decimal strings."""
        return [_decimal(self._r[i]) for i in range(self._n)]

    def mu_matrix(self):
        """Full lower triangular mu matrix, decimal strings."""
        cdef Py_ssize_t i, j, n = self._n
        return [[_decimal(self._mu[i*n+j]) for j in range(n)]
                for i in range(n)]

    def valid_swap_mu(self):
        """All subdiagonal mu values at successful swap decisions."""
        return list(self._valid_swap_mu)

    def final_subdiagonal_mu(self):
        """All mu[i+1,i] at the current state, in MPFR decimal strings."""
        cdef Py_ssize_t i, n = self._n
        return [_decimal(self._mu[(i+1)*n+i]) for i in range(n-1)]

    def snapshot(self):
        return {
            'r': self.profile(),
            'mu': self.mu_matrix(),
            'precision': self._prec,
            'condition': self.condition,
            'swaps': int(self.swaps),
            'swaps_last_block': int(self.swaps_last_block),
            'tests': int(self.tests),
            'valid_swap_mu': self.valid_swap_mu(),
            'final_subdiagonal_mu': self.final_subdiagonal_mu(),
        }


def gso_mu_entry(MatGSO M, Py_ssize_t i, Py_ssize_t j):
    """Read a single live GSO mu(i,j) without binary64 conversion (MPFR backend).

    Caller must ensure the row is up to date, e.g. M.update_gso_row(i,i).
    Intended for instrumented pre-swap logging.
    """
    cdef FP_NR[mpfr_t] t
    if not (0 <= j < i < M.B.nrows):
        raise IndexError('require 0 <= j < i < n')
    if M._type == mat_gso_mpz_mpfr:
        M._core.mpz_mpfr.get_mu(t, <int>i, <int>j)
        return _decimal(t.get_data())
    if M._type == mat_gso_long_mpfr:
        M._core.long_mpfr.get_mu(t, <int>i, <int>j)
        return _decimal(t.get_data())
    return repr(M.get_mu(i, j))  # fallback (only for non-MPFR GSO)


def gso_swap_is_valid(MatGSO M, Py_ssize_t k, double delta, double eta,
                      bint siegel=False):
    """Evaluate the fpylll-style adjacent test with native MPFR GSO values.

    No get_d(), no binary64 GSO conversion. delta and eta are binary64 values
    because that is fpylll.LLL.Reduction's public parameter ABI.
    The caller must keep the relevant GSO rows up to date.
    """
    cdef FP_NR[mpfr_t] prev_r, curr_r, mu
    cdef mpfr_t d, e, lhs, rhs, tmp
    cdef long prec
    from fpylll import FPLLL
    if k <= 0 or k >= M.B.nrows:
        raise IndexError('swap index k must satisfy 0 < k < n')
    if M._type == mat_gso_mpz_mpfr:
        M._core.mpz_mpfr.get_r(prev_r, <int>(k-1), <int>(k-1))
        M._core.mpz_mpfr.get_r(curr_r, <int>k, <int>k)
        M._core.mpz_mpfr.get_mu(mu, <int>k, <int>(k-1))
    elif M._type == mat_gso_long_mpfr:
        M._core.long_mpfr.get_r(prev_r, <int>(k-1), <int>(k-1))
        M._core.long_mpfr.get_r(curr_r, <int>k, <int>k)
        M._core.long_mpfr.get_mu(mu, <int>k, <int>(k-1))
    else:
        raise TypeError('gso_swap_is_valid requires float_type="mpfr"')
    prec = int(FPLLL.get_precision('mpfr'))
    mpfr_init2(d, prec)
    mpfr_init2(e, prec)
    mpfr_init2(lhs, prec)
    mpfr_init2(rhs, prec)
    mpfr_init2(tmp, prec)
    try:
        mpfr_set_d(d, delta, MPFR_RNDN)
        mpfr_set_d(e, eta, MPFR_RNDN)
        if siegel:
            mpfr_sqr(tmp, e, MPFR_RNDN)
            mpfr_sub(d, d, tmp, MPFR_RNDN)
            mpfr_mul(lhs, d, prev_r.get_data(), MPFR_RNDN)
            return mpfr_cmp(lhs, curr_r.get_data()) > 0
        mpfr_mul(lhs, d, prev_r.get_data(), MPFR_RNDN)
        mpfr_sqr(tmp, mu.get_data(), MPFR_RNDN)
        mpfr_mul(rhs, tmp, prev_r.get_data(), MPFR_RNDN)
        mpfr_add(rhs, rhs, curr_r.get_data(), MPFR_RNDN)
        return mpfr_cmp(lhs, rhs) > 0
    finally:
        mpfr_clear(d)
        mpfr_clear(e)
        mpfr_clear(lhs)
        mpfr_clear(rhs)
        mpfr_clear(tmp)
