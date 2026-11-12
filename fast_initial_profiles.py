"""Fast synthetic initial states for sandpile_mpfr.SandpileSeq.

No fpylll, Sage, integer lattice matrix, or GSO is constructed. Requires mpmath.
Profiles contain SQUARED Gram--Schmidt norms and lower triangular mu.

qary: exact for row basis [qI_k,0]; [H,I_(n-k)], with centered H/q.
intrel: exact GSO for rows (a_i,e_i), d x (d+1).
coppersmith: explicitly heuristic profile for monic modular polynomial shifts;
             preserves the triangular determinant (not exact GSO).

The separate source data are converted to decimal strings at the requested MPFR
precision; using strings avoids an accidental intermediate binary64 conversion.
"""

from __future__ import annotations

import random
from typing import Optional
from mpmath import mp


def _check(n, precision, seed, delta, eta, condition):
    if not isinstance(n, int) or n < 2:
        raise ValueError('dimension must be an integer >= 2')
    if not isinstance(precision, int) or precision < 53:
        raise ValueError('precision must be an integer >= 53')
    if not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if condition not in ('lovasz', 'siegel'):
        raise ValueError("condition must be 'lovasz' or 'siegel'")
    if not (0.25 < delta <= 1.0 and 0.5 <= eta < 1.0):
        raise ValueError('require 0.25 < delta <= 1 and 0.5 <= eta < 1')


def _zero_mu(n):
    return [['0'] * n for _ in range(n)]


def _num(x, precision):
    # At least ceil(precision * log10(2)) + a guard digit.
    return mp.nstr(x, n=int(precision * 0.30103) + 9,
                   strip_zeros=False, min_fixed=-8, max_fixed=12)


def _make_sim(r, mu, *, condition, delta, eta, precision, seed):
    from sandpile_mpfr import SandpileSeq
    return SandpileSeq((r, mu), condition=condition, delta=delta,
                       eta=eta, precision=precision, seed=seed)


def initial_qary(n: int, k: int, bits: float, *, precision: int = 212,
                 seed: int = 42, condition: str = 'lovasz',
                 delta: float = 0.99, eta: float = 0.51):
    """Synthetic qary, q-vectors FIRST, with q=2**bits.

    k is fplll's exponent of det=q**k: number of q-diagonal rows.
    bits may be fractional. The quotient H/q is centered uniform (-1/2,1/2).
    For integral q with H uniform modulo q this is only an approximation
    to the discrete distribution; profile remains exact for any q>0.
    """
    _check(n, precision, seed, delta, eta, condition)
    if not isinstance(k, int) or not (0 <= k <= n):
        raise ValueError('k must be an integer in [0,n]')
    if not (float(bits) > 0 and float(bits) < float('inf')):
        raise ValueError('bits must be finite and positive')
    rng = random.Random(seed)
    mu = _zero_mu(n)
    with mp.workprec(precision + 32):
        q2 = mp.power(2, 2 * mp.mpf(str(bits)))
        r = [_num(q2, precision)] * k + ['1'] * (n-k)
    for i in range(k, n):
        for j in range(k):
            mu[i][j] = repr(rng.random() - 0.5)
    return _make_sim(r, mu, condition=condition, delta=delta, eta=eta,
                     precision=precision, seed=seed)


def initial_intrel(n: int, bits: int, *, precision: int = 212,
                   seed: int = 42, condition: str = 'lovasz',
                   delta: float = 0.99, eta: float = 0.51,
                   mu_mode: str = 'exact', zero_probability: float = 0.85):
    """GSO for n rows (a_i,e_i) in ambient dimension n+1.

    Exact: r_i = S_i/S_(i-1), mu_ij = a_i*a_j/S_j,
    with S_i=1+sum_{h<=i} a_h**2. Even for huge bits, MPFR values
    are never converted through Python float.

    mu_mode='sparse_proxy' is an intentionally non-geometric approximation
    to model mostly-zero coefficients; it is NOT the GSO of the generated
    intrel basis. Exact mu is typically dense.
    """
    _check(n, precision, seed, delta, eta, condition)
    if not isinstance(bits, int) or bits < 1:
        raise ValueError('bits must be a positive integer')
    if mu_mode not in ('exact', 'sparse_proxy'):
        raise ValueError("mu_mode must be 'exact' or 'sparse_proxy'")
    if not (0 <= zero_probability <= 1):
        raise ValueError('zero_probability must be in [0,1]')
    rng = random.Random(seed)
    # Mimic independent random integers having at most bits bits.
    a = [rng.getrandbits(bits) * (-1 if rng.getrandbits(1) else 1)
         for _ in range(n)]
    mu = _zero_mu(n)
    with mp.workprec(precision + 32):
        sums = []
        s = mp.mpf(1)
        for v in a:
            s += mp.mpf(v)**2
            sums.append(s)
        r = []
        previous = mp.mpf(1)
        for s in sums:
            r.append(_num(s/previous, precision))
            previous = s
        if mu_mode == 'exact':
            for i in range(1, n):
                ai = mp.mpf(a[i])
                for j in range(i):
                    mu[i][j] = _num(ai * a[j] / sums[j], precision)
        else:
            for i in range(1, n):
                for j in range(i):
                    if rng.random() >= zero_probability:
                        mu[i][j] = repr((rng.random()*2-1)*eta)
    return _make_sim(r, mu, condition=condition, delta=delta, eta=eta,
                     precision=precision, seed=seed)


def _polymul(a, b):
    out = [0]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i+j] += x*y
    return out


def _log_row_sq(poly, scalar, shift, log_x):
    # logsumexp over nonzero coefficients of [scalar * x^shift * poly(x)](X*x).
    values = [mp.log(abs(c)) + (j+shift)*log_x for j, c in enumerate(poly) if c]
    largest = max(values)
    lognorm = largest + mp.log(sum(mp.exp(2*(v-largest)) for v in values))/2
    return 2*(mp.log(scalar) + lognorm)


def initial_coppersmith(degree: int = 2, m: int = 4, t: int = 2,
                         modulus_bits: int = 32, X: int = 8, *,
                         precision: int = 212, seed: int = 42,
                         condition: str = 'lovasz', delta: float = 0.99,
                         eta: float = 0.51, profile_mix: float = 0.50,
                         mu_width: Optional[float] = None):
    """Fast Coppersmith-shaped, determinant-correct *heuristic* GSO state.

    Mimics the bench generator's ordered rows N^(m-i)*x^j*f(x)^i,
    0<=i<m, 0<=j<degree, plus x^j*f(x)^m for 0<=j<t,
    with coefficient scaling x -> X*x and monic degree-d f(root)=0 mod N.

    Uses row log norms and the analytic triangular determinant to estimate
    GSO logs; never materializes a D-by-D integer basis or performs QR.
    profile_mix=0 gives pure squared diagonal-pivot scales and >0 adds
    row-norm information, then shifts all logs to match the exact determinant.
    This is an approximate profile, NOT an exact Coppersmith GSO.
    """
    for v, name in ((degree,'degree'),(m,'m'),(modulus_bits,'modulus_bits'),(X,'X')):
        if not isinstance(v,int) or v < (2 if name == 'X' else 1):
            raise ValueError(f'{name} is invalid')
    if not isinstance(t,int) or t < 0 or t > degree*m:
        raise ValueError('t must be a nonnegative integer <= degree*m')
    D = degree*m+t
    _check(D,precision,seed,delta,eta,condition)
    if not (0 <= profile_mix <= 1):
        raise ValueError('profile_mix must be in [0,1]')
    if mu_width is None:
        mu_width = eta
    if not 0 <= mu_width <= eta:
        raise ValueError('mu_width must belong to [0,eta]')
    rng = random.Random(seed)
    # No integer modulus factorization: generate an odd N of requested size.
    N = (1 << (modulus_bits-1)) | rng.getrandbits(modulus_bits-1) | 1
    root = rng.randrange(1, X)
    f = [0] + [rng.randrange(N) for _ in range(degree-1)] + [1]
    f[0] = (-sum(f[j]*pow(root,j) for j in range(1,degree+1))) % N
    powers = [[1]]
    for _ in range(m):
        powers.append(_polymul(powers[-1], f))
    mu = _zero_mu(D)
    with mp.workprec(precision + 32):
        logN, logX = mp.log(N), mp.log(X)
        diaglogs, rawlogs = [], []
        for i in range(m):
            scalar = N**(m-i)
            poly = powers[i]
            for j in range(degree):
                idx = degree*i+j
                diaglogs.append(2*((m-i)*logN + idx*logX))
                rawlogs.append(_log_row_sq(poly,scalar,j,logX))
        for j in range(t):
            idx = degree*m+j
            diaglogs.append(2*idx*logX)
            rawlogs.append(_log_row_sq(powers[m],1,j,logX))
        # Sum log(r_i) = 2 log(det) exactly at working precision.
        candidate = [(1-profile_mix)*p + profile_mix*v
                     for p,v in zip(diaglogs,rawlogs)]
        adjustment = (sum(candidate)-sum(diaglogs))/D
        r = [_num(mp.exp(v-adjustment),precision) for v in candidate]
    for i in range(1,D):
        for j in range(i):
            mu[i][j] = repr((rng.random()*2-1)*mu_width)
    return _make_sim(r,mu,condition=condition,delta=delta,eta=eta,
                     precision=precision,seed=seed)


# Compatibility aliases / discoverability
make_qary_simulator = initial_qary
make_intrel_simulator = initial_intrel
make_coppersmith_simulator = initial_coppersmith
