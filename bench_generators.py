"""Deterministic, sequential lattice generation. Sage/fpylll imports are lazy.

NTRU follows the user's demo.ipynb (AlexGit256/lllsp-shenanigans):
  Rq=(Z/qZ)[x]/(x^n-1), h=g/f (mod q), row basis [0,qI; I,H].
"""
from __future__ import annotations
import gzip
import json
from pathlib import Path


def write_json_gz(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as fd:
        json.dump(obj, fd, separators=(',', ':'), ensure_ascii=True)


def read_json_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as fd:
        return json.load(fd)


def matrix_to_rows(B):
    n = B.nrows() if callable(getattr(B, 'nrows', None)) else B.nrows
    m = B.ncols() if callable(getattr(B, 'ncols', None)) else B.ncols
    return [[str(B[i, j]) for j in range(m)] for i in range(n)]


def from_rows(rows):
    from fpylll import IntegerMatrix
    # Avoid inferring unusual integer types / losing large integers.
    return IntegerMatrix.from_matrix([[int(x) for x in row] for row in rows], int_type='mpz')


def _ntru(spec):
    from sage.all import (ZZ, Zmod, PolynomialRing, matrix,
                          identity_matrix, zero_matrix, block_matrix)
    from sage.stats.distributions.discrete_gaussian_integer import (
        DiscreteGaussianDistributionIntegerSampler,
    )
    n = int(spec['degree'])
    q = int(spec['q'])
    sigma_f = float(spec.get('sigma_f', spec.get('sigma', 1.0)))
    sigma_g = float(spec.get('sigma_g', spec.get('sigma', 1.0)))
    Df = DiscreteGaussianDistributionIntegerSampler(sigma=sigma_f)
    Dg = DiscreteGaussianDistributionIntegerSampler(sigma=sigma_g)
    PZ = PolynomialRing(ZZ, 'X')
    Pq = PolynomialRing(Zmod(q), 'x')
    x = Pq.gen()
    Rq = Pq.quotient(x**n - 1, names=('xx',))
    xx = Rq.gen()
    attempts = int(spec.get('max_inversion_attempts', 10000))
    for _ in range(attempts):
        fZ = PZ([ZZ(Df()) for _ in range(n)])
        f = Rq(Pq([fZ[i] for i in range(n)]))
        try:
            f_inv = f**(-1)
            break
        except (ZeroDivisionError, ArithmeticError, ValueError):
            continue
    else:
        raise RuntimeError('failed to sample invertible NTRU f')
    gZ = PZ([ZZ(Dg()) for _ in range(n)])
    h = Rq(Pq([gZ[i] for i in range(n)])) * f_inv
    # The i-th row is the coefficient vector of x^i*h (mod x^n-1).
    # Using cyclic rotations rather than evaluating xx**i*h repeatedly.
    coeffs = [ZZ(h[j]) for j in range(n)]
    def center(t):
        a = ZZ(t) % q
        return a if a <= q//2 else a-q
    H = matrix(ZZ, [[center(coeffs[(j-i) % n]) for j in range(n)]
                    for i in range(n)])
    I = identity_matrix(ZZ, n)
    Z = zero_matrix(ZZ, n)
    B = block_matrix(ZZ, [[Z, q*I], [I, H]])
    return B, {'ring': 'Zq[x]/(x^n-1)', 'ntru_degree': n,
               'secret_f': [str(fZ[i]) for i in range(n)],
               'secret_g': [str(gZ[i]) for i in range(n)]}


def _coppersmith(spec, seed):
    """Monic univariate modular small-root shift lattice (not a generic NTRU).

    Rows: x^j*N^(m-i)*f(x)^i, 0<=i<m, 0<=j<degree;
          x^j*f(x)^m, 0<=j<t.
    Columns are coefficients of p(X*x), giving size degree*m+t.
    """
    import random
    from sage.all import ZZ, PolynomialRing, matrix, next_prime
    rng = random.Random(seed)
    d = int(spec.get('degree', 2))
    m = int(spec.get('m', 4))
    t = int(spec.get('t', 2))
    bits = int(spec.get('modulus_bits', 32))
    X = int(spec.get('X', 8))
    if d < 1 or m < 1 or t < 0 or bits < 8 or X < 2:
        raise ValueError('invalid Coppersmith generator parameters')
    a = ZZ(rng.randrange(1, X))
    b1 = bits // 2
    p = next_prime(ZZ(rng.randrange(2**(b1-1), 2**b1)))
    q = next_prime(ZZ(rng.randrange(2**(bits-b1-1), 2**(bits-b1))))
    N = p*q
    P = PolynomialRing(ZZ, 'x')
    x = P.gen()
    coeffs = [ZZ(0)] + [ZZ(rng.randrange(int(N))) for _ in range(d-1)] + [ZZ(1)]
    coeffs[0] = ZZ(-sum(coeffs[j]*(a**j) for j in range(1, d+1))) % N
    f = P(coeffs)  # monic and f(a)=0 mod N
    D = d*m+t
    polynomials = [N**(m-i) * x**j * f**i for i in range(m) for j in range(d)]
    polynomials += [x**j * f**m for j in range(t)]
    B = matrix(ZZ, [[ZZ(pol[k]) * ZZ(X)**k for k in range(D)]
                    for pol in polynomials])
    return B, {'N': str(N), 'X': str(X), 'root': str(a), 'polynomial': str(f)}


def generate_one(family, spec, seed):
    from fpylll import FPLLL, IntegerMatrix
    from sage.all import set_random_seed
    set_random_seed(int(seed))
    FPLLL.set_random_seed(int(seed))
    if family == 'ntru':
        B, meta = _ntru(spec)
    elif family == 'qary':
        n = int(spec['dimension'])
        k = int(spec.get('k', n//2))
        qargs = {'q': int(spec['q'])} if 'q' in spec else {'bits': int(spec['bits'])}
        B = IntegerMatrix.random(n, 'qary', k=k, **qargs)
        meta = {'k': k}
    elif family == 'intrel':
        n = int(spec['dimension'])
        B = IntegerMatrix.random(n, 'intrel', bits=int(spec['bits']))
        meta = {'ambient_dimension': n+1}
    elif family == 'coppersmith':
        B, meta = _coppersmith(spec, seed)
    else:
        raise ValueError(f'unknown lattice family {family!r}')
    return B, meta

def matrix_shape(B):
    nrows = B.nrows
    ncols = B.ncols
    return (
        nrows() if callable(nrows) else nrows,
        ncols() if callable(ncols) else ncols,
    )

def generate_instances(config, root, overwrite=False):
    """Generate ALL instances in this process; reduction jobs run separately."""
    root = Path(root)
    target = root/'instances'
    target.mkdir(parents=True, exist_ok=True)
    manifest_file = root/'manifest.json'
    if manifest_file.exists() and not overwrite:
        raise FileExistsError(f'{manifest_file} exists; pass --overwrite to regenerate')
    import random
    rng = random.Random(int(config.get('seed', 42)))
    instances = []
    for family, variants in config['families'].items():
        if isinstance(variants, dict):
            variants = [variants]
        for variant, spec in enumerate(variants):
            for idx in range(int(spec.get('count', 1))):
                seed = rng.randrange(1, 2**31)
                instance_id = f'{family}_v{variant:02d}_{idx:05d}'
                B, meta = generate_one(family, spec, seed)
                rel_path = f'instances/{instance_id}.json.gz'
                write_json_gz(root/rel_path, {
                    'id': instance_id, 'family': family, 'variant': variant,
                    'seed': seed, 'spec': spec, 'meta': meta,
                    'basis': matrix_to_rows(B),
                })
                instances.append({'id': instance_id, 'family': family,
                                  'variant': variant, 'spec': spec,
                                  'seed': seed, 'file': rel_path})
                print(f'[generate] {instance_id}: {matrix_shape(B)}', flush=True)
    manifest = {'schema': 1, 'config': config, 'instances': instances}
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
