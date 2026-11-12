"""Quick tests: run in the Sage environment after building sandpile_mpfr."""
from mpmath import mp
from fast_initial_profiles import initial_qary, initial_intrel, initial_coppersmith


def test_qary():
    s = initial_qary(24, 9, 13.5, precision=212, seed=11)
    snap = s.snapshot()
    r, mu = snap['r'], snap['mu']
    with mp.workprec(200):
        for x in r[:9]:
            assert abs(mp.log(mp.mpf(x),2) - 27) < mp.mpf('1e-40')
        assert all(mp.mpf(x) == 1 for x in r[9:])
    assert all(mp.mpf(mu[i][j]) == 0 for i in range(9) for j in range(i))
    assert all(mp.mpf(mu[i][j]) == 0 for i in range(9,24) for j in range(9,i))


def test_intrel():
    s = initial_intrel(20, 1024, precision=212, seed=11)
    snap = s.snapshot()
    with mp.workprec(160):
        product = mp.mpf(1)
        for x in snap['r']:
            assert mp.mpf(x) > 0
            product *= mp.mpf(x)
        assert product > 1
    assert any(abs(float(snap['mu'][i][j])) > 0.01 for i in range(1,20) for j in range(i))


def test_coppersmith():
    d,m,t,X = 2,4,2,8
    s = initial_coppersmith(d,m,t,32,X,precision=212,seed=11)
    snap = s.snapshot()
    assert len(snap['r']) == d*m+t
    assert all(mp.mpf(x)>0 for x in snap['r'])
    assert any(mp.mpf(snap['mu'][i][j]) != 0 for i in range(1,d*m+t) for j in range(i))


if __name__ == '__main__':
    test_qary()
    test_intrel()
    test_coppersmith()
    print('All fast-initial-profile smoke tests passed.')
