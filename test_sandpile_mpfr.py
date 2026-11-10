"""Run with Sage/fpylll Python after building the extension."""
from sandpile_mpfr import SandpileSeq
from fpylll import IntegerMatrix, FPLLL, GSO
from sage.all import RealField, matrix


def test_conditions():
    R = RealField(212)
    mat = matrix(R, 2, 2, 0)
    mat[1, 0] = R('0.5')
    x = (['100', '70'], mat)
    a = SandpileSeq(x, precision=212, condition='lovasz', mubnd='0', seed=1)
    b = SandpileSeq(x, precision=212, condition='siegel', mubnd='0', seed=1)
    a.run()
    b.run()
    assert a.swaps == 0
    assert b.swaps == 1
    assert R(b.snapshot()['mu'][1][0]) == 0  # resampled when mubnd=0


def test_full_matrix_and_two_copies():
    mu = [['0', '0', '0'], ['0.2', '0', '0'],
          ['0.123456789123456789123456789', '0.1', '0']]
    initial = SandpileSeq((['100', '100', '100'], mu), precision=212)
    a = SandpileSeq((['100', '100', '100'], mu), precision=212, seed=1)
    b = SandpileSeq((['100', '100', '100'], mu), precision=212, seed=2)
    a.run()
    assert a.swaps == 0
    assert a.snapshot()['mu'] == initial.snapshot()['mu']
    assert b.swaps == 0  # unaffected by a.run()


def test_high_precision_from_fpylll():
    FPLLL.set_precision(212)
    B = IntegerMatrix(2, 2)
    B[0, 0], B[0, 1] = 3, 0
    B[1, 0], B[1, 1] = 1, 1
    M = GSO.Mat(B, float_type='mpfr')
    assert M.update_gso()
    a = SandpileSeq(M, condition='lovasz', precision=212)
    b = SandpileSeq(M, condition='siegel', precision=212)
    R = RealField(212)
    exact = R(a.snapshot()['mu'][1][0])
    assert abs(exact - R(1)/3) < R(2)**(-200)
    assert exact != R(M.get_mu(1, 0))  # public accessor rounds to double
    assert a.snapshot()['mu'] == b.snapshot()['mu']
    a.run()
    assert b.swaps == 0


if __name__ == '__main__':
    test_conditions()
    test_full_matrix_and_two_copies()
    test_high_precision_from_fpylll()
    print('All tests passed')
