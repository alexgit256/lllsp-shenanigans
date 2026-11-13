"""Execute in user's Sage environment after compiling Cython module."""
from sandpile_mpfr import SandpileSeq, gso_mu_entry, gso_swap_is_valid
from fpylll import IntegerMatrix, FPLLL, GSO
from sage.all import RealField, matrix


def test_conditions():
    R=RealField(212)
    mu=matrix(R, 2, 2, 0)
    mu[1,0]=R('0.5')
    data=(['100','69.5'],mu)
    # Lovasz threshold: (0.95-0.25)*100 = 70
    # Siegel threshold: (0.95-0.51^2)*100 = 68.99
    lovasz=SandpileSeq(
        data,
        precision=212,
        condition='lovasz',
        delta=0.95,
        eta=0.51,
        seed=1,
        trace_swaps=True,
    )
    siegel=SandpileSeq(
        data,
        precision=212,
        condition='siegel',
        delta=0.95,
        eta=0.51,
        seed=1,
        trace_swaps=True,
    )
    lovasz.run(max_swaps=100)
    siegel.run(max_swaps=100)
    assert lovasz.swaps >= 1
    assert siegel.swaps == 0
    assert R(lovasz.valid_swap_mu()[0])==R('0.5')
    assert len(lovasz.final_subdiagonal_mu())==1


def test_full_matrix():
    mu=[['0','0','0'],['0.2','0','0'],
        ['0.123456789123456789123456789','0.1','0']]
    data=(['100','100','100'],mu)
    a=SandpileSeq(data,precision=212,seed=1)
    b=SandpileSeq(data,precision=212,seed=2)
    a.run()
    assert a.swaps==0
    assert b.swaps==0
    assert R212(a.mu_matrix()[2][0]) == R212(mu[2][0])


def R212(x):
    return RealField(212)(x)


def test_native_bridge_and_decision():
    FPLLL.set_precision(212)
    B=IntegerMatrix(2,2)
    B[0,0],B[0,1]=3,0
    B[1,0],B[1,1]=1,1
    M=GSO.Mat(B,float_type='mpfr')
    assert M.update_gso()
    source=SandpileSeq(M,condition='lovasz',precision=212)
    exact=R212(gso_mu_entry(M,1,0))
    assert abs(exact - R212(1)/3)<R212(2)**(-200)
    assert exact != R212(M.get_mu(1,0))
    assert source.mu_matrix()[1][0]==gso_mu_entry(M,1,0)
    assert isinstance(gso_swap_is_valid(M,1,0.99,0.51,False),bool)


if __name__=='__main__':
    test_conditions()
    test_full_matrix()
    test_native_bridge_and_decision()
    print('Sandpile MPFR tests passed')
