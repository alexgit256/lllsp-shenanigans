"""Native dd/qd smoke and fidelity tests; requires rebuilt extension and QD-enabled fpylll."""
import unittest
from decimal import Decimal, localcontext

from fpylll import GSO, IntegerMatrix
from sandpile_mpfr import SandpileSeq


class QDBackendTests(unittest.TestCase):
    def test_array_low_bits_survive(self):
        with localcontext() as ctx:
            ctx.prec = 110
            v = Decimal(1)/2 + Decimal(2) ** -90
            text = str(v)
            for typ in ('dd', 'qd'):
                with self.subTest(typ=typ):
                    s = SandpileSeq((['1', '0.8'], [['0','0'],[text,'0']]),
                                   fptype=typ, condition='siegel', seed=4)
                    self.assertNotEqual(Decimal(s.mu_matrix()[1][0]), Decimal('0.5'))
                    self.assertLess(abs(Decimal(s.mu_matrix()[1][0])-v), Decimal(2)**-95)

    def test_matgso_native_import(self):
        B = IntegerMatrix.from_matrix([[31, 0, 0], [15, 1, 0], [7, 0, 1]])
        for typ in ('dd', 'qd'):
            with self.subTest(typ=typ):
                M = GSO.Mat(B, float_type=typ)
                self.assertTrue(M.update_gso())
                s = SandpileSeq(M, fptype=typ, seed=42, trace_swaps=True)
                self.assertEqual(s.n, 3)
                self.assertEqual(s.fptype, typ)
                self.assertEqual(len(s.profile()), 3)
                self.assertEqual(len(s.mu_matrix()), 3)
                s.run(max_swaps=10000)
                self.assertEqual(len(s.valid_swap_mu()), s.swaps)
                self.assertEqual(s.snapshot()['fptype'], typ)

    def test_block_run_and_external_resample(self):
        for typ in ('dd', 'qd'):
            with self.subTest(typ=typ):
                r = ['4','1','2','3']
                mu = [['0']*4 for _ in range(4)]
                s = SandpileSeq((r, mu), fptype=typ, seed=17,
                               condition='siegel', trace_swaps=True)
                s.run(max_swaps=10000, start=0, kappa=3)
                self.assertEqual(Decimal(s.profile()[3]), Decimal('3'))
                before = s.mu_matrix()
                s.resample_external_mu(start=0, kappa=3)
                after = s.mu_matrix()
                self.assertEqual(before[2], after[2])
                self.assertEqual(before[3][3], after[3][3])


if __name__ == '__main__':
    unittest.main()
