"""Deferred numerical-validation hooks, intentionally not implemented (Phase 1 TODO).

The benchmark currently checks that GSO computations succeed and that the
initial basis meets the requested size-reduction bound. The following more
expensive mathematical consistency checks are intentionally postponed.
"""


def verify_determinant_conservation(initial_profile, final_profile, precision):
    """TODO: compare sum log2 r_i at requested MPFR precision."""
    raise NotImplementedError('Deferred by design')


def verify_potential_monotonicity(swap_history, precision):
    """TODO: certify potential change at every simulated/real swap."""
    raise NotImplementedError('Deferred by design')


def verify_terminal_reduction(profile, mu, delta, eta, condition, precision):
    """TODO: check full size-reduction / appropriate terminal inequality."""
    raise NotImplementedError('Deferred by design')


def verify_high_precision_stability(instance, precisions=(106, 212, 424)):
    """TODO: re-run matched lattice at several precisions."""
    raise NotImplementedError('Deferred by design')


def deferred_check_names():
    return [
        'determinant_conservation',
        'potential_monotonicity',
        'terminal_reduction',
        'precision_stability',
    ]
