"""Sequential block-LLL-SP estimator driven by immutable scheduled tours.

No changes to SandpileSeq and no parallel execution. The ``parallelizable``
field on a scheduler is metadata only; a future executor may use it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from time import perf_counter
from typing import Callable
from lllsp_schedulers import Scheduler, Window


def log_gs_profile(sim):
    """Convert simulator's squared GSO profile to ln(||b_i*||).

    MPFR strings are parsed using mpmath to avoid overflow for huge intrel
    entries; double values use ordinary Python math.log.
    """
    values = sim.profile()
    if sim.fptype == 'double':
        return [0.5*math.log(float(v)) for v in values]
    from mpmath import mp
    with mp.workdps(35):
        return [float(mp.log(mp.mpf(v))/2) for v in values]


@dataclass
class BlockRecord:
    tour: int
    index: int
    window: Window
    swaps: int
    seconds: float


@dataclass
class EstimatorResult:
    reason: str
    tours: int
    block_reductions: int
    swaps: int
    wall_seconds: float
    final_profile: list[float]
    records: list[BlockRecord] = field(default_factory=list)


def run_block_estimator(sim, scheduler: Scheduler, *,
                        max_swaps_per_block: int = 0x400000000,
                        resample_external: bool = True,
                        refresh_every_block: bool = False,
                        stop_after_block: Callable | None = None,
                        record_blocks: bool = False) -> EstimatorResult:
    """Drive a scheduler with repeated block calls to an existing SandpileSeq.

    Freeze complete window sequence at the beginning of each tour. The
    scheduler can stop after a tour, or mid-tour via a block budget or an
    optional stop_after_block callback. No execution starts outside bounds.

    ``stop_after_block(scheduler, sim, profile, record)`` may return True to
    abort the remaining windows. Even when a scheduler advertises
    ``parallelizable=True``, execution here remains sequential.

    When ``resample_external`` is True, call the simulator's separate
    downstream-mu sampler AFTER each block, preserving the existing model.
    """
    if scheduler.n != sim.n:
        raise ValueError('scheduler and simulator dimension mismatch')
    if max_swaps_per_block < 1:
        raise ValueError('max_swaps_per_block must be positive')
    profile = log_gs_profile(sim)
    records = []
    clock = perf_counter()
    reason = 'scheduler_stop'
    while not scheduler.should_stop(profile):
        tour = scheduler.next_tour(profile)
        if tour is None:
            break
        finished = False
        try:
            for index, window in enumerate(tour):
                if scheduler.max_blocks is not None and scheduler.block_reductions >= scheduler.max_blocks:
                    reason = 'block_budget'
                    break
                t0 = perf_counter()
                sim.run(max_swaps=max_swaps_per_block,
                        start=window.start, kappa=window.stop)
                if resample_external:
                    sim.resample_external_mu(start=window.start, kappa=window.stop)
                dt = perf_counter()-t0
                swaps = int(sim.swaps_last_block)
                scheduler.observe(swaps)
                record = BlockRecord(tour.number,index,window,swaps,dt)
                if record_blocks:
                    records.append(record)
                # Adaptive schedulers re-evaluate the profile at the NEXT tour.
                # For fixed tours this refresh is only needed for stop callbacks.
                if refresh_every_block or stop_after_block is not None or scheduler.stop is not None:
                    profile = log_gs_profile(sim)
                if stop_after_block is not None and stop_after_block(scheduler, sim, profile, record):
                    reason = 'callback_stop'
                    break
                if scheduler.stop is not None and scheduler.stop(scheduler, profile):
                    reason = 'scheduler_predicate'
                    break
            else:
                finished = True
        except BaseException:
            scheduler.finish_tour(completed=False)
            raise
        scheduler.finish_tour(completed=finished)
        profile = log_gs_profile(sim)
        if not finished:
            if reason == 'scheduler_stop':
                reason = 'interrupted_tour'
            break
    else:
        reason = 'scheduler_stop'
    return EstimatorResult(reason,scheduler.tour,scheduler.block_reductions,
                           scheduler.total_swaps,perf_counter()-clock,
                           profile,records)
