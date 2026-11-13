"""Stateful tour schedulers for block LLL-SP.

A tour freezes its full, ordered sequence of half-open windows before any
reduction. Windows MAY overlap. ``parallelizable`` is an explicit declaration,
not a property inferred by the executor. No concurrent execution is implemented.

All profiles supplied to schedulers are x[i] = ln(||b_i^*||).
The Flatter and BLASter variants are *schedule models*, not faithful copies
of the corresponding complete algorithms. In particular BLASter's actual
segmentation sequence has not been transplanted here.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Callable, Optional


@dataclass(frozen=True)
class Window:
    start: int
    stop: int
    phase: int = 0
    label: str = ""

    @property
    def beta(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class Tour:
    number: int                  # 0-based tour number
    windows: tuple[Window, ...] # complete, immutable schedule for this tour
    parallelizable: bool

    def __iter__(self):
        return iter(self.windows)


def _potential(x):
    """Log potential on a window, x[i] = ln ||b_i^*||."""
    m = len(x)
    return math.fsum((m-k)*float(v) for k, v in enumerate(x))


def _bounds(x, delta, eta):
    """Unmodified nominal bounds from the supplied greedy scheduler."""
    m = len(x)
    if m < 2:
        return 0., 0.
    c = -.5*math.log(delta-eta*eta)
    initial = _potential(x)
    logdet = math.fsum(float(v) for v in x)
    max_final = .5*(m+1)*logdet + m*(m*m-1)*c/12
    margin = 1e-10*(1+abs(initial)+abs(max_final))
    return (max(0., initial-max_final-margin),
            max(0., initial-_potential(sorted(x))+margin))


def _validate(n: int, beta: int | None = None):
    if not isinstance(n, int) or n < 2:
        raise ValueError('n must be >= 2')
    if beta is not None and (not isinstance(beta, int) or not 2 <= beta <= n):
        raise ValueError('beta must be an integer in [2,n]')


def _check_profile(profile, n):
    if len(profile) != n or not all(math.isfinite(float(v)) for v in profile):
        raise ValueError('expected n finite log GS norms')


class Scheduler:
    """Stateful scheduler with complete tour plans and explicit stopping rule.

    A single plan is outstanding at most once. The driver calls observe() once
    per executed block, and finish_tour() exactly once for each planned tour.
    A tour may finish early (e.g. user-defined stop): its unexecuted blocks
    remain unexecuted, and the partial tour is recorded explicitly.
    """
    parallelizable = False

    def __init__(self, n: int, *, max_tours: int | None = 1,
                 max_blocks: int | None = None, parallelizable: bool = False,
                 stop: Optional[Callable[['Scheduler', object], bool]] = None):
        _validate(n)
        if max_tours is not None and max_tours < 1:
            raise ValueError('max_tours must be positive or None')
        if max_blocks is not None and max_blocks < 1:
            raise ValueError('max_blocks must be positive or None')
        self.n, self.max_tours, self.max_blocks, self.stop = n, max_tours, max_blocks, stop
        self.parallelizable = bool(parallelizable)
        self.reset()

    def reset(self):
        self.tour = 0  # completed tours (including early-completed partial tours)
        self.block_reductions = 0
        self.total_swaps = 0
        self.last_tour_swaps = None
        self.last_tour_complete = None
        self._tour_swaps = 0
        self._tour_blocks = 0
        self._active: Tour | None = None
        self._stopped = False

    def _make_windows(self, profile) -> tuple[Window, ...]:
        raise NotImplementedError

    def _stopping_condition(self, profile) -> bool:
        return False

    def should_stop(self, profile) -> bool:
        _check_profile(profile, self.n)
        if self._stopped:
            return True
        if self.max_blocks is not None and self.block_reductions >= self.max_blocks:
            return True
        if self.max_tours is not None and self.tour >= self.max_tours:
            return True
        if self.stop is not None and self.stop(self, profile):
            return True
        return self._stopping_condition(profile)

    def next_tour(self, profile) -> Tour | None:
        if self._active is not None:
            raise RuntimeError('finish the active tour before requesting another')
        if self.should_stop(profile):
            return None
        windows = tuple(self._make_windows(profile))
        if not windows:
            self._stopped = True
            return None
        for w in windows:
            if not 0 <= w.start < w.stop <= self.n or w.beta < 2:
                raise ValueError(f'invalid scheduled window: {w}')
        ordered = sorted(windows, key=lambda w: w.start)
        if self.parallelizable and any(a.stop > b.start
                for a, b in zip(ordered, ordered[1:])):
            raise ValueError('parallelizable tour contains overlapping windows')
        self._active = Tour(self.tour, windows, self.parallelizable)
        self._tour_swaps = 0
        self._tour_blocks = 0
        return self._active

    def observe(self, swaps: int):
        if self._active is None:
            raise RuntimeError('no active tour')
        if not isinstance(swaps, int) or swaps < 0:
            raise ValueError('swaps must be a nonnegative integer')
        if self._tour_blocks >= len(self._active.windows):
            raise RuntimeError('too many observations for this tour')
        self._tour_blocks += 1
        self._tour_swaps += swaps
        self.block_reductions += 1
        self.total_swaps += swaps

    def finish_tour(self, *, completed: bool = True):
        if self._active is None:
            raise RuntimeError('no active tour')
        if completed and self._tour_blocks != len(self._active.windows):
            raise RuntimeError('tour declared complete with unobserved windows')
        self.last_tour_swaps = self._tour_swaps
        self.last_tour_complete = completed
        self.tour += 1
        self._active = None
        if not completed:
            self._stopped = True


class BKZScheduler(Scheduler):
    def __init__(self, n, beta, *, jump=1, **kwargs):
        _validate(n, beta)
        if jump < 1:
            raise ValueError('jump must be >=1')
        self.beta, self.jump = beta, jump
        super().__init__(n, **kwargs)

    def _make_windows(self, profile):
        starts = list(range(0, self.n-1, self.jump))
        if starts[-1] != self.n-2:
            starts.append(self.n-2)
        return tuple(Window(i, min(i+self.beta, self.n), label='bkz') for i in starts)


class DyadicScheduler(Scheduler):
    """Centered dyadic schedule: ONE scale / up to THREE blocks per tour.

    At 0-based tour t the window width is min(n, 2**(t+1)), and the
    displacement to either side is 2**t. The center is fixed from the
    largest INITIAL GS drop, rather than recomputed between tours.

    The ordered windows are centered, left-shifted, right-shifted.
    Near boundaries, duplicates are omitted; at the final, full-width
    scale only one distinct window exists. This is not a global sweep.
    """
    def __init__(self, n, *, max_levels=None, **kwargs):
        if max_levels is not None and max_levels < 1:
            raise ValueError('max_levels must be positive')
        self.max_levels = max_levels
        self.center = None
        super().__init__(n, **kwargs)

    def reset(self):
        super().reset()
        self.center = None

    def _stopping_condition(self, profile):
        if self.max_levels is not None and self.tour >= self.max_levels:
            return True
        # The preceding tour already covered the whole basis.
        return self.tour > 0 and (2 << (self.tour - 1)) >= self.n

    def _make_windows(self, profile):
        if self.center is None:
            self.center = max(
                range(self.n - 1),
                key=lambda i: float(profile[i]) - float(profile[i + 1])
            ) + 1

        t = self.tour
        radius = 1 << t
        width = min(self.n, 2 * radius)
        max_start = self.n - width
        centered = max(0, min(max_start, self.center - radius))
        starts = (
            centered,
            max(0, centered - radius),
            min(max_start, centered + radius),
        )
        windows = []
        for name, start in zip(('center', 'left', 'right'), starts):
            if start not in {w.start for w in windows}:
                windows.append(Window(start, start + width,
                                      phase=t, label='dyadic-' + name))
        return tuple(windows)


class FlatterScheduler(Scheduler):
    """Recursive-overlap ordering proxy; not actual Flatter compression."""
    def __init__(self, n, *, leaf_size=32, **kwargs):
        _validate(n)
        if not 2 <= leaf_size <= n:
            raise ValueError('invalid leaf size')
        self.leaf_size = leaf_size
        super().__init__(n, **kwargs)

    def _make_windows(self, profile):
        out = []
        def visit(a, b, depth):
            length = b-a
            if length <= self.leaf_size:
                out.append(Window(a,b,depth,'flatter-leaf'))
                return
            child = max(2, math.ceil(2*length/3))
            if child >= length:
                out.append(Window(a,b,depth,'flatter-leaf'))
                return
            visit(a,a+child,depth+1)
            visit(b-child,b,depth+1)
            out.append(Window(a,b,depth,'flatter-merge'))
        visit(0,self.n,0)
        return tuple(out)


class BlasterScheduler(Scheduler):
    """Fixed-blocksize segmented sweep PROXY, no overlap parameter.

    This is not the real BLASter sequence; replace _make_windows with the
    published scheduling function when integrating that implementation.
    """
    def __init__(self, n, beta, **kwargs):
        _validate(n,beta)
        self.beta = beta
        super().__init__(n, **kwargs)

    def _make_windows(self, profile):
        # Alternate partitions, then one final truncated tail block to cover
        # boundary pairs. Overlap is NOT a configurable parameter.
        out = []
        for offset in (0, self.beta//2):
            for a in range(offset, self.n-1, self.beta):
                if min(a+self.beta,self.n)-a >= 2:
                    out.append(Window(a,min(a+self.beta,self.n),label='blaster-proxy'))
        if not any(w.start <= self.n-2 and w.stop == self.n for w in out):
            out.append(Window(self.n-2,self.n,label='blaster-proxy-tail'))
        return tuple(out)


class GreedyScheduler(Scheduler):
    """One window per tour, recomputed from the current profile.

    metric=0 (lower) stops when all lower bounds vanish. metric=1 (upper)
    MUST use an explicit finite max_tours or max_blocks; upper bounds need not
    vanish, even when no useful reductions remain.
    """
    def __init__(self,n,beta,*,delta=.99,eta=.51,metric=0,stride=1,
                 max_tours=None,max_blocks=None,stop=None,parallelizable=False):
        _validate(n,beta)
        if metric not in (0,1) or stride < 1 or not 0 < eta or not eta*eta < delta <= 1:
            raise ValueError('invalid greedy parameters')
        if metric == 1 and max_tours is None and max_blocks is None and stop is None:
            raise ValueError('upper-bound greedy needs a stopping budget/predicate')
        self.beta,self.delta,self.eta,self.metric,self.stride = beta,delta,eta,metric,stride
        self.last_score = 0.0
        super().__init__(n,max_tours=max_tours,max_blocks=max_blocks,
                         stop=stop,parallelizable=parallelizable)

    def reset(self):
        super().reset()
        self.last_score = 0.0

    def _best(self,profile):
        starts = list(range(0,self.n-self.beta+1,self.stride))
        if starts[-1] != self.n-self.beta:
            starts.append(self.n-self.beta)
        score, start = 0., None
        for a in starts:
            val = _bounds(profile[a:a+self.beta],self.delta,self.eta)[self.metric]
            if val > score:
                score,start = val,a
        return score,start

    def _stopping_condition(self,profile):
        if self.metric == 0:
            self.last_score, _ = self._best(profile)
            return self.last_score <= 0
        return False

    def _make_windows(self,profile):
        self.last_score, a = self._best(profile)
        if a is None:
            return ()
        return (Window(a,a+self.beta,label='greedy'),)


def bkz_like(n,beta,tours=1,*,stride=1):
    return BKZScheduler(n,beta,jump=stride,max_tours=tours)


def flatter_like(n,*,leaf_size=32,tours=1):
    return FlatterScheduler(n,leaf_size=leaf_size,max_tours=tours)


def blaster_like(n,beta,tours=1):
    return BlasterScheduler(n,beta,max_tours=tours)


def make_scheduler(name,n,**kwargs):
    implementations = {'bkz':BKZScheduler,'dyadic':DyadicScheduler,
                       'flatter':FlatterScheduler,'blaster':BlasterScheduler,
                       'greedy':GreedyScheduler}
    if name not in implementations:
        raise ValueError(f'unknown scheduler {name!r}; choices={tuple(implementations)}')
    return implementations[name](n,**kwargs)
