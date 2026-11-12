"""Profile-driven window selectors for LLL-SP, independent of the reducer.

All windows are half-open [start, stop). Profile x contains ln(||b_i^*||).
Schedulers return a Window or None from next_window(x); call again with the
*updated* profile after performing the reduction externally.

The flatter_like and blaster_like policies model their overlapping/segmented
block *ordering*, not the original algorithms' compression, precision,
parallelism, Seysen reduction, or numerical guarantees.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class Window:
    start: int
    stop: int
    phase: int = 0
    label: str = ""

    @property
    def beta(self) -> int:
        return self.stop - self.start


def _potential(x):
    """Log potential on a window, x[i] = ln ||b_i^*||."""
    m = len(x)
    return math.fsum((m-k)*float(v) for k, v in enumerate(x))


def _bounds(x, delta, eta):
    """Exactly the same nominal bounds as the supplied greedy scheduler."""
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
    if n < 2:
        raise ValueError('n must be at least 2')
    if beta is not None and not 2 <= beta <= n:
        raise ValueError('beta must be between 2 and n')


def _check_profile(x, n):
    if len(x) != n:
        raise ValueError(f'Expected {n} log GS norms, got {len(x)}')
    if not all(math.isfinite(float(t)) for t in x):
        raise ValueError('Log GS norms must be finite')


class PlannedScheduler:
    """Finite, precomputed schedule. next_window(x) validates x for consistency."""
    def __init__(self, n, windows):
        _validate(n)
        self.n = n
        self.windows = tuple(windows)
        for w in self.windows:
            if not (0 <= w.start < w.stop <= n and w.beta >= 2):
                raise ValueError(f'Invalid window: {w}')
        self.cursor = 0

    def reset(self):
        self.cursor = 0

    def next_window(self, x):
        _check_profile(x, self.n)
        if self.cursor >= len(self.windows):
            return None
        w = self.windows[self.cursor]
        self.cursor += 1
        return w


def bkz_like(n: int, beta: int, tours: int = 1, *, stride: int = 1):
    """Left-to-right floating width-beta window per tour, with truncated tail.

    This is a block-LLL *schedule*, not BKZ enumeration.
    """
    _validate(n, beta)
    if tours < 1 or stride < 1:
        raise ValueError('tours and stride must be positive')
    windows = []
    for tour in range(tours):
        starts = list(range(0, n-1, stride))
        if starts[-1] != n-2:
            starts.append(n-2)  # make sure the last pair can be visited
        for a in starts:
            windows.append(Window(a, min(n, a+beta), tour, 'bkz-like'))
    return PlannedScheduler(n, windows)


def _cover(n: int, width: int, overlap: int):
    """Minimum left-to-right coverage using constant-width, overlapping blocks."""
    width = min(n, width)
    if width == n:
        return [(0, n)]
    if not 0 <= overlap < width:
        raise ValueError('overlap must be in [0,width)')
    step = width-overlap
    starts = list(range(0, n-width+1, step))
    if starts[-1] != n-width:
        starts.append(n-width)
    return [(s, s+width) for s in starts]


def blaster_like(n: int, beta: int, tours: int = 1, *, overlap: int | None = None,
                 alternate: bool = True):
    """Prescribed-size overlapping segmented sweeps (BLASter-inspired only).

    No claim to reproduce BLASter's actual segmentation or Seysen/QR steps.
    """
    _validate(n, beta)
    if tours < 1:
        raise ValueError('tours must be positive')
    if overlap is None:
        overlap = beta//2
    blocks = _cover(n, beta, overlap)
    windows = []
    for tour in range(tours):
        ordered = blocks[::-1] if alternate and tour % 2 else blocks
        windows.extend(Window(a, b, tour, 'blaster-like') for a, b in ordered)
    return PlannedScheduler(n, windows)


def flatter_like(n: int, *, leaf_size: int = 32, tours: int = 1):
    """Recursive overlapping half-block order, inspired by Flatter.

    Children are [a,m) and [m-overlap,b), overlapping by roughly half
    of a child; visit children before parent. The original Flatter uses
    iterated compression and is NOT simulated by this ordering alone.
    """
    _validate(n)
    if not 2 <= leaf_size <= n or tours < 1:
        raise ValueError('invalid leaf_size/tours')
    windows = []
    def visit(a, b, depth, tour):
        length = b-a
        if length <= leaf_size:
            windows.append(Window(a,b,tour,f'flatter-leaf-{depth}'))
            return
        # Two half-overlapping blocks of size ceil(2*length/3): both
        # strictly shorter than the parent for length>=4.
        child_size = max(2, math.ceil(2*length/3))
        if child_size >= length:
            windows.append(Window(a,b,tour,f'flatter-leaf-{depth}'))
            return
        visit(a,a+child_size,depth+1,tour)
        visit(b-child_size,b,depth+1,tour)
        windows.append(Window(a,b,tour,f'flatter-merge-{depth}'))
    for tour in range(tours):
        visit(0,n,0,tour)
    return PlannedScheduler(n, windows)


class DyadicScheduler:
    """Centered expanding powers-of-two windows around the largest GS drop.

    At level t (starting at t=0), width is approximately 2**(t+1),
    followed by overlapping windows covering the full domain at that width.
    The center is chosen once from the input profile on the first call.
    This models the 2**t *radius* variant; no final full LLL is implied.
    """
    def __init__(self, n: int, *, max_levels: int | None = None):
        _validate(n)
        self.n = n
        self.max_levels = max_levels
        self.reset()

    def reset(self):
        self._windows = None
        self._pos = 0

    def _plan(self, x):
        center = max(range(self.n-1), key=lambda i: float(x[i])-float(x[i+1])) + 1
        windows = []
        level = 0
        while True:
            radius = 1 << level
            width = min(self.n, 2*radius)
            a = max(0, min(self.n-width, center-radius))
            b = a+width
            # Center first, then shift outward in both directions.
            starts = [a]
            step = max(1, width//2)
            for j in range(1, (self.n//step)+2):
                left, right = a-j*step, a+j*step
                inserted = False
                if left >= 0:
                    starts.append(left); inserted = True
                if right <= self.n-width:
                    starts.append(right); inserted = True
                if not inserted:
                    break
            starts += [0, self.n-width]
            for s in dict.fromkeys(starts):
                windows.append(Window(s, s+width, level, 'dyadic'))
            level += 1
            if width == self.n or (self.max_levels is not None and level >= self.max_levels):
                break
        return tuple(windows)

    def next_window(self, x):
        _check_profile(x, self.n)
        if self._windows is None:
            self._windows = self._plan(x)
        if self._pos >= len(self._windows):
            return None
        w = self._windows[self._pos]
        self._pos += 1
        return w


class GreedyScheduler:
    """Recompute the highest positive bound after every block reduction.

    metric=0 selects the first returned bound; metric=1 selects the second.
    This selector is intentionally stateless: without a positive bound it
    returns None; if a reducer does not improve the profile it may choose
    the same block indefinitely. Set a cap in the outer driver.
    """
    def __init__(self, n: int, beta: int, *, delta: float = 0.99,
                 eta: float = 0.51, metric: int = 0, stride: int = 1):
        _validate(n, beta)
        if metric not in (0,1):
            raise ValueError('metric must be 0 or 1')
        if stride < 1 or not (0 < eta and eta*eta < delta <= 1):
            raise ValueError('invalid stride/delta/eta')
        self.n, self.beta = n, beta
        self.delta, self.eta, self.metric, self.stride = delta, eta, metric, stride
        self.iteration = 0
        self.last_score = 0.0

    def reset(self):
        self.iteration = 0
        self.last_score = 0.0

    def next_window(self, x):
        _check_profile(x, self.n)
        starts = list(range(0,self.n-self.beta+1,self.stride))
        if starts[-1] != self.n-self.beta:
            starts.append(self.n-self.beta)
        best_score, best_a = 0.0, None
        for a in starts:
            score = _bounds(x[a:a+self.beta], self.delta, self.eta)[self.metric]
            if score > best_score:
                best_score, best_a = score, a
        self.last_score = best_score
        if best_a is None:
            return None
        self.iteration += 1
        return Window(best_a,best_a+self.beta,self.iteration,'greedy')


def make_scheduler(name: str, n: int, **kwargs):
    """Convenience factory; all results expose next_window(profile)."""
    choices = {
        'bkz': bkz_like,
        'dyadic': DyadicScheduler,
        'flatter': flatter_like,
        'blaster': blaster_like,
        'greedy': GreedyScheduler,
    }
    if name not in choices:
        raise ValueError(f'unknown scheduler {name!r}; choices={tuple(choices)}')
    return choices[name](n, **kwargs)


if __name__ == '__main__':
    n = 64
    x = [5.0]*32 + [0.0]*32
    configs = {
        'bkz': dict(beta=16,tours=1),
        'dyadic': dict(),
        'flatter': dict(leaf_size=16),
        'blaster': dict(beta=16),
        'greedy': dict(beta=16,metric=0),
    }
    for name, kw in configs.items():
        s = make_scheduler(name, n, **kw)
        samples = []
        for _ in range(10000):
            w = s.next_window(x)
            if w is None:
                break
            samples.append((w.start,w.stop))
            if name == 'greedy':
                break  # reducer must update x before requesting the next window
        print(f'{name:8s} count={len(samples):3d} first={samples[:6]}')
