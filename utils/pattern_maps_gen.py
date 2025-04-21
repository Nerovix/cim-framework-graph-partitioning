# pattern_maps.py  ───────────────────────────────────────────────────────────
"""
Generate four core-ID maps for any square number of cores.

    pattern_maps_64   -> list[4][8][8]
    pattern_maps_256  -> list[4][16][16] ...

Patterns
--------
0 : S-snake          (classic serpentine rows)
1 : two-sides        (left half counts up, right half counts down)
2 : broken-snake     (alternating 2*2 vertical/horizontal mini-snakes)
3 : drunk-snake (*)  (Hilbert curve - only when side is a power of two)

(*)  when side is **not** a power of two, pattern #3 duplicates #2 and
    `UserWarning` is issued so callers know a true drunk path wasn't possible
    for that size.
"""

from __future__ import annotations
from pprint import pformat
from typing import Dict, List, Tuple
import math
import sys, textwrap
import warnings

Grid = List[List[int]]


# ─────────────────────────────────────────────────────────────────────
#  pattern 0  - “S-snake”  (simple serpentine rows)
# ─────────────────────────────────────────────────────────────────────
def _s_snake(side: int) -> Grid:
    g: Grid = []
    for r in range(side):
        row = list(range(r * side, (r + 1) * side))
        if r & 1:
            row.reverse()
        g.append(row)
    return g


# ─────────────────────────────────────────────────────────────────────
#  pattern 1  - “two-sides”  (left half ↑, right half ↓)
# ─────────────────────────────────────────────────────────────────────
def _two_sides(side: int) -> Grid:
    half = side // 2
    g = [[0] * side for _ in range(side)]

    nxt = 0
    for r in range(side):
        cols = range(half - 1, -1, -1) if r % 2 == 0 else range(half)
        for c in cols:
            g[r][c] = nxt
            nxt += 1

    nxt = side * side - 1
    for r in range(side):
        cols = range(half, side) if r % 2 == 0 else range(side - 1, half - 1, -1)
        for c in cols:
            g[r][c] = nxt
            nxt -= 1
    return g


# ─────────────────────────────────────────────────────────────────────
#  pattern 2  - “broken snake of snakes”
#               (alternate vertical / horizontal inside every 2×2 block)
# ─────────────────────────────────────────────────────────────────────
def _broken_snake(side: int) -> Grid:
    if side & 1:
        raise ValueError("broken-snake needs an even side length")
    g: Grid = [[0] * side for _ in range(side)]
    nxt = 0
    for br in range(0, side, 2):
        forward = ((br // 2) & 1) == 0
        col_blocks = range(0, side, 2) if forward else range(side - 2, -1, -2)
        for block_idx, bc in enumerate(col_blocks):
            alt = block_idx & 1
            if forward:
                order = (
                    [(0, 0), (1, 0), (1, 1), (0, 1)]  # vertical first
                    if alt == 0
                    else [(0, 0), (0, 1), (1, 1), (1, 0)]
                )
            else:
                order = (
                    [(0, 1), (1, 1), (1, 0), (0, 0)]
                    if alt == 0
                    else [(0, 1), (0, 0), (1, 0), (1, 1)]
                )
            for dr, dc in order:
                g[br + dr][bc + dc] = nxt
                nxt += 1
    return g


# ─────────────────────────────────────────────────────────────────────
#  pattern 3  - “drunk-snake”  (Hilbert curve for power-of-two sides)
# ─────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Hilbert helper (power-of-two side only)
# ──────────────────────────────────────────────────────────────
def _rot(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
    if ry == 0:
        if rx == 1:
            x, y = n - 1 - x, n - 1 - y
        x, y = y, x
    return x, y


def _hilbert_xy(side_pow2: int, d: int) -> Tuple[int, int]:
    """Map Hilbert index→(row,col) for a *power-of-two* square."""
    x = y = 0
    s = 1
    t = d
    while s < side_pow2:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return y, x        # return (row, col)

# ──────────────────────────────────────────────────────────────
#  Patched-Hilbert for ANY even side ≥ 4
# ──────────────────────────────────────────────────────────────
def _patched_hilbert(side: int) -> Grid:
    """Continuous Hilbert-like path for arbitrary *even* side."""
    if side & 1:
        raise ValueError("patched-Hilbert requires an even side")
    grid: Grid = [[0] * side for _ in range(side)]

    # 1) largest power of two ≤ side  → genuine Hilbert in TL quadrant
    k = 1 << (side.bit_length() - 1)          # e.g. side=12 → k=8
    idx = 0
    for d in range(k * k):
        r, c = _hilbert_xy(k, d)
        grid[r][c] = idx
        idx += 1

    # 2) spill-over columns (k … side-1) for rows 0 … k-1
    extra_cols = side - k
    if extra_cols:
        for ec in range(extra_cols):
            col = k + ec
            # Alternate downward / upward to keep continuity
            rows = range(k) if (ec % 2 == 0) else range(k - 1, -1, -1)
            for r in rows:
                grid[r][col] = idx
                idx += 1

    # 3) residual rows (k … side-1).  Start directly beneath the last filled
    #    cell and snake left↔right, flipping each row.
    last_row, last_col = max(
        ((r, c) for r in range(k) for c in range(side) if grid[r][c] == idx - 1),
        key=lambda rc: rc[0] * side + rc[1],
    )
    # first residual row:
    first_row = k
    if first_row < side:
        # go straight down into (first_row, last_col)
        forward = last_col != 0            # direction for that row
        cols = (
            list(range(last_col, -1, -1)) + list(range(last_col + 1, side))
            if forward
            else list(range(last_col, side)) + list(range(last_col - 1, -1, -1))
        )
        for c in cols:
            grid[first_row][c] = idx
            idx += 1
        # 4) remaining residual rows, simple serpentine
        for r in range(first_row + 1, side):
            cols = range(side) if ((r - first_row) % 2 == 1) else range(side - 1, -1, -1)
            for c in cols:
                grid[r][c] = idx
                idx += 1
    return grid


# public face  ──────────────────────────────────────────────────
def drunk_snake(side: int) -> Grid:
    """Return a continuous drunk-snake for any even `side` (≥4)."""
    if side & 1 or side < 4:
        raise ValueError("side must be an even integer ≥ 4")
    return _patched_hilbert(side)


# convenience: generate all four patterns like earlier
def make_pattern_maps(n_cores: int, pm_path: str) -> Dict[str, Grid]:
    side = int(math.isqrt(n_cores))
    if side * side != n_cores:
        raise ValueError(f"{n_cores} is not a perfect square")

    patt0 = _s_snake(side)
    patt1 = _two_sides(side)
    patt2 = _broken_snake(side)
    patt3 = drunk_snake(side)

    globals()[f"pattern_maps_{n_cores}"] = [patt0, patt1, patt2, patt3]
    # write them to a file
    with open(pm_path, "w") as f:
        f.write(f"# auto-generated — commit to repo so it needn't be rebuilt at runtime\n")
        f.write(f"pattern_maps_{n_cores} = \\\n")
        f.write(f"{textwrap.indent(pformat([patt0, patt1, patt2, patt3], width=100, compact=False), '    ')}\n")

    return {"s_snake": patt0, "two_sides": patt1, "broken_snake": patt2, "drunk_snake": patt3}


# ──────────────────────────────────────────────────────────────────────────
#  CLI helper:  python pattern_maps.py 256 > pattern256.py
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python pattern_maps.py <square-core-count>", file=sys.stderr)
        sys.exit(1)

    N   = int(sys.argv[1])
    var = f"pattern_maps_{N}"
    make_pattern_maps(N)
    src = (
        "# auto-generated — commit to repo so it needn't be rebuilt at runtime\n"
        f"{var} = \\\n"
        f"{textwrap.indent(pformat(globals()[var], width=100, compact=False), '    ')}\n"
    )
    print(src)