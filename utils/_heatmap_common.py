"""Shared colormap + norm for the utils/plot_signed_heatmap*.py scripts.

Unlike src/view.py's _make_signed_effect_norm (which fades gradually inside
its significance threshold and uses a saturating gamma curve outside it),
this pair reserves a fixed, visible slice of the colorbar for the
non-significant zone (|value| <= thresh) and renders it as a genuinely flat
grey -- not a fade -- then maps everything outside thresh linearly (no
saturation curve) so real magnitude differences stay visible.

Getting a truly flat colour (not just reserved space) needs both pieces
together: the norm positions |value| <= thresh within a fixed band of
normalized colorbar space, and the colormap has an actual constant-colour
segment across that same band (two control points at the same colour --
LinearSegmentedColormap interpolates two identical colours to a constant).
A norm alone can't do it: imshow/colorbar always compute cmap(norm(value)),
so if norm(value) still varies across the band, a 3-point red-grey-blue
colormap keeps interpolating through it regardless of how the norm is
shaped.

Kept local to these analysis scripts rather than changing the shared
core-pipeline norm/colormap.
"""
import numpy as np
from matplotlib.colors import FuncNorm, LinearSegmentedColormap

# Fraction of the [0,1] colorbar each side of center devoted to the
# non-significant zone -- shared by the norm and the colormap so their
# bands line up exactly. Totals 2x this fraction of the bar's full height
# (0.05 -> 10%).
GREY_HALF_WIDTH = 0.05
_GREY = "#d9d9d9"
_RED = "#b2182b"
_BLUE = "#2166ac"


def make_flat_grey_cmap(grey_half_width: float = GREY_HALF_WIDTH) -> LinearSegmentedColormap:
    GH = grey_half_width
    return LinearSegmentedColormap.from_list(
        "RdGreyBu_flat",
        [(0.0, _RED), (0.5 - GH, _GREY), (0.5 + GH, _GREY), (1.0, _BLUE)],
    )


def make_flat_grey_norm(vmax: float, thresh: float, grey_half_width: float = GREY_HALF_WIDTH) -> FuncNorm:
    GH = grey_half_width
    safe_v = max(vmax - thresh, 1e-8)

    def forward(x):
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)

        pm = x > thresh
        nm = x < -thresh
        dm = ~pm & ~nm

        out[dm] = 0.5 + (x[dm] / thresh) * GH if thresh > 0 else 0.5
        u = np.clip((x[pm] - thresh) / safe_v, 0.0, 1.0)
        out[pm] = (0.5 + GH) + (0.5 - GH) * u
        u = np.clip((-x[nm] - thresh) / safe_v, 0.0, 1.0)
        out[nm] = (0.5 - GH) - (0.5 - GH) * u

        return np.clip(out, 0.0, 1.0)

    def inverse(y):
        y = np.asarray(y, dtype=float)
        out = np.zeros_like(y)

        g_lo, g_hi = 0.5 - GH, 0.5 + GH
        pm = y > g_hi
        nm = y < g_lo
        dm = ~pm & ~nm

        out[dm] = (y[dm] - 0.5) / GH * thresh if GH > 0 else 0.0
        u = np.clip((y[pm] - g_hi) / (1.0 - g_hi), 0.0, 1.0)
        out[pm] = thresh + safe_v * u
        u = np.clip((g_lo - y[nm]) / g_lo, 0.0, 1.0)
        out[nm] = -(thresh + safe_v * u)

        return out

    return FuncNorm((forward, inverse), vmin=-vmax, vmax=vmax)
