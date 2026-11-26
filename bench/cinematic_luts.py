from __future__ import annotations

import numpy as np

from lut_match.color import linear_to_oklab, linear_to_srgb, oklab_to_linear, srgb_to_linear
from lut_match.io import identity_lut


def _s_curve(x: np.ndarray, contrast: float = 0.0, black_lift: float = 0.0, white_ceil: float = 1.0) -> np.ndarray:
    c = np.clip(x, 0.0, 1.0)
    if contrast != 0.0:
        # Smooth cubic S-curve around 0.5
        s = c - contrast * np.sin(2.0 * np.pi * c) / (2.0 * np.pi)
    else:
        s = c
    return np.clip(black_lift + (white_ceil - black_lift) * s, 0.0, 1.0)


def _split_tone_oklab(
    rgb: np.ndarray,
    shadow_ab: tuple[float, float],
    mid_ab: tuple[float, float],
    high_ab: tuple[float, float],
    sat_scale: float = 1.0,
    contrast: float = 0.0,
    black_lift: float = 0.0,
    white_ceil: float = 1.0,
    hue_rot_deg: float = 0.0,
) -> np.ndarray:
    """Apply smooth analytical cinematic grading in sRGB + OKLab."""
    toned = _s_curve(rgb, contrast=contrast, black_lift=black_lift, white_ceil=white_ceil)
    lab = linear_to_oklab(srgb_to_linear(toned))
    l = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    w_sh = np.exp(-0.5 * ((l - 0.18) / 0.22) ** 2)
    w_md = np.exp(-0.5 * ((l - 0.52) / 0.24) ** 2)
    w_hi = np.exp(-0.5 * ((l - 0.85) / 0.22) ** 2)
    w_sum = w_sh + w_md + w_hi + 1e-12
    da = (w_sh * shadow_ab[0] + w_md * mid_ab[0] + w_hi * high_ab[0]) / w_sum
    db = (w_sh * shadow_ab[1] + w_md * mid_ab[1] + w_hi * high_ab[1]) / w_sum

    if hue_rot_deg != 0.0:
        rad = np.radians(hue_rot_deg)
        ca, sa = np.cos(rad), np.sin(rad)
        a_rot = ca * a - sa * b
        b_rot = sa * a + ca * b
        a, b = a_rot, b_rot

    a_out = a * sat_scale + da
    b_out = b * sat_scale + db
    out_lab = np.stack([l, a_out, b_out], axis=-1)
    return np.clip(linear_to_srgb(np.clip(oklab_to_linear(out_lab), 0.0, 1.0)), 0.0, 1.0)


def _teal_orange_lut(grid: np.ndarray) -> np.ndarray:
    graded = _split_tone_oklab(
        grid,
        shadow_ab=(-0.030, -0.025),
        mid_ab=(0.008, 0.014),
        high_ab=(0.024, 0.032),
        sat_scale=1.10,
        contrast=0.22,
        black_lift=0.015,
        white_ceil=0.985,
    )
    return graded


def _bleach_bypass_lut(grid: np.ndarray) -> np.ndarray:
    luma = 0.2126 * grid[..., 0] + 0.7152 * grid[..., 1] + 0.0722 * grid[..., 2]
    desat = 0.42 * grid + 0.58 * luma[..., None]
    return _split_tone_oklab(
        desat,
        shadow_ab=(-0.006, -0.010),
        mid_ab=(0.000, 0.002),
        high_ab=(0.004, 0.008),
        sat_scale=0.88,
        contrast=0.38,
        black_lift=0.01,
        white_ceil=0.99,
    )


def _cross_process_lut(grid: np.ndarray) -> np.ndarray:
    r = _s_curve(grid[..., 0], contrast=0.28, black_lift=0.02, white_ceil=0.96)
    g = _s_curve(grid[..., 1], contrast=0.16, black_lift=0.01, white_ceil=0.99)
    b = _s_curve(grid[..., 2], contrast=-0.10, black_lift=0.06, white_ceil=0.92)
    base = np.stack([r, g, b], axis=-1)
    return _split_tone_oklab(
        base,
        shadow_ab=(-0.018, 0.014),
        mid_ab=(-0.008, 0.010),
        high_ab=(0.018, 0.022),
        sat_scale=1.08,
    )


def build_cinematic_luts(size: int = 33) -> dict[str, np.ndarray]:
    """Generate 15 distinct, smooth, realistic cinematic 3D LUTs of shape (size, size, size, 3)."""
    g = identity_lut(size)
    return {
        "kodak_2383_print": _split_tone_oklab(
            g, shadow_ab=(-0.014, -0.012), mid_ab=(0.006, 0.012), high_ab=(0.010, 0.022),
            sat_scale=1.08, contrast=0.26, black_lift=0.012, white_ceil=0.985,
        ),
        "fuji_eterna_250d": _split_tone_oklab(
            g, shadow_ab=(-0.016, 0.004), mid_ab=(-0.005, 0.008), high_ab=(0.008, 0.016),
            sat_scale=0.94, contrast=0.12, black_lift=0.025, white_ceil=0.975,
        ),
        "teal_and_orange": _teal_orange_lut(g),
        "bleach_bypass": _bleach_bypass_lut(g),
        "golden_hour_warm": _split_tone_oklab(
            g, shadow_ab=(0.008, 0.018), mid_ab=(0.018, 0.030), high_ab=(0.020, 0.036),
            sat_scale=1.06, contrast=0.14, black_lift=0.010, white_ceil=0.990,
        ),
        "cool_nocturne": _split_tone_oklab(
            g, shadow_ab=(-0.020, -0.032), mid_ab=(-0.015, -0.022), high_ab=(-0.005, -0.010),
            sat_scale=0.95, contrast=0.22, black_lift=0.008, white_ceil=0.960,
        ),
        "vintage_faded_film": _split_tone_oklab(
            g, shadow_ab=(0.010, 0.006), mid_ab=(0.008, 0.016), high_ab=(0.012, 0.018),
            sat_scale=0.84, contrast=-0.06, black_lift=0.065, white_ceil=0.940,
        ),
        "cross_process": _cross_process_lut(g),
        "emerald_shadows": _split_tone_oklab(
            g, shadow_ab=(-0.028, 0.012), mid_ab=(-0.010, 0.008), high_ab=(0.012, 0.020),
            sat_scale=1.04, contrast=0.20, black_lift=0.015, white_ceil=0.980,
        ),
        "warm_tungsten": _split_tone_oklab(
            g, shadow_ab=(0.012, 0.014), mid_ab=(0.020, 0.026), high_ab=(0.016, 0.024),
            sat_scale=1.02, contrast=0.16, black_lift=0.012, white_ceil=0.985,
        ),
        "subtle_pastel_matte": _split_tone_oklab(
            g, shadow_ab=(0.006, -0.008), mid_ab=(0.008, 0.008), high_ab=(0.012, 0.010),
            sat_scale=0.78, contrast=-0.10, black_lift=0.055, white_ceil=0.950,
        ),
        "high_contrast_cinema": _split_tone_oklab(
            g, shadow_ab=(-0.012, -0.016), mid_ab=(0.004, 0.006), high_ab=(0.010, 0.014),
            sat_scale=1.12, contrast=0.36, black_lift=0.004, white_ceil=0.996,
        ),
        "cyberpunk_split": _split_tone_oklab(
            g, shadow_ab=(-0.028, -0.034), mid_ab=(0.016, -0.016), high_ab=(0.032, -0.010),
            sat_scale=1.18, contrast=0.24, black_lift=0.010, white_ceil=0.985, hue_rot_deg=-6.0,
        ),
        "autumn_earth": _split_tone_oklab(
            g, shadow_ab=(0.010, 0.012), mid_ab=(0.016, 0.028), high_ab=(0.014, 0.024),
            sat_scale=0.96, contrast=0.18, black_lift=0.020, white_ceil=0.975, hue_rot_deg=5.0,
        ),
        "soft_agfa_vista": _split_tone_oklab(
            g, shadow_ab=(-0.008, 0.010), mid_ab=(0.010, 0.014), high_ab=(0.012, 0.016),
            sat_scale=0.92, contrast=0.08, black_lift=0.032, white_ceil=0.965,
        ),
    }
