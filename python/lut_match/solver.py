from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import isotonic_regression

from lut_match.color import apply_idt, oklab_to_srgb, srgb_to_oklab
from lut_match.io import apply_lut, identity_lut

_BAND_CENTERS = np.array([0.14, 0.32, 0.50, 0.68, 0.86], dtype=np.float64)
_BAND_SIGMA = 0.16
_QUANTILES = np.array(
    [0.01, 0.03, 0.07, 0.12, 0.20, 0.30, 0.42, 0.55, 0.68, 0.80, 0.90, 0.96, 0.99],
    dtype=np.float64,
)
_HUE_BINS = 72
_HUE_BIN_CENTERS = -np.pi + (2.0 * np.pi * (np.arange(_HUE_BINS, dtype=np.float64) + 0.5)) / _HUE_BINS
_HUE_ANCHORS = -np.pi + (2.0 * np.pi * (np.arange(24, dtype=np.float64) + 0.5)) / 24.0
_HUE_SIGMA = np.radians(28.0)
_HUE_DIFF_MB = _HUE_BIN_CENTERS[None, :] - _HUE_ANCHORS[:, None]
_HUE_DIFF_MB = np.where(
    _HUE_DIFF_MB > np.pi,
    _HUE_DIFF_MB - 2.0 * np.pi,
    np.where(_HUE_DIFF_MB < -np.pi, _HUE_DIFF_MB + 2.0 * np.pi, _HUE_DIFF_MB),
)
_HUE_KERNEL_MB = np.exp(-(_HUE_DIFF_MB * _HUE_DIFF_MB) / (2.0 * _HUE_SIGMA * _HUE_SIGMA))
_HUE_BASIS = np.column_stack(
    [
        np.cos(_HUE_ANCHORS),
        np.sin(_HUE_ANCHORS),
        np.cos(2.0 * _HUE_ANCHORS),
        np.sin(2.0 * _HUE_ANCHORS),
        np.cos(3.0 * _HUE_ANCHORS),
        np.sin(3.0 * _HUE_ANCHORS),
    ]
)
_HUE_REG_DIAG = np.diag([0.06, 0.06, 0.12, 0.12, 0.24, 0.24])
_PAIRED_REG_DIAG = np.diag(
    [1e-6, 1e-5, 2e-3, 2e-3, 2e-3, 2e-2, 2e-2, 5e-3, 8e-3, 8e-3, 1.5e-2, 1.5e-2, 3e-2, 3e-2]
)
_MODE_CONFIG: dict[str, dict[str, float]] = {
    "balanced": {
        "luma_alpha": 0.35,
        "mkl_alpha": 0.45,
        "shift_alpha": 0.55,
        "shift_max_scale": 0.065,
        "shift_max_base": 0.015,
        "hue_alpha": 0.35,
        "sat_alpha": 0.35,
        "max_hue_rad": float(np.radians(10.0)),
        "max_log_sat": 0.22,
    },
    "creative": {
        "luma_alpha": 0.70,
        "mkl_alpha": 0.75,
        "shift_alpha": 0.75,
        "shift_max_scale": 0.075,
        "shift_max_base": 0.018,
        "hue_alpha": 0.55,
        "sat_alpha": 0.55,
        "max_hue_rad": float(np.radians(14.0)),
        "max_log_sat": 0.28,
    },
}


def _subsample_rgb(rgb: np.ndarray, max_samples: int) -> np.ndarray:
    flat = np.asarray(rgb, dtype=np.float64).reshape((-1, 3))
    n = flat.shape[0]
    if n <= max_samples:
        return flat
    idx = np.linspace(0, n - 1, max_samples, dtype=np.intp)
    return flat[idx]


def _density_weights(lab: np.ndarray) -> np.ndarray:
    """Inverse sublinear color-voxel density weights to cancel large flat-background bias."""
    q_l = np.clip(np.floor(lab[:, 0] / 0.04).astype(np.int64), 0, 25)
    q_a = np.clip(np.floor((lab[:, 1] + 0.40) / 0.025).astype(np.int64), 0, 32)
    q_b = np.clip(np.floor((lab[:, 2] + 0.40) / 0.025).astype(np.int64), 0, 32)
    keys = (q_l * 33 + q_a) * 33 + q_b
    _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return np.power(counts[inv].astype(np.float64), -0.75)


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, probs: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum = np.cumsum(w)
    if cum[-1] <= 0.0:
        return np.quantile(values, probs)
    cdf = (cum - 0.5 * w) / cum[-1]
    return np.interp(probs, cdf, v)


def _fit_luminance_curve(
    l_src: np.ndarray,
    w_src: np.ndarray,
    l_ref: np.ndarray,
    w_ref: np.ndarray,
    shadow_lift: float,
    highlight_rolloff: float,
) -> PchipInterpolator:
    q_src = _weighted_quantiles(l_src, w_src, _QUANTILES)
    q_ref = _weighted_quantiles(l_ref, w_ref, _QUANTILES)

    # Regularize extreme exposure shifts between disparate scenes while keeping curve shape
    med_src = float(q_src[6])
    med_ref = float(q_ref[6])
    spread_src = max(float(q_src[-1] - q_src[0]), 0.05)
    spread_ref = max(float(q_ref[-1] - q_ref[0]), 0.05)
    scale = np.clip(spread_ref / spread_src, 0.55, 1.65)
    # Softly anchor median shift so extreme dark/bright reference scenes don't blow exposure
    d_med = np.clip(med_ref - med_src, -0.18, 0.18)
    raw_targets = med_src + d_med + (q_ref - med_ref) * np.clip(scale * spread_src / spread_ref, 0.5, 1.0)
    # Blend softly with direct quantile targets (favoring direct quantiles for valid range)
    q_target = 0.78 * q_ref + 0.22 * raw_targets

    # Ensure strictly increasing x_knots and monotone y_knots with bounded local slopes
    x_knots = np.empty(len(_QUANTILES) + 2, dtype=np.float64)
    y_knots = np.empty(len(_QUANTILES) + 2, dtype=np.float64)
    x_knots[0] = 0.0
    x_knots[-1] = 1.0
    x_knots[1:-1] = np.clip(q_src, 1e-4, 1.0 - 1e-4)
    for i in range(1, len(x_knots)):
        if x_knots[i] <= x_knots[i - 1] + 1e-4:
            x_knots[i] = min(1.0, x_knots[i - 1] + 1e-4)
    if x_knots[-2] >= 1.0 - 1e-4:
        x_knots[1:-1] = np.linspace(0.01, 0.99, len(_QUANTILES))

    slope_low = np.clip((q_target[1] - q_target[0]) / max(x_knots[2] - x_knots[1], 1e-3), 0.2, 1.8)
    slope_high = np.clip((q_target[-1] - q_target[-2]) / max(x_knots[-2] - x_knots[-3], 1e-3), 0.2, 1.8)
    y_knots[0] = np.clip(q_target[0] - slope_low * x_knots[1], 0.0, 0.25)
    y_knots[-1] = np.clip(q_target[-1] + slope_high * (1.0 - x_knots[-2]), 0.75, 1.0)
    y_knots[1:-1] = np.clip(q_target, 0.0, 1.0)

    for i in range(1, len(y_knots)):
        dx = x_knots[i] - x_knots[i - 1]
        min_y = y_knots[i - 1] + 0.05 * dx
        max_y = y_knots[i - 1] + 2.8 * dx
        y_knots[i] = np.clip(y_knots[i], min_y, max_y)
    y_knots = np.clip(y_knots, 0.0, 1.0)
    for i in range(1, len(y_knots)):
        if y_knots[i] <= y_knots[i - 1]:
            y_knots[i] = min(1.0, y_knots[i - 1] + 1e-5)

    pchip = PchipInterpolator(x_knots, y_knots, extrapolate=True)
    if shadow_lift == 0.0 and highlight_rolloff == 0.0:
        return pchip

    # Bake shadow_lift & highlight_rolloff cleanly into a monotone PCHIP re-fit
    l_grid = np.linspace(0.0, 1.0, 65, dtype=np.float64)
    l_out = np.clip(pchip(l_grid), 0.0, 1.0)
    if shadow_lift != 0.0:
        l_out = l_out + shadow_lift * 0.14 * np.power(1.0 - l_out, 3.0)
    if highlight_rolloff != 0.0:
        l_out = l_out - highlight_rolloff * 0.14 * np.power(l_out, 3.0)
    l_out = np.maximum.accumulate(np.clip(l_out, 0.0, 1.0))
    return PchipInterpolator(l_grid, l_out, extrapolate=True)


def _huber_mean_cov(
    ab: np.ndarray,
    weights: np.ndarray,
    neutral_bias_sigma: float = 0.14,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute neutral-anchored mean, Huber-robust 2x2 covariance, and neutral coverage."""
    w_sum = float(np.sum(weights))
    if w_sum <= 1e-12:
        return np.zeros(2, dtype=np.float64), np.eye(2, dtype=np.float64) * 1e-3, 0.0

    c_norm = np.hypot(ab[:, 0], ab[:, 1])
    w_neu = weights * np.exp(-0.5 * (c_norm / neutral_bias_sigma) ** 2)
    w_neu_sum = float(np.sum(w_neu))
    neutral_coverage = w_neu_sum / w_sum

    if w_neu_sum > 1e-12:
        mu = (w_neu[:, None] * ab).sum(axis=0) / w_neu_sum
    else:
        mu = (weights[:, None] * ab).sum(axis=0) / w_sum

    # Huber-like IRLS trimming for robust 2x2 covariance around weighted center
    mu_cov = (weights[:, None] * ab).sum(axis=0) / w_sum
    for _ in range(3):
        diff = ab - mu_cov[None, :]
        dist = np.hypot(diff[:, 0], diff[:, 1])
        w_rob = weights / np.sqrt(1.0 + (dist / 0.12) ** 2)
        w_rob_sum = float(np.sum(w_rob))
        if w_rob_sum > 1e-12:
            mu_cov = (w_rob[:, None] * ab).sum(axis=0) / w_rob_sum

    diff = ab - mu_cov[None, :]
    dist = np.hypot(diff[:, 0], diff[:, 1])
    w_rob = weights / np.sqrt(1.0 + (dist / 0.12) ** 2)
    w_rob_sum = float(np.sum(w_rob))
    if w_rob_sum <= 1e-12:
        cov = np.eye(2, dtype=np.float64) * 1e-3
    else:
        cov = (diff * w_rob[:, None]).T @ diff / w_rob_sum

    return mu, cov, float(neutral_coverage)


def _regularized_mkl_2x2(
    cov_src: np.ndarray,
    cov_ref: np.ndarray,
    reg: float = 2e-4,
) -> np.ndarray:
    """2x2 Monge-Kantorovich Linear optimal transport with spectral-mismatch shrinkage."""
    eye2 = np.eye(2, dtype=np.float64)
    s_reg = cov_src + reg * eye2
    r_reg = cov_ref + reg * eye2

    eig_s, vec_s = np.linalg.eigh(s_reg)
    eig_s = np.maximum(eig_s, 1e-8)
    s_half = vec_s @ np.diag(np.sqrt(eig_s)) @ vec_s.T
    s_inv_half = vec_s @ np.diag(1.0 / np.sqrt(eig_s)) @ vec_s.T

    mid = s_half @ r_reg @ s_half
    eig_m, vec_m = np.linalg.eigh(mid)
    eig_m = np.maximum(eig_m, 1e-10)
    mid_half = vec_m @ np.diag(np.sqrt(eig_m)) @ vec_m.T
    t_mkl = s_inv_half @ mid_half @ s_inv_half

    # Effective rank in [1, 2]: near 1 means monochromatic/collinear chromatic distribution
    eig_s_raw = np.maximum(np.linalg.eigvalsh(cov_src), 0.0)
    eig_r_raw = np.maximum(np.linalg.eigvalsh(cov_ref), 0.0)
    eff_s = (np.sum(eig_s_raw) ** 2) / max(float(np.sum(eig_s_raw ** 2)), 1e-12)
    eff_r = (np.sum(eig_r_raw) ** 2) / max(float(np.sum(eig_r_raw ** 2)), 1e-12)

    alpha_rank = np.clip((eff_s - 1.05) / 0.45, 0.0, 1.0) * np.clip((eff_r - 1.05) / 0.45, 0.0, 1.0)
    s_iso = np.clip(
        np.sqrt((np.sum(eig_r_raw) + 2.0 * reg) / (np.sum(eig_s_raw) + 2.0 * reg)),
        0.55,
        1.45,
    )
    t_shrunk = alpha_rank * t_mkl + (1.0 - alpha_rank) * (s_iso * eye2)

    # Bound singular values to avoid exploding low-variance chromatic axes
    u, sv, vt = np.linalg.svd(t_shrunk)
    sv_clamped = np.clip(sv, 0.35, 1.95)
    return u @ np.diag(sv_clamped) @ vt


def _smoothstep_band(x: np.ndarray, low: float, high: float, edge: float) -> np.ndarray:
    t_lo = np.clip((x - (low - edge)) / max(2.0 * edge, 1e-6), 0.0, 1.0)
    t_hi = np.clip(((high + edge) - x) / max(2.0 * edge, 1e-6), 0.0, 1.0)
    s_lo = t_lo * t_lo * (3.0 - 2.0 * t_lo)
    s_hi = t_hi * t_hi * (3.0 - 2.0 * t_hi)
    return s_lo * s_hi


def _apply_skin_guard(
    lab_in: np.ndarray,
    lab_out: np.ndarray,
    skin_protect: float,
) -> np.ndarray:
    if skin_protect <= 0.0:
        return lab_out

    l_in = lab_in[..., 0]
    a_in = lab_in[..., 1]
    b_in = lab_in[..., 2]
    c_in = np.hypot(a_in, b_in)
    h_in = np.degrees(np.arctan2(b_in, a_in)) % 360.0

    m_l = _smoothstep_band(l_in, 0.25, 0.85, 0.10)
    m_c = _smoothstep_band(c_in, 0.015, 0.19, 0.025)
    dh_locus = ((h_in - 45.0 + 180.0) % 360.0) - 180.0
    m_h = np.exp(-0.5 * (dh_locus / 22.0) ** 2)
    sp = 1.0 - (1.0 - skin_protect) ** 1.8
    w_prot = np.clip(sp * m_l * m_c * m_h, 0.0, 1.0)

    a_out = lab_out[..., 1]
    b_out = lab_out[..., 2]
    c_out = np.hypot(a_out, b_out)
    h_out = np.degrees(np.arctan2(b_out, a_out)) % 360.0

    dh = ((h_out - h_in + 180.0) % 360.0) - 180.0
    dh_safe = dh * (1.0 - 0.94 * w_prot)

    c_ratio = np.where(c_in > 1e-6, c_out / np.maximum(c_in, 1e-6), 1.0)
    c_ratio_safe = 1.0 + (c_ratio - 1.0) * (1.0 - 0.55 * w_prot)
    c_safe = np.where(c_in > 1e-6, c_in * c_ratio_safe, c_out)

    h_safe_rad = np.radians(h_in + dh_safe)
    out = lab_out.copy()
    out[..., 1] = c_safe * np.cos(h_safe_rad)
    out[..., 2] = c_safe * np.sin(h_safe_rad)
    return out


def _laplacian_smooth_3d(lut: np.ndarray, smoothness: float) -> np.ndarray:
    if smoothness <= 0.0:
        return lut
    n_iters = max(1, int(round(smoothness * 10.0)))
    alpha = 0.14 * min(smoothness, 1.0)
    anchor = 0.06
    cur = lut.copy()
    orig = lut.copy()
    for _ in range(n_iters):
        avg = (
            cur[2:, 1:-1, 1:-1]
            + cur[:-2, 1:-1, 1:-1]
            + cur[1:-1, 2:, 1:-1]
            + cur[1:-1, :-2, 1:-1]
            + cur[1:-1, 1:-1, 2:]
            + cur[1:-1, 1:-1, :-2]
        ) / 6.0
        cur[1:-1, 1:-1, 1:-1] = (
            (1.0 - alpha) * cur[1:-1, 1:-1, 1:-1]
            + alpha * avg
            + anchor * (orig[1:-1, 1:-1, 1:-1] - cur[1:-1, 1:-1, 1:-1])
        )
    return np.clip(cur, 0.0, 1.0)


def _pava_1d(y: np.ndarray) -> np.ndarray:
    """Exact O(N) Pool Adjacent Violators Algorithm via SciPy compiled isotonic regression."""
    return np.asarray(isotonic_regression(y).x, dtype=np.float64)


def _monotone_separable_smooth_3d(lut: np.ndarray, smoothness: float) -> np.ndarray:
    """Apply separable positive-weight [w, 1-2w, w] smoothing preserving 3-axis monotonicity."""
    if smoothness <= 0.0:
        return lut
    n_passes = max(2, int(round(smoothness * 14.0)))
    w = 0.24 * min(smoothness / 0.35, 1.0)
    w = min(w, 0.28)
    cur = lut.copy()
    for _ in range(n_passes):
        # Axis 0 (R) interior 1..n-2
        nxt = cur.copy()
        nxt[1:-1] = w * cur[:-2] + (1.0 - 2.0 * w) * cur[1:-1] + w * cur[2:]
        cur = nxt
        # Axis 1 (G) interior 1..n-2
        nxt = cur.copy()
        nxt[:, 1:-1] = w * cur[:, :-2] + (1.0 - 2.0 * w) * cur[:, 1:-1] + w * cur[:, 2:]
        cur = nxt
        # Axis 2 (B) interior 1..n-2
        nxt = cur.copy()
        nxt[:, :, 1:-1] = w * cur[:, :, :-2] + (1.0 - 2.0 * w) * cur[:, :, 1:-1] + w * cur[:, :, 2:]
        cur = nxt
    return cur


def _waterfill_luma_shift(cols: np.ndarray, target_luma: np.ndarray, luma_w: np.ndarray) -> np.ndarray:
    """Shift RGB columns (K, N, 3) into [0, 1] so clipped luma matches target_luma."""
    out = cols.copy()
    for _ in range(4):
        cur_luma = out @ luma_w
        err = target_luma - cur_luma
        if np.max(np.abs(err)) < 1e-10:
            break
        # Weight free (unclipped in shift direction) channels by their luma contribution
        free = np.where(err[..., None] >= 0.0, out < 1.0 - 1e-12, out > 1e-12)
        denom = np.sum(free * luma_w[None, None, :], axis=-1)
        step = np.where(denom > 1e-9, err / np.maximum(denom, 1e-9), 0.0)
        out = np.clip(out + step[..., None] * free, 0.0, 1.0)
    return out


def _project_monotone_lattice(lut: np.ndarray, max_iters: int = 32) -> np.ndarray:
    """Project 3D LUT onto joint per-axis channel and luma monotonicity cone via cyclic PAVA."""
    arr = np.clip(lut.copy(), 0.0, 1.0)
    luma_w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    tol = -1e-8

    for _ in range(max_iters):
        changed = False

        # Self-channel monotonicity along R (axis 0), G (axis 1), B (axis 2)
        viol_gb = np.argwhere(np.any(np.diff(arr[..., 0], axis=0) < tol, axis=0))
        if len(viol_gb) > 0:
            changed = True
            for g, b in viol_gb:
                arr[:, g, b, 0] = _pava_1d(arr[:, g, b, 0])

        viol_rb = np.argwhere(np.any(np.diff(arr[..., 1], axis=1) < tol, axis=1))
        if len(viol_rb) > 0:
            changed = True
            for r, b in viol_rb:
                arr[r, :, b, 1] = _pava_1d(arr[r, :, b, 1])

        viol_rg = np.argwhere(np.any(np.diff(arr[..., 2], axis=2) < tol, axis=2))
        if len(viol_rg) > 0:
            changed = True
            for r, g in viol_rg:
                arr[r, g, :, 2] = _pava_1d(arr[r, g, :, 2])

        # Luminance monotonicity along all 3 axes via water-filling
        luma = arr @ luma_w
        viol_l0 = np.argwhere(np.any(np.diff(luma, axis=0) < tol, axis=0))
        if len(viol_l0) > 0:
            changed = True
            gs, bs = viol_l0[:, 0], viol_l0[:, 1]
            tgt = np.array([_pava_1d(luma[:, g, b]) for g, b in viol_l0])
            cols = np.transpose(arr[:, gs, bs, :], (1, 0, 2))
            arr[:, gs, bs, :] = np.transpose(_waterfill_luma_shift(cols, tgt, luma_w), (1, 0, 2))

        luma = arr @ luma_w
        viol_l1 = np.argwhere(np.any(np.diff(luma, axis=1) < tol, axis=1))
        if len(viol_l1) > 0:
            changed = True
            rs, bs = viol_l1[:, 0], viol_l1[:, 1]
            tgt = np.array([_pava_1d(luma[r, :, b]) for r, b in viol_l1])
            cols = np.transpose(arr[rs, :, bs, :], (0, 1, 2))
            arr[rs, :, bs, :] = _waterfill_luma_shift(cols, tgt, luma_w)

        luma = arr @ luma_w
        viol_l2 = np.argwhere(np.any(np.diff(luma, axis=2) < tol, axis=2))
        if len(viol_l2) > 0:
            changed = True
            rs, gs = viol_l2[:, 0], viol_l2[:, 1]
            tgt = np.array([_pava_1d(luma[r, g, :]) for r, g in viol_l2])
            cols = arr[rs, gs, :, :]
            arr[rs, gs, :, :] = _waterfill_luma_shift(cols, tgt, luma_w)

        if not changed:
            break
    return np.clip(arr, 0.0, 1.0)


def _count_inversions(lut: np.ndarray) -> int:
    eps = 1e-6
    luma = 0.2126 * lut[..., 0] + 0.7152 * lut[..., 1] + 0.0722 * lut[..., 2]
    inv_r = np.sum(np.diff(lut[..., 0], axis=0) < -eps) + np.sum(np.diff(luma, axis=0) < -eps)
    inv_g = np.sum(np.diff(lut[..., 1], axis=1) < -eps) + np.sum(np.diff(luma, axis=1) < -eps)
    inv_b = np.sum(np.diff(lut[..., 2], axis=2) < -eps) + np.sum(np.diff(luma, axis=2) < -eps)
    return int(inv_r + inv_g + inv_b)


def _pearson_corr(x: np.ndarray, y: np.ndarray, min_var: float = 1e-12) -> float:
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    var_x = float(np.mean(xc * xc))
    var_y = float(np.mean(yc * yc))
    if var_x <= min_var or var_y <= min_var:
        return 0.0
    return float(np.mean(xc * yc) / np.sqrt(var_x * var_y))


def _is_paired_shot(src_working: np.ndarray, ref_rgb: np.ndarray) -> bool:
    """Detect same-shot paired before/after images via aspect ratio and 32x32 luma + gradient correlation."""
    sh, sw, _ = src_working.shape
    rh, rw, _ = ref_rgb.shape
    if sh <= 0 or sw <= 0 or rh <= 0 or rw <= 0:
        return False
    aspect_src = sw / sh
    aspect_ref = rw / rh
    if abs(aspect_src - aspect_ref) / max(aspect_src, aspect_ref) > 0.02:
        return False

    ys_s = (np.arange(32, dtype=np.intp) * max(sh - 1, 0)) // 31
    xs_s = (np.arange(32, dtype=np.intp) * max(sw - 1, 0)) // 31
    ys_r = (np.arange(32, dtype=np.intp) * max(rh - 1, 0)) // 31
    xs_r = (np.arange(32, dtype=np.intp) * max(rw - 1, 0)) // 31
    s_grid = src_working[np.ix_(ys_s, xs_s)]
    r_grid = np.clip(ref_rgb[np.ix_(ys_r, xs_r)], 0.0, 1.0)

    ls_2d = 0.2126 * s_grid[..., 0] + 0.7152 * s_grid[..., 1] + 0.0722 * s_grid[..., 2]
    lr_2d = 0.2126 * r_grid[..., 0] + 0.7152 * r_grid[..., 1] + 0.0722 * r_grid[..., 2]

    dx_s = (ls_2d[:, 1:] - ls_2d[:, :-1]).ravel()
    dx_r = (lr_2d[:, 1:] - lr_2d[:, :-1]).ravel()
    dy_s = (ls_2d[1:, :] - ls_2d[:-1, :]).ravel()
    dy_r = (lr_2d[1:, :] - lr_2d[:-1, :]).ravel()

    if (
        float(np.std(dx_s)) <= 0.015
        or float(np.std(dx_r)) <= 0.015
        or float(np.std(dy_s)) <= 0.015
        or float(np.std(dy_r)) <= 0.015
    ):
        return False

    if _pearson_corr(ls_2d.ravel(), lr_2d.ravel(), min_var=1e-6) <= 0.98:
        return False
    if _pearson_corr(dx_s, dx_r) <= 0.92:
        return False
    if _pearson_corr(dy_s, dy_r) <= 0.92:
        return False
    return True


def _paired_features(lab: np.ndarray) -> np.ndarray:
    """Build 14-term smooth OKLab polynomial + k=1,2,3 chroma-weighted hue-harmonic basis."""
    l_v = lab[:, 0]
    a_v = lab[:, 1]
    b_v = lab[:, 2]
    c = np.sqrt(a_v * a_v + b_v * b_v)
    valid = c > 1e-8
    c_safe = np.where(valid, c, 1.0)
    ct = np.where(valid, a_v / c_safe, 0.0)
    st = np.where(valid, b_v / c_safe, 0.0)
    c2 = np.where(valid, 2.0 * ct * ct - 1.0, 0.0)
    s2 = np.where(valid, 2.0 * st * ct, 0.0)
    c3 = np.where(valid, 4.0 * ct * ct * ct - 3.0 * ct, 0.0)
    s3 = np.where(valid, 3.0 * st - 4.0 * st * st * st, 0.0)
    wc = np.where(valid, (c * c) / (c + 0.04), 0.0)
    return np.column_stack(
        [
            np.ones_like(l_v),
            l_v,
            a_v,
            b_v,
            l_v * l_v,
            l_v * a_v,
            l_v * b_v,
            l_v * l_v * l_v,
            wc * ct,
            wc * st,
            wc * c2,
            wc * s2,
            wc * c3,
            wc * s3,
        ]
    )


def _hue_sector_stats(ab: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute smooth chroma-weighted circular density, angular offset, and mean chroma over 24 hue anchors."""
    c_sq = ab[:, 0] * ab[:, 0] + ab[:, 1] * ab[:, 1]
    c = np.sqrt(c_sq)
    gc = c_sq / (c_sq + 0.035 * 0.035)
    wc = weights * gc
    keep = (c_sq > 1e-12) & (wc > 1e-12)
    if not np.any(keep):
        return (
            np.zeros(24, dtype=np.float64),
            np.zeros(24, dtype=np.float64),
            np.full(24, 0.05, dtype=np.float64),
        )
    ab_k = ab[keep]
    c_k = c[keep]
    wc_k = wc[keep]
    tot = max(float(np.sum(wc_k)), 1e-12)
    theta = np.arctan2(ab_k[:, 1], ab_k[:, 0])
    scale_bin = _HUE_BINS / (2.0 * np.pi)
    bins = np.clip(np.floor((theta + np.pi) * scale_bin).astype(np.intp), 0, _HUE_BINS - 1)

    hist_w = np.bincount(bins, weights=wc_k, minlength=_HUE_BINS).astype(np.float64)
    hist_c = np.bincount(bins, weights=wc_k * c_k, minlength=_HUE_BINS).astype(np.float64)
    hist_dt = np.bincount(
        bins, weights=wc_k * (theta - _HUE_BIN_CENTERS[bins]), minlength=_HUE_BINS
    ).astype(np.float64)

    mass = _HUE_KERNEL_MB @ hist_w
    ang_num = (_HUE_KERNEL_MB * _HUE_DIFF_MB) @ hist_w + _HUE_KERNEL_MB @ hist_dt
    chr_num = _HUE_KERNEL_MB @ hist_c

    p = mass / tot
    delta = np.where(mass > 1e-12, ang_num / np.maximum(mass, 1e-12), 0.0)
    mean_c = np.where(mass > 1e-12, chr_num / np.maximum(mass, 1e-12), 0.05)
    return p, delta, mean_c


def _fit_hue_harmonic_residual(
    ab_src_mkl: np.ndarray,
    w_src: np.ndarray,
    ab_ref: np.ndarray,
    w_ref: np.ndarray,
    hue_alpha: float,
    sat_alpha: float,
    max_hue_rad: float,
    max_log_sat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit smooth k=1,2,3 Fourier hue-vs-hue and hue-vs-sat residuals guarded by joint hue overlap."""
    p_s, delta_s, c_s = _hue_sector_stats(ab_src_mkl, w_src)
    p_r, delta_r, c_r = _hue_sector_stats(ab_ref, w_ref)

    wm = np.minimum(p_s, p_r)
    wm_sum = float(np.sum(wm))
    denom_ov = max(float(np.sum(p_s)), float(np.sum(p_r)), 1e-12)
    overlap_ratio = wm_sum / denom_ov
    hue_conf = float(np.clip((overlap_ratio - 0.18) / 0.52, 0.0, 1.0))

    if wm_sum <= 1e-12 or hue_conf <= 1e-9:
        return np.zeros(6, dtype=np.float64), np.zeros(6, dtype=np.float64)

    y_hue = np.clip(delta_r - delta_s, -max_hue_rad, max_hue_rad)
    y_sat = np.clip(np.log((c_r + 0.025) / (c_s + 0.025)), -max_log_sat, max_log_sat)
    w_norm = wm / wm_sum

    bt_w = _HUE_BASIS.T * w_norm[None, :]
    lhs = bt_w @ _HUE_BASIS + _HUE_REG_DIAG
    rhs = np.column_stack([bt_w @ y_hue, bt_w @ y_sat])
    sol = np.linalg.solve(lhs, rhs)
    coef_hue = sol[:, 0] * (hue_alpha * hue_conf)
    coef_sat = sol[:, 1] * (sat_alpha * hue_conf)
    return coef_hue, coef_sat


def _apply_hue_harmonic_residual(
    ab: np.ndarray,
    coef_hue: np.ndarray,
    coef_sat: np.ndarray,
    max_hue_rad: float,
    max_log_sat: float,
) -> np.ndarray:
    """Apply smooth chroma-gated k=1,2,3 hue-vs-hue rotation and hue-vs-sat scaling."""
    if not (np.any(coef_hue != 0.0) or np.any(coef_sat != 0.0)):
        return ab
    a_v = ab[:, 0]
    b_v = ab[:, 1]
    c = np.sqrt(a_v * a_v + b_v * b_v)
    valid = c > 1e-7
    c_safe = np.where(valid, c, 1.0)
    ct = np.where(valid, a_v / c_safe, 0.0)
    st = np.where(valid, b_v / c_safe, 0.0)
    c2 = 2.0 * ct * ct - 1.0
    s2 = 2.0 * st * ct
    c3 = 4.0 * ct * ct * ct - 3.0 * ct
    s3 = 3.0 * st - 4.0 * st * st * st

    gate = np.where(valid, (c * c) / (c * c + 0.03 * 0.03), 0.0)
    raw_dh = (
        coef_hue[0] * ct
        + coef_hue[1] * st
        + coef_hue[2] * c2
        + coef_hue[3] * s2
        + coef_hue[4] * c3
        + coef_hue[5] * s3
    )
    raw_ds = (
        coef_sat[0] * ct
        + coef_sat[1] * st
        + coef_sat[2] * c2
        + coef_sat[3] * s2
        + coef_sat[4] * c3
        + coef_sat[5] * s3
    )
    dtheta = np.clip(raw_dh, -max_hue_rad, max_hue_rad) * gate
    dlogc = np.clip(raw_ds, -max_log_sat, max_log_sat) * gate
    scale_c = np.exp(dlogc)
    cd = np.cos(dtheta)
    sd = np.sin(dtheta)

    a_out = np.where(valid, scale_c * (cd * a_v - sd * b_v), a_v)
    b_out = np.where(valid, scale_c * (sd * a_v + cd * b_v), b_v)
    return np.column_stack([a_out, b_out])


def _solve_paired_working_lut(
    src_working: np.ndarray,
    ref_rgb: np.ndarray,
    size: int,
    strength: float,
    skin_protect: float,
    shadow_lift: float,
    highlight_rolloff: float,
    smoothness: float,
) -> np.ndarray:
    """Fit a smooth, monotone working-space 3D LUT via OKLab polynomial + hue-harmonic ridge."""
    sh, sw, _ = src_working.shape
    rh, rw, _ = ref_rgb.shape
    ys_s = (np.arange(64, dtype=np.intp) * max(sh - 1, 0)) // 63
    xs_s = (np.arange(64, dtype=np.intp) * max(sw - 1, 0)) // 63
    ys_r = (np.arange(64, dtype=np.intp) * max(rh - 1, 0)) // 63
    xs_r = (np.arange(64, dtype=np.intp) * max(rw - 1, 0)) // 63

    s_samp = src_working[np.ix_(ys_s, xs_s)].reshape((-1, 3))
    r_samp = np.clip(ref_rgb[np.ix_(ys_r, xs_r)], 0.0, 1.0).reshape((-1, 3))
    s_lab = srgb_to_oklab(s_samp)
    r_lab = srgb_to_oklab(r_samp)
    delta = r_lab - s_lab

    x_mat = _paired_features(s_lab)
    w_mat = np.linalg.solve(x_mat.T @ x_mat + _PAIRED_REG_DIAG, x_mat.T @ delta)

    ident = identity_lut(size)
    lab_in = srgb_to_oklab(ident.reshape((-1, 3)))
    x_grid = _paired_features(lab_in)
    lab_out = lab_in + x_grid @ w_mat
    l_out = np.clip(lab_out[:, 0], 0.0, 1.0)
    if shadow_lift != 0.0:
        l_out = np.clip(l_out + shadow_lift * 0.14 * np.power(1.0 - l_out, 3.0), 0.0, 1.0)
    if highlight_rolloff != 0.0:
        l_out = np.clip(l_out - highlight_rolloff * 0.14 * np.power(l_out, 3.0), 0.0, 1.0)
    lab_out[:, 0] = l_out

    if skin_protect > 0.0:
        lab_out = _apply_skin_guard(lab_in, lab_out, skin_protect=skin_protect)

    srgb_out = np.clip(oklab_to_srgb(lab_out), 0.0, 1.0).reshape((size, size, size, 3))
    sm = smoothness * 0.15
    srgb_out = _laplacian_smooth_3d(srgb_out, smoothness=sm)
    srgb_out = _project_monotone_lattice(srgb_out, max_iters=64)
    srgb_out = _monotone_separable_smooth_3d(srgb_out, smoothness=sm)

    if strength < 1.0:
        srgb_out = (1.0 - strength) * ident + strength * srgb_out
    return np.clip(srgb_out, 0.0, 1.0)


def solve_lut(
    src_rgb: np.ndarray | Sequence[np.ndarray],
    ref_rgbs: np.ndarray | Sequence[np.ndarray],
    size: int = 33,
    strength: float = 1.0,
    skin_protect: float = 0.5,
    shadow_lift: float = 0.0,
    highlight_rolloff: float = 0.0,
    smoothness: float = 0.35,
    idt: str = "srgb",
    mode: str = "balanced",
    max_samples: int = 32768,
) -> np.ndarray:
    """Fit a smooth, monotone 3D LUT mapping src_rgb look toward ref_rgbs."""
    if size not in (17, 33, 65):
        raise ValueError(f"LUT size must be 17, 33, or 65; got {size}")
    if not (0.0 <= strength <= 1.0):
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    if not (0.0 <= skin_protect <= 1.0):
        raise ValueError(f"skin_protect must be in [0, 1], got {skin_protect}")
    mode_key = mode.lower().strip()
    if mode_key not in _MODE_CONFIG:
        raise ValueError(f"mode must be 'balanced' or 'creative'; got {mode!r}")
    cfg = _MODE_CONFIG[mode_key]

    if isinstance(src_rgb, np.ndarray) and src_rgb.ndim == 3:
        src_list = [src_rgb]
    else:
        src_list = list(src_rgb)
    if not src_list:
        raise ValueError("At least one source image is required")

    if isinstance(ref_rgbs, np.ndarray) and ref_rgbs.ndim == 3:
        ref_list = [ref_rgbs]
    else:
        ref_list = list(ref_rgbs)
    if not ref_list:
        raise ValueError("At least one reference image is required")

    idt_key = idt.lower().strip()
    src_working_list = [apply_idt(s, idt=idt_key) for s in src_list]
    ident = identity_lut(size)

    lut_working: np.ndarray | None = None
    if (
        len(src_working_list) == 1
        and len(ref_list) == 1
        and _is_paired_shot(src_working_list[0], np.asarray(ref_list[0], dtype=np.float64))
    ):
        candidate = _solve_paired_working_lut(
            src_working=src_working_list[0],
            ref_rgb=np.asarray(ref_list[0], dtype=np.float64),
            size=size,
            strength=strength,
            skin_protect=skin_protect,
            shadow_lift=shadow_lift,
            highlight_rolloff=highlight_rolloff,
            smoothness=smoothness,
        )
        if _count_inversions(candidate) == 0:
            lut_working = candidate

    if lut_working is None:
        per_src_budget = max(2048, max_samples // len(src_working_list))
        src_labs = []
        src_weights = []
        for src_w in src_working_list:
            s_samp = _subsample_rgb(src_w, per_src_budget)
            s_lab = srgb_to_oklab(s_samp)
            s_w = _density_weights(s_lab)
            s_w = s_w / max(float(np.sum(s_w)), 1e-12)
            src_labs.append(s_lab)
            src_weights.append(s_w)

        src_lab = np.concatenate(src_labs, axis=0)
        w_src_dens = np.concatenate(src_weights, axis=0)

        per_ref_budget = max(2048, max_samples // len(ref_list))
        ref_labs = []
        ref_weights = []
        for ref in ref_list:
            r_samp = _subsample_rgb(np.clip(np.asarray(ref, dtype=np.float64), 0.0, 1.0), per_ref_budget)
            r_lab = srgb_to_oklab(r_samp)
            r_w = _density_weights(r_lab)
            # Equal total weight per reference still regardless of resolution
            r_w = r_w / max(float(np.sum(r_w)), 1e-12)
            ref_labs.append(r_lab)
            ref_weights.append(r_w)

        ref_lab = np.concatenate(ref_labs, axis=0)
        w_ref_dens = np.concatenate(ref_weights, axis=0)

        l_curve = _fit_luminance_curve(
            src_lab[:, 0],
            w_src_dens,
            ref_lab[:, 0],
            w_ref_dens,
            shadow_lift=0.0,
            highlight_rolloff=0.0,
        )

        # Compute per-band neutral-anchored means and regularized 2x2 MKL transport
        eye2 = np.eye(2, dtype=np.float64)
        mkl_alpha = cfg["mkl_alpha"]
        shift_alpha = cfg["shift_alpha"]
        shift_max_scale = cfg["shift_max_scale"]
        shift_max_base = cfg["shift_max_base"]
        band_transforms: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for c_k in _BAND_CENTERS:
            ws = w_src_dens * np.exp(-0.5 * ((src_lab[:, 0] - c_k) / _BAND_SIGMA) ** 2)
            wr = w_ref_dens * np.exp(-0.5 * ((ref_lab[:, 0] - c_k) / _BAND_SIGMA) ** 2)

            mu_s, cov_s, cov_neu_s = _huber_mean_cov(src_lab[:, 1:3], ws)
            mu_r, cov_r, cov_neu_r = _huber_mean_cov(ref_lab[:, 1:3], wr)
            t_k = (1.0 - mkl_alpha) * eye2 + mkl_alpha * _regularized_mkl_2x2(cov_s, cov_r)

            # Shrink extreme neutral-axis shifts when reference band lacks neutral support
            neu_confidence = np.clip(min(cov_neu_s, cov_neu_r) / 0.18, 0.15, 1.0)
            shift = shift_alpha * (mu_r - mu_s)
            shift_norm = float(np.hypot(shift[0], shift[1]))
            max_shift = shift_max_scale * neu_confidence + shift_max_base
            if shift_norm > max_shift:
                shift = shift * (max_shift / shift_norm)
            mu_r_safe = mu_s + shift
            band_transforms.append((mu_s, mu_r_safe, t_k))

        # Evaluate 5-band MKL on sampled source pixels, then fit smooth k=1,2,3 hue-harmonic residual
        raw_gs = np.exp(-0.5 * ((src_lab[:, 0:1] - _BAND_CENTERS[None, :]) / _BAND_SIGMA) ** 2)
        g_src = raw_gs / np.maximum(np.sum(raw_gs, axis=1, keepdims=True), 1e-12)
        ab_src_mkl = np.zeros_like(src_lab[:, 1:3])
        for k, (mu_s, mu_r, t_k) in enumerate(band_transforms):
            ab_src_mkl += g_src[:, k : k + 1] * (mu_r[None, :] + (src_lab[:, 1:3] - mu_s[None, :]) @ t_k.T)

        coef_hue, coef_sat = _fit_hue_harmonic_residual(
            ab_src_mkl=ab_src_mkl,
            w_src=w_src_dens,
            ab_ref=ref_lab[:, 1:3],
            w_ref=w_ref_dens,
            hue_alpha=cfg["hue_alpha"],
            sat_alpha=cfg["sat_alpha"],
            max_hue_rad=cfg["max_hue_rad"],
            max_log_sat=cfg["max_log_sat"],
        )

        # Evaluate continuous mapping over the uniform working sRGB lattice
        grid_flat = ident.reshape((-1, 3))
        lab_in = srgb_to_oklab(grid_flat)

        luma_alpha = cfg["luma_alpha"]
        l_out = (1.0 - luma_alpha) * lab_in[:, 0] + luma_alpha * np.clip(l_curve(lab_in[:, 0]), 0.0, 1.0)
        if shadow_lift != 0.0:
            l_out = np.clip(l_out + shadow_lift * 0.14 * np.power(1.0 - l_out, 3.0), 0.0, 1.0)
        if highlight_rolloff != 0.0:
            l_out = np.clip(l_out - highlight_rolloff * 0.14 * np.power(l_out, 3.0), 0.0, 1.0)

        # Normalized Gaussian band weights evaluated at input L
        raw_g = np.exp(-0.5 * ((lab_in[:, 0:1] - _BAND_CENTERS[None, :]) / _BAND_SIGMA) ** 2)
        g_weights = raw_g / np.maximum(np.sum(raw_g, axis=1, keepdims=True), 1e-12)

        ab_in = lab_in[:, 1:3]
        ab_out = np.zeros_like(ab_in)
        for k, (mu_s, mu_r, t_k) in enumerate(band_transforms):
            mapped_k = mu_r[None, :] + (ab_in - mu_s[None, :]) @ t_k.T
            ab_out += g_weights[:, k : k + 1] * mapped_k

        ab_out = _apply_hue_harmonic_residual(
            ab_out,
            coef_hue=coef_hue,
            coef_sat=coef_sat,
            max_hue_rad=cfg["max_hue_rad"],
            max_log_sat=cfg["max_log_sat"],
        )

        lab_out = np.column_stack([l_out, ab_out])
        lab_out = _apply_skin_guard(lab_in, lab_out, skin_protect=skin_protect)

        srgb_out = np.clip(oklab_to_srgb(lab_out), 0.0, 1.0).reshape((size, size, size, 3))
        srgb_out = _laplacian_smooth_3d(srgb_out, smoothness=smoothness)
        srgb_out = _project_monotone_lattice(srgb_out)
        srgb_out = _monotone_separable_smooth_3d(srgb_out, smoothness=smoothness)

        if strength < 1.0:
            srgb_out = (1.0 - strength) * ident + strength * srgb_out
        lut_working = np.clip(srgb_out, 0.0, 1.0)

    if idt_key == "srgb":
        return lut_working

    # Compose IDT into the 3D LUT input so Log/Rec709 cubes map Source -> Graded Display directly
    grid_working = apply_idt(ident, idt=idt_key)
    return np.clip(apply_lut(grid_working, lut_working), 0.0, 1.0)
