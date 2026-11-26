from __future__ import annotations

from typing import Callable

import numpy as np

from lut_match.color import (
    linear_to_srgb,
    oklab_to_srgb,
    rgb_to_lab,
    srgb_to_linear,
    srgb_to_oklab,
)
from lut_match.io import identity_lut

_M_XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)
_D65_XYZ = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)


def _lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (l + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    eps = 6.0 / 29.0
    kappa_inv = 3.0 * eps * eps
    xr = np.where(fx > eps, fx ** 3, kappa_inv * (fx - 4.0 / 29.0))
    yr = np.where(fy > eps, fy ** 3, kappa_inv * (fy - 4.0 / 29.0))
    zr = np.where(fz > eps, fz ** 3, kappa_inv * (fz - 4.0 / 29.0))

    xyz = np.stack([xr, yr, zr], axis=-1) * _D65_XYZ
    lin = xyz @ _M_XYZ_TO_SRGB.T
    return np.clip(linear_to_srgb(np.clip(lin, 0.0, 1.0)), 0.0, 1.0)


def _fit_mkl_transform(x_src: np.ndarray, x_ref: np.ndarray, reg: float = 1e-5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu_s = np.mean(x_src, axis=0)
    mu_r = np.mean(x_ref, axis=0)
    cov_s = np.cov(x_src, rowvar=False) + reg * np.eye(3)
    cov_r = np.cov(x_ref, rowvar=False) + reg * np.eye(3)

    eig_s, vec_s = np.linalg.eigh(cov_s)
    eig_s = np.maximum(eig_s, 1e-8)
    s_half = vec_s @ np.diag(np.sqrt(eig_s)) @ vec_s.T
    s_inv_half = vec_s @ np.diag(1.0 / np.sqrt(eig_s)) @ vec_s.T

    mid = s_half @ cov_r @ s_half
    eig_m, vec_m = np.linalg.eigh(mid)
    eig_m = np.maximum(eig_m, 1e-10)
    mid_half = vec_m @ np.diag(np.sqrt(eig_m)) @ vec_m.T
    t = s_inv_half @ mid_half @ s_inv_half
    return mu_s, mu_r, t


def fit_reinhard_lab(src_rgb: np.ndarray, ref_rgb: np.ndarray, size: int = 33) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    src_lab = rgb_to_lab(src_rgb).reshape((-1, 3))
    ref_lab = rgb_to_lab(ref_rgb).reshape((-1, 3))
    mu_s = np.mean(src_lab, axis=0)
    sd_s = np.maximum(np.std(src_lab, axis=0), 1e-4)
    mu_r = np.mean(ref_lab, axis=0)
    sd_r = np.std(ref_lab, axis=0)

    def apply_fn(img: np.ndarray) -> np.ndarray:
        lab = rgb_to_lab(img)
        out_lab = (lab - mu_s) * (sd_r / sd_s) + mu_r
        return _lab_to_srgb(out_lab)

    lut = apply_fn(identity_lut(size))
    return lut, apply_fn


def fit_mkl_rgb(src_rgb: np.ndarray, ref_rgb: np.ndarray, size: int = 33) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    s_flat = src_rgb.reshape((-1, 3))
    r_flat = ref_rgb.reshape((-1, 3))
    mu_s, mu_r, t = _fit_mkl_transform(s_flat, r_flat)

    def apply_fn(img: np.ndarray) -> np.ndarray:
        out = mu_r + (img.reshape((-1, 3)) - mu_s) @ t.T
        return np.clip(out.reshape(img.shape), 0.0, 1.0)

    lut = apply_fn(identity_lut(size))
    return lut, apply_fn


def fit_mkl_oklab(src_rgb: np.ndarray, ref_rgb: np.ndarray, size: int = 33) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    s_flat = srgb_to_oklab(src_rgb).reshape((-1, 3))
    r_flat = srgb_to_oklab(ref_rgb).reshape((-1, 3))
    mu_s, mu_r, t = _fit_mkl_transform(s_flat, r_flat)

    def apply_fn(img: np.ndarray) -> np.ndarray:
        ok = srgb_to_oklab(img).reshape((-1, 3))
        out = mu_r + (ok - mu_s) @ t.T
        return np.clip(oklab_to_srgb(out.reshape(img.shape)), 0.0, 1.0)

    lut = apply_fn(identity_lut(size))
    return lut, apply_fn


def _histogram_match_1d(vals: np.ndarray, src_ref: np.ndarray, tgt_ref: np.ndarray) -> np.ndarray:
    """Continuous per-channel empirical CDF matching suitable for both images and LUT grids."""
    q_grid = np.linspace(0.0, 1.0, 513)
    src_q = np.quantile(src_ref, q_grid)
    tgt_q = np.quantile(tgt_ref, q_grid)
    # Enforce strictly non-decreasing src_q with tiny jitter for np.interp stability
    src_q = np.maximum.accumulate(src_q)
    return np.interp(vals, src_q, tgt_q)


def fit_hist_match(src_rgb: np.ndarray, ref_rgb: np.ndarray, size: int = 33) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    s_flat = src_rgb.reshape((-1, 3))
    r_flat = ref_rgb.reshape((-1, 3))

    def apply_fn(img: np.ndarray) -> np.ndarray:
        flat = img.reshape((-1, 3))
        out = np.empty_like(flat)
        for ch in range(3):
            out[:, ch] = _histogram_match_1d(flat[:, ch], s_flat[:, ch], r_flat[:, ch])
        return np.clip(out.reshape(img.shape), 0.0, 1.0)

    lut = apply_fn(identity_lut(size))
    return lut, apply_fn


def fit_hm_mvgd_hm(src_rgb: np.ndarray, ref_rgb: np.ndarray, size: int = 33) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """color-matcher style HM-MVGD-HM: marginal HM -> MVGD (MKL) -> marginal HM."""
    s_flat = src_rgb.reshape((-1, 3))
    r_flat = ref_rgb.reshape((-1, 3))

    # Stage 1: initial marginal histogram match of src -> ref
    hm1_s = np.empty_like(s_flat)
    for ch in range(3):
        hm1_s[:, ch] = _histogram_match_1d(s_flat[:, ch], s_flat[:, ch], r_flat[:, ch])

    # Stage 2: MVGD linear Gaussian transport from hm1_s -> r_flat
    mu_s, mu_r, t = _fit_mkl_transform(hm1_s, r_flat)
    mvgd_s = np.clip(mu_r + (hm1_s - mu_s) @ t.T, 0.0, 1.0)

    def apply_fn(img: np.ndarray) -> np.ndarray:
        flat = img.reshape((-1, 3))
        stage1 = np.empty_like(flat)
        for ch in range(3):
            stage1[:, ch] = _histogram_match_1d(flat[:, ch], s_flat[:, ch], r_flat[:, ch])
        stage2 = np.clip(mu_r + (stage1 - mu_s) @ t.T, 0.0, 1.0)
        stage3 = np.empty_like(stage2)
        for ch in range(3):
            stage3[:, ch] = _histogram_match_1d(stage2[:, ch], mvgd_s[:, ch], r_flat[:, ch])
        return np.clip(stage3.reshape(img.shape), 0.0, 1.0)

    lut = apply_fn(identity_lut(size))
    return lut, apply_fn
