from __future__ import annotations

import numpy as np

SUPPORTED_IDTS = ("srgb", "rec709", "apple-log", "slog3", "vlog", "clog3")

_M_SRGB_TO_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float64,
)
_M_LMS_TO_SRGB = np.linalg.inv(_M_SRGB_TO_LMS)

_M_LMS_TO_OKLAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float64,
)
_M_OKLAB_TO_LMS = np.linalg.inv(_M_LMS_TO_OKLAB)

_M_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_D65_XYZ = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)


def _npm_from_xy(
    rx: float,
    ry: float,
    gx: float,
    gy: float,
    bx: float,
    by: float,
    wx: float = 0.3127,
    wy: float = 0.3290,
) -> np.ndarray:
    """Compute Normalized Primary Matrix mapping linear RGB to CIE XYZ."""
    primaries = np.array(
        [
            [rx / ry, gx / gy, bx / by],
            [1.0, 1.0, 1.0],
            [(1.0 - rx - ry) / ry, (1.0 - gx - gy) / gy, (1.0 - bx - by) / by],
        ],
        dtype=np.float64,
    )
    white_xyz = np.array([wx / wy, 1.0, (1.0 - wx - wy) / wy], dtype=np.float64)
    scales = np.linalg.solve(primaries, white_xyz)
    return primaries * scales[None, :]


_NPM_REC709 = _npm_from_xy(0.64, 0.33, 0.30, 0.60, 0.15, 0.06)
_INV_NPM_REC709 = np.linalg.inv(_NPM_REC709)

_M_REC2020_TO_REC709 = _INV_NPM_REC709 @ _npm_from_xy(0.708, 0.292, 0.170, 0.797, 0.131, 0.046)
_M_SGAMUT3CINE_TO_REC709 = _INV_NPM_REC709 @ _npm_from_xy(0.766, 0.275, 0.225, 0.800, 0.089, -0.087)
_M_VGAMUT_TO_REC709 = np.array(
    [
        [1.806576, -0.695697, -0.110879],
        [-0.170090, 1.305955, -0.135865],
        [-0.025206, -0.154468, 1.179674],
    ],
    dtype=np.float64,
)
_M_CGAMUT_TO_REC709 = _INV_NPM_REC709 @ _npm_from_xy(0.740, 0.270, 0.170, 1.140, 0.080, -0.100)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float64)
    return np.where(
        x <= 0.04045,
        x / 12.92,
        np.power((np.maximum(x, 0.04045) + 0.055) / 1.055, 2.4),
    )


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float64)
    return np.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * np.power(np.maximum(x, 0.0031308), 1.0 / 2.4) - 0.055,
    )


def linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float64)
    lms = x @ _M_SRGB_TO_LMS.T
    lms_cbrt = np.cbrt(lms)
    return lms_cbrt @ _M_LMS_TO_OKLAB.T


def oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    x = np.asarray(lab, dtype=np.float64)
    lms_cbrt = x @ _M_OKLAB_TO_LMS.T
    lms = lms_cbrt * lms_cbrt * lms_cbrt
    return lms @ _M_LMS_TO_SRGB.T


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    return linear_to_oklab(srgb_to_linear(rgb))


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    return linear_to_srgb(oklab_to_linear(lab))


def _rec709_to_linear(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float64)
    alpha = 1.09929682680944
    cut = 0.0812428582986315
    return np.where(
        x < cut,
        x / 4.5,
        np.power((np.maximum(x, cut) + (alpha - 1.0)) / alpha, 1.0 / 0.45),
    )


def _apple_log_to_linear(rgb: np.ndarray) -> np.ndarray:
    p = np.asarray(rgb, dtype=np.float64)
    r_0 = -0.05641088
    r_t = 0.01
    c = 47.28711236
    beta = 0.00964052
    gamma = 0.08550479
    delta = 0.69336945
    p_t = c * (r_t - r_0) ** 2
    lin = np.where(
        p >= p_t,
        np.power(2.0, (p - delta) / gamma) - beta,
        np.where(p > 0.0, np.sqrt(np.maximum(p, 0.0) / c) + r_0, r_0),
    )
    return lin @ _M_REC2020_TO_REC709.T


def _slog3_to_linear(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float64)
    t = x * 1023.0
    cut = 171.2102946929
    lin = np.where(
        t >= cut,
        np.power(10.0, (t - 420.0) / 261.5) * 0.19 - 0.01,
        (t - 95.0) * 0.01125 / (cut - 95.0),
    )
    return lin @ _M_SGAMUT3CINE_TO_REC709.T


def _vlog_to_linear(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float64)
    lin = np.where(
        x >= 0.181,
        np.power(10.0, (x - 0.598206) / 0.241514) - 0.00873,
        (x - 0.125) / 5.6,
    )
    return lin @ _M_VGAMUT_TO_REC709.T


def _clog3_to_linear(rgb: np.ndarray) -> np.ndarray:
    y = np.asarray(rgb, dtype=np.float64)
    lin = np.where(
        y >= 0.15277891,
        (np.power(10.0, (y - 0.12240537) / 0.36726845) - 1.0) / 14.98325,
        np.where(
            y <= 0.09746547,
            -(np.power(10.0, (0.12783901 - y) / 0.36726845) - 1.0) / 14.98325,
            (y - 0.12512219) / 1.9754798,
        ),
    )
    return lin @ _M_CGAMUT_TO_REC709.T


def _tone_map_scene_to_display(lin_rgb: np.ndarray) -> np.ndarray:
    """Smooth photographic highlight rolloff preserving 18% mid-gray."""
    x = np.maximum(lin_rgb, 0.0)
    # Rational filmic rolloff calibrated so scene 0.18 -> ~0.18 display linear and inf -> 1.0
    a = 1.0
    b = 0.85
    mapped = (x * (a + 0.12 * x)) / (x * (b + 0.12 * x) + 0.95)
    return np.clip(mapped, 0.0, 1.0)


def apply_idt(rgb: np.ndarray, idt: str = "srgb") -> np.ndarray:
    """Convert encoded image RGB from the specified IDT into working sRGB [0, 1]."""
    key = idt.lower().strip()
    if key not in SUPPORTED_IDTS:
        raise ValueError(f"Unsupported IDT '{idt}'. Expected one of: {', '.join(SUPPORTED_IDTS)}")
    x = np.asarray(rgb, dtype=np.float64)
    if key == "srgb":
        return np.clip(x, 0.0, 1.0)
    if key == "rec709":
        return np.clip(linear_to_srgb(np.clip(_rec709_to_linear(x), 0.0, 1.0)), 0.0, 1.0)
    if key == "apple-log":
        lin = _apple_log_to_linear(x)
    elif key == "slog3":
        lin = _slog3_to_linear(x)
    elif key == "vlog":
        lin = _vlog_to_linear(x)
    else:
        lin = _clog3_to_linear(x)
    return np.clip(linear_to_srgb(_tone_map_scene_to_display(lin)), 0.0, 1.0)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB [0, 1] to CIE L*a*b* (D65)."""
    lin = srgb_to_linear(np.clip(rgb, 0.0, 1.0))
    xyz = lin @ _M_SRGB_TO_XYZ.T
    scaled = xyz / _D65_XYZ
    eps = (6.0 / 29.0) ** 3
    kappa_div = 1.0 / (3.0 * (6.0 / 29.0) ** 2)
    f = np.where(scaled > eps, np.cbrt(np.maximum(scaled, eps)), kappa_div * scaled + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def ciede2000(
    lab1: np.ndarray,
    lab2: np.ndarray,
    k_L: float = 1.0,
    k_C: float = 1.0,
    k_H: float = 1.0,
) -> np.ndarray:
    """Exact Sharma, Wu & Dalal (2005) CIEDE2000 color difference."""
    l1, a1, b1 = [np.asarray(lab1[..., i], dtype=np.float64) for i in range(3)]
    l2, a2, b2 = [np.asarray(lab2[..., i], dtype=np.float64) for i in range(3)]

    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_bar = 0.5 * (c1 + c2)
    c_bar7 = c_bar ** 7
    pow25_7 = 25.0 ** 7
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + pow25_7)))

    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = np.hypot(a1p, b1)
    c2p = np.hypot(a2p, b2)

    h1p = np.where((b1 == 0.0) & (a1p == 0.0), 0.0, np.degrees(np.arctan2(b1, a1p)) % 360.0)
    h2p = np.where((b2 == 0.0) & (a2p == 0.0), 0.0, np.degrees(np.arctan2(b2, a2p)) % 360.0)

    dlp = l2 - l1
    dcp = c2p - c1p

    prod_cp = c1p * c2p
    h_diff = h2p - h1p
    dhp = np.where(
        prod_cp == 0.0,
        0.0,
        np.where(
            np.abs(h_diff) <= 180.0,
            h_diff,
            np.where(h_diff > 180.0, h_diff - 360.0, h_diff + 360.0),
        ),
    )
    dHp = 2.0 * np.sqrt(prod_cp) * np.sin(np.radians(0.5 * dhp))

    l_bar_p = 0.5 * (l1 + l2)
    c_bar_p = 0.5 * (c1p + c2p)

    h_sum = h1p + h2p
    h_bar_p = np.where(
        prod_cp == 0.0,
        h_sum,
        np.where(
            np.abs(h1p - h2p) <= 180.0,
            0.5 * h_sum,
            np.where(h_sum < 360.0, 0.5 * (h_sum + 360.0), 0.5 * (h_sum - 360.0)),
        ),
    )

    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar_p - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar_p))
        + 0.32 * np.cos(np.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar_p - 63.0))
    )
    d_theta = 30.0 * np.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    c_bar_p7 = c_bar_p ** 7
    r_c = 2.0 * np.sqrt(c_bar_p7 / (c_bar_p7 + pow25_7))

    l_delta_sq = (l_bar_p - 50.0) ** 2
    s_l = 1.0 + (0.015 * l_delta_sq) / np.sqrt(20.0 + l_delta_sq)
    s_c = 1.0 + 0.045 * c_bar_p
    s_h = 1.0 + 0.015 * c_bar_p * t
    r_t = -np.sin(np.radians(2.0 * d_theta)) * r_c

    term_l = dlp / (k_L * s_l)
    term_c = dcp / (k_C * s_c)
    term_h = dHp / (k_H * s_h)
    return np.sqrt(term_l ** 2 + term_c ** 2 + term_h ** 2 + r_t * term_c * term_h)
