from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BenchScene:
    name: str
    rgb: np.ndarray
    skin_mask: np.ndarray
    is_adversarial: bool = False


def _smooth_texture(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    nx = xx / w
    ny = yy / h
    acc = np.zeros((h, w), dtype=np.float64)
    for freq in (2.0, 5.0, 11.0, 23.0):
        px = rng.uniform(0, 2 * np.pi)
        py = rng.uniform(0, 2 * np.pi)
        ang = rng.uniform(0, 2 * np.pi)
        u = nx * np.cos(ang) + ny * np.sin(ang)
        v = -nx * np.sin(ang) + ny * np.cos(ang)
        acc += (np.sin(2 * np.pi * freq * u + px) * np.cos(2 * np.pi * freq * v + py)) / freq
    return acc / (np.max(np.abs(acc)) + 1e-8)


def _shade_patch(base_rgb: np.ndarray, tex: np.ndarray, l_low: float, l_high: float) -> np.ndarray:
    shade = l_low + (l_high - l_low) * (0.5 + 0.5 * tex)
    return np.clip(base_rgb[None, None, :] * shade[..., None], 0.02, 0.98)


def _make_portrait(
    name: str,
    skin_rgb: tuple[float, float, float],
    bg_top: tuple[float, float, float],
    bg_bot: tuple[float, float, float],
    cloth_rgb: tuple[float, float, float],
    accent_rgb: tuple[float, float, float],
    seed: int,
    h: int = 256,
    w: int = 256,
    bg_fraction_large: bool = False,
    is_adversarial: bool = False,
) -> BenchScene:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ny = yy / (h - 1)
    nx = xx / (w - 1)
    tex = _smooth_texture(h, w, seed)
    tex2 = _smooth_texture(h, w, seed + 101)

    # Smooth background gradient with lighting variation
    bg = (
        (1.0 - ny[..., None]) * np.array(bg_top)[None, None, :]
        + ny[..., None] * np.array(bg_bot)[None, None, :]
    )
    bg = np.clip(bg * (0.88 + 0.16 * tex[..., None] + 0.08 * tex2[..., None]), 0.01, 0.99)

    img = bg.copy()
    # Neutral grey reference card / wall strip on side (small in adversarial scenes)
    neu_mask = (nx > 0.05) & (nx < (0.14 if bg_fraction_large else 0.25)) & (ny > 0.15) & (ny < 0.85)
    neu_val = 0.18 + 0.65 * (ny - 0.15) / 0.70 + 0.03 * tex
    img[neu_mask] = np.clip(np.repeat(neu_val[neu_mask, None], 3, axis=1), 0.03, 0.96)

    # Accent object (foliage/prop/sky patch)
    acc_mask = ((nx - 0.82) ** 2 + (ny - 0.25) ** 2) < (0.012 if bg_fraction_large else 0.035)
    img[acc_mask] = _shade_patch(np.array(accent_rgb), tex2, 0.60, 1.15)[acc_mask]

    # Clothing / torso
    torso_r = 0.20 if bg_fraction_large else 0.28
    torso = ((nx - 0.52) / torso_r) ** 2 + ((ny - 0.92) / 0.28) ** 2 < 1.0
    img[torso] = _shade_patch(np.array(cloth_rgb), tex, 0.55, 1.12)[torso]

    # Portrait face / skin oval with natural shading (highlights, midtones, chin shadow)
    rx = 0.14 if bg_fraction_large else 0.19
    ry = 0.19 if bg_fraction_large else 0.25
    dx = (nx - 0.52) / rx
    dy = (ny - 0.48) / ry
    r2 = dx ** 2 + dy ** 2
    skin_mask = r2 < 1.0
    # 3D-like spherical Lambertian shading + subtle warm/cool micro-variation
    nz = np.sqrt(np.maximum(1.0 - r2, 0.0))
    lambert = 0.55 + 0.55 * np.clip(0.35 * (-dx) + 0.40 * (-dy) + 0.75 * nz, 0.0, 1.0)
    skin_base = np.array(skin_rgb)[None, None, :] * lambert[..., None]
    skin_base *= 1.0 + 0.03 * tex[..., None]
    img[skin_mask] = np.clip(skin_base[skin_mask], 0.03, 0.97)

    return BenchScene(name=name, rgb=img, skin_mask=skin_mask, is_adversarial=is_adversarial)


def _make_landscape(
    name: str,
    sky_top: tuple[float, float, float],
    sky_bot: tuple[float, float, float],
    mid_rgb: tuple[float, float, float],
    fg_rgb: tuple[float, float, float],
    accent_rgb: tuple[float, float, float],
    seed: int,
    h: int = 256,
    w: int = 256,
) -> BenchScene:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ny = yy / (h - 1)
    nx = xx / (w - 1)
    tex = _smooth_texture(h, w, seed)
    tex2 = _smooth_texture(h, w, seed + 77)

    img = np.zeros((h, w, 3), dtype=np.float64)
    sky_zone = ny < (0.45 + 0.06 * tex)
    mid_zone = (~sky_zone) & (ny < (0.72 + 0.05 * tex2))
    fg_zone = ~(sky_zone | mid_zone)

    sky = (1.0 - (ny / 0.5)[..., None]) * np.array(sky_top)[None, None, :] + (ny / 0.5)[..., None] * np.array(sky_bot)[None, None, :]
    img[sky_zone] = np.clip(sky[sky_zone] * (0.92 + 0.12 * tex2[sky_zone, None]), 0.02, 0.99)
    img[mid_zone] = _shade_patch(np.array(mid_rgb), tex, 0.45, 1.15)[mid_zone]
    img[fg_zone] = _shade_patch(np.array(fg_rgb), tex2, 0.38, 1.18)[fg_zone]

    # Diverse color accents (buildings/rocks/flowers/neutrals)
    for cx, cy, rad, col in [
        (0.22, 0.62, 0.07, accent_rgb),
        (0.75, 0.58, 0.08, (0.55, 0.55, 0.56)),
        (0.50, 0.80, 0.09, (0.22, 0.21, 0.20)),
    ]:
        m = ((nx - cx) ** 2 + (ny - cy) ** 2) < rad ** 2
        img[m] = _shade_patch(np.array(col), tex, 0.50, 1.20)[m]

    return BenchScene(name=name, rgb=np.clip(img, 0.0, 1.0), skin_mask=np.zeros((h, w), dtype=bool))


def _make_market(h: int = 256, w: int = 256) -> BenchScene:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ny = yy / (h - 1)
    nx = xx / (w - 1)
    tex = _smooth_texture(h, w, 901)
    palette = [
        (0.76, 0.22, 0.18),
        (0.84, 0.62, 0.18),
        (0.22, 0.55, 0.26),
        (0.18, 0.42, 0.72),
        (0.56, 0.28, 0.58),
        (0.45, 0.45, 0.46),
        (0.85, 0.82, 0.76),
        (0.20, 0.19, 0.18),
        (0.74, 0.48, 0.34),
    ]
    img = np.zeros((h, w, 3), dtype=np.float64)
    grid_x = np.clip((nx * 3.0).astype(int), 0, 2)
    grid_y = np.clip((ny * 3.0).astype(int), 0, 2)
    for r in range(3):
        for c in range(3):
            m = (grid_y == r) & (grid_x == c)
            col = np.array(palette[r * 3 + c])
            img[m] = _shade_patch(col, tex, 0.42, 1.22)[m]
    return BenchScene(name="colorful_market", rgb=np.clip(img, 0.0, 1.0), skin_mask=np.zeros((h, w), dtype=bool))


def build_test_scenes() -> dict[str, BenchScene]:
    """Generate 16 diverse 256x256 float RGB test scenes (13 natural + 3 adversarial)."""
    scenes = [
        _make_portrait(
            "portrait_fair_neutral",
            skin_rgb=(0.86, 0.67, 0.57),
            bg_top=(0.48, 0.52, 0.56),
            bg_bot=(0.24, 0.26, 0.28),
            cloth_rgb=(0.22, 0.28, 0.38),
            accent_rgb=(0.30, 0.46, 0.28),
            seed=11,
        ),
        _make_portrait(
            "portrait_medium_warm",
            skin_rgb=(0.72, 0.50, 0.37),
            bg_top=(0.66, 0.58, 0.48),
            bg_bot=(0.30, 0.26, 0.22),
            cloth_rgb=(0.56, 0.48, 0.36),
            accent_rgb=(0.28, 0.48, 0.62),
            seed=12,
        ),
        _make_portrait(
            "portrait_deep_cool",
            skin_rgb=(0.42, 0.27, 0.20),
            bg_top=(0.36, 0.46, 0.58),
            bg_bot=(0.16, 0.20, 0.26),
            cloth_rgb=(0.68, 0.66, 0.62),
            accent_rgb=(0.70, 0.32, 0.24),
            seed=13,
        ),
        _make_portrait(
            "portrait_olive_studio",
            skin_rgb=(0.76, 0.60, 0.46),
            bg_top=(0.62, 0.62, 0.60),
            bg_bot=(0.38, 0.38, 0.38),
            cloth_rgb=(0.18, 0.20, 0.22),
            accent_rgb=(0.62, 0.42, 0.26),
            seed=14,
        ),
        _make_landscape(
            "foliage_nature",
            sky_top=(0.36, 0.60, 0.86),
            sky_bot=(0.70, 0.82, 0.92),
            mid_rgb=(0.24, 0.46, 0.22),
            fg_rgb=(0.34, 0.52, 0.24),
            accent_rgb=(0.56, 0.40, 0.24),
            seed=21,
        ),
        _make_landscape(
            "sky_architecture",
            sky_top=(0.22, 0.48, 0.84),
            sky_bot=(0.66, 0.78, 0.92),
            mid_rgb=(0.64, 0.62, 0.58),
            fg_rgb=(0.36, 0.37, 0.39),
            accent_rgb=(0.74, 0.36, 0.26),
            seed=22,
        ),
        _make_landscape(
            "urban_neutrals",
            sky_top=(0.54, 0.58, 0.62),
            sky_bot=(0.72, 0.74, 0.75),
            mid_rgb=(0.44, 0.43, 0.42),
            fg_rgb=(0.24, 0.25, 0.26),
            accent_rgb=(0.66, 0.48, 0.28),
            seed=23,
        ),
        _make_portrait(
            "high_key_studio",
            skin_rgb=(0.88, 0.72, 0.62),
            bg_top=(0.90, 0.90, 0.89),
            bg_bot=(0.76, 0.76, 0.75),
            cloth_rgb=(0.84, 0.80, 0.74),
            accent_rgb=(0.48, 0.62, 0.74),
            seed=31,
        ),
        _make_landscape(
            "low_key_night",
            sky_top=(0.06, 0.08, 0.14),
            sky_bot=(0.12, 0.16, 0.24),
            mid_rgb=(0.18, 0.20, 0.26),
            fg_rgb=(0.10, 0.11, 0.13),
            accent_rgb=(0.76, 0.44, 0.22),
            seed=32,
        ),
        _make_market(),
        _make_landscape(
            "sunset_coast",
            sky_top=(0.34, 0.44, 0.68),
            sky_bot=(0.88, 0.62, 0.40),
            mid_rgb=(0.24, 0.44, 0.54),
            fg_rgb=(0.68, 0.56, 0.42),
            accent_rgb=(0.42, 0.32, 0.24),
            seed=33,
        ),
        _make_landscape(
            "tungsten_cafe",
            sky_top=(0.52, 0.40, 0.28),
            sky_bot=(0.72, 0.56, 0.38),
            mid_rgb=(0.48, 0.36, 0.26),
            fg_rgb=(0.26, 0.20, 0.16),
            accent_rgb=(0.28, 0.44, 0.42),
            seed=34,
        ),
        _make_landscape(
            "autumn_park",
            sky_top=(0.42, 0.58, 0.78),
            sky_bot=(0.76, 0.82, 0.86),
            mid_rgb=(0.70, 0.44, 0.20),
            fg_rgb=(0.46, 0.34, 0.20),
            accent_rgb=(0.28, 0.44, 0.22),
            seed=35,
        ),
        # 3 Adversarial dominant-color scenes with skin subject
        _make_portrait(
            "adv_red_wall",
            skin_rgb=(0.78, 0.58, 0.46),
            bg_top=(0.86, 0.12, 0.12),
            bg_bot=(0.62, 0.08, 0.09),
            cloth_rgb=(0.24, 0.24, 0.26),
            accent_rgb=(0.82, 0.14, 0.14),
            seed=41,
            bg_fraction_large=True,
            is_adversarial=True,
        ),
        _make_portrait(
            "adv_cyan_fog",
            skin_rgb=(0.74, 0.54, 0.42),
            bg_top=(0.12, 0.82, 0.86),
            bg_bot=(0.08, 0.58, 0.66),
            cloth_rgb=(0.22, 0.26, 0.28),
            accent_rgb=(0.14, 0.78, 0.82),
            seed=42,
            bg_fraction_large=True,
            is_adversarial=True,
        ),
        _make_portrait(
            "adv_green_wall",
            skin_rgb=(0.80, 0.60, 0.48),
            bg_top=(0.14, 0.78, 0.22),
            bg_bot=(0.10, 0.56, 0.16),
            cloth_rgb=(0.25, 0.25, 0.25),
            accent_rgb=(0.16, 0.74, 0.24),
            seed=43,
            bg_fraction_large=True,
            is_adversarial=True,
        ),
    ]
    return {s.name: s for s in scenes}
