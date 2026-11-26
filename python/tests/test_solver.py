import numpy as np
import pytest

from lut_match.color import _M_SGAMUT3CINE_TO_REC709, apply_idt, ciede2000, rgb_to_lab, srgb_to_linear, srgb_to_oklab
from lut_match.io import apply_lut, identity_lut, inspect_lut
from lut_match.solver import _is_paired_shot, solve_lut


def _synthetic_photo(seed: int, tint: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.05, 0.92, size=(64, 64, 3))
    # Include a realistic skin patch + neutral patch
    base[:20, :20] = np.clip(np.array([0.78, 0.58, 0.46]) + rng.normal(0.0, 0.03, (20, 20, 3)), 0.05, 0.95)
    base[20:40, :20] = np.repeat(np.linspace(0.1, 0.9, 20)[:, None], 3, axis=1)[None, :, :]
    return np.clip(base * np.array(tint)[None, None, :], 0.0, 1.0)


def test_zero_inversions_across_20_random_and_adversarial_solves():
    rng = np.random.default_rng(99)
    for trial in range(20):
        src = _synthetic_photo(seed=100 + trial)
        if trial % 4 == 0:
            # Extreme monochromatic adversarial reference
            ref = np.zeros((64, 64, 3), dtype=np.float64)
            ref[..., trial % 3] = rng.uniform(0.5, 0.95, size=(64, 64))
            ref[:12, :12] = np.array([0.75, 0.55, 0.44])
        else:
            tint = tuple(rng.uniform(0.7, 1.3, size=3))
            ref = _synthetic_photo(seed=200 + trial, tint=tint)

        size = (17, 33, 65)[trial % 3] if trial < 3 else 17
        lut = solve_lut(
            src_rgb=src,
            ref_rgbs=ref,
            size=size,
            strength=float(rng.uniform(0.4, 1.0)),
            skin_protect=float(rng.uniform(0.0, 1.0)),
            shadow_lift=float(rng.uniform(-0.1, 0.2)),
            highlight_rolloff=float(rng.uniform(-0.1, 0.2)),
            smoothness=float(rng.uniform(0.0, 0.6)),
        )
        metrics = inspect_lut(lut)
        assert metrics["inversion_rate"] == 0.0, f"Trial {trial} had inversions: {metrics}"


def test_strength_and_skin_protection_behavior():
    src = _synthetic_photo(seed=1)
    # Reference strongly tinted green/cyan
    ref = _synthetic_photo(seed=2, tint=(0.65, 1.15, 1.10))

    lut_zero = solve_lut(src, ref, size=17, strength=0.0)
    assert np.allclose(lut_zero, identity_lut(17), atol=1e-12)

    lut_unprot = solve_lut(src, ref, size=17, skin_protect=0.0)
    lut_prot = solve_lut(src, ref, size=17, skin_protect=1.0)
    skin_patch = np.full((10, 10, 3), [0.76, 0.56, 0.44])
    h_in = np.degrees(np.arctan2(srgb_to_oklab(skin_patch)[0, 0, 2], srgb_to_oklab(skin_patch)[0, 0, 1]))
    ok_unprot = srgb_to_oklab(apply_lut(skin_patch, lut_unprot))
    ok_prot = srgb_to_oklab(apply_lut(skin_patch, lut_prot))
    dh_unprot = abs(((np.degrees(np.arctan2(ok_unprot[0, 0, 2], ok_unprot[0, 0, 1])) - h_in + 180) % 360) - 180)
    dh_prot = abs(((np.degrees(np.arctan2(ok_prot[0, 0, 2], ok_prot[0, 0, 1])) - h_in + 180) % 360) - 180)
    assert dh_prot < dh_unprot


def test_multi_reference_averaging_and_validation():
    src = _synthetic_photo(seed=10)
    ref_warm = _synthetic_photo(seed=11, tint=(1.15, 1.0, 0.85))
    ref_cool = _synthetic_photo(seed=12, tint=(0.85, 1.0, 1.15))

    lut_multi = solve_lut(src, [ref_warm, ref_cool], size=17)
    assert lut_multi.shape == (17, 17, 17, 3)
    assert inspect_lut(lut_multi)["inversion_rate"] == 0.0

    src_alt = _synthetic_photo(seed=13)
    lut_multi_both = solve_lut([src, src_alt], [ref_warm, ref_cool], size=17)
    assert lut_multi_both.shape == (17, 17, 17, 3)
    assert inspect_lut(lut_multi_both)["inversion_rate"] == 0.0

    with pytest.raises(ValueError):
        solve_lut([], ref_warm, size=17)
    with pytest.raises(ValueError):
        solve_lut(src, ref_warm, size=15)
    with pytest.raises(ValueError):
        solve_lut(src, [], size=17)
    with pytest.raises(ValueError):
        solve_lut(src, ref_warm, size=17, strength=1.5)


def test_paired_detection_rejects_unrelated_landscapes_ramps_and_aspect_mismatch():
    h, w = 128, 128
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ny = yy / (h - 1)
    nx = xx / (w - 1)

    # 1) Unrelated horizon landscapes (blue sky/green hills vs warm sunset/red canyon)
    sky_mask_1 = ny < 0.45 + 0.05 * np.sin(2.0 * np.pi * 2.0 * nx)
    tex_1 = 0.5 + 0.5 * np.sin(2.0 * np.pi * 5.0 * nx) * np.cos(2.0 * np.pi * 4.0 * ny)
    landscape_1 = np.where(
        sky_mask_1[..., None],
        np.stack([0.30 + 0.25 * ny, 0.55 + 0.20 * ny, 0.88 - 0.10 * ny], axis=-1),
        np.stack([0.20 + 0.12 * tex_1, 0.45 + 0.25 * tex_1, 0.18 + 0.08 * tex_1], axis=-1),
    )

    sky_mask_2 = ny < 0.46 + 0.05 * np.cos(2.0 * np.pi * 3.0 * nx + 1.1)
    tex_2 = 0.5 + 0.5 * np.sin(2.0 * np.pi * 7.0 * nx + 0.7) * np.cos(2.0 * np.pi * 6.0 * ny + 0.4)
    landscape_2 = np.where(
        sky_mask_2[..., None],
        np.stack([0.90 - 0.12 * ny, 0.52 + 0.18 * ny, 0.28 + 0.14 * ny], axis=-1),
        np.stack([0.62 + 0.22 * tex_2, 0.24 + 0.12 * tex_2, 0.16 + 0.06 * tex_2], axis=-1),
    )

    assert not _is_paired_shot(landscape_1, landscape_2)
    lut_land = solve_lut(landscape_1, landscape_2, size=17)
    assert inspect_lut(lut_land)["inversion_count"] == 0

    # 2) Unrelated linear 1D ramps & mismatched aspect ratios (16:9 vs 4:5)
    ramp_16_9 = np.repeat(np.linspace(0.05, 0.95, 160)[None, :, None], 90, axis=0)
    ramp_16_9 = np.concatenate([ramp_16_9, 0.5 * ramp_16_9, 0.25 * ramp_16_9], axis=-1)
    ramp_4_5 = np.repeat(np.linspace(0.10, 0.90, 96)[None, :, None], 120, axis=0)
    ramp_4_5 = np.concatenate([0.2 * ramp_4_5, 0.8 * ramp_4_5, ramp_4_5], axis=-1)

    assert not _is_paired_shot(ramp_16_9, ramp_4_5)
    lut_aspect = solve_lut(ramp_16_9, ramp_4_5, size=17)
    assert inspect_lut(lut_aspect)["inversion_count"] == 0

    # Same-aspect 1D smooth ramps (no 2D structure) must also be rejected
    ramp_sq_a = np.repeat(np.linspace(0.05, 0.95, 128)[:, None, None], 128, axis=1)
    ramp_sq_a = np.repeat(ramp_sq_a, 3, axis=2)
    ramp_sq_b = np.clip(ramp_sq_a * np.array([1.05, 0.95, 0.85]), 0.0, 1.0)
    assert not _is_paired_shot(ramp_sq_a, ramp_sq_b)
    assert inspect_lut(solve_lut(ramp_sq_a, ramp_sq_b, size=17))["inversion_count"] == 0

    # 3) True same-scene graded pair (A vs LUT(A)) still triggers paired mode and recovers dE00 < 2.0
    graded_1 = np.clip(0.90 * landscape_1 * np.array([1.08, 0.98, 0.88]) + 0.03, 0.0, 1.0)
    assert _is_paired_shot(landscape_1, graded_1)
    lut_paired = solve_lut(landscape_1, graded_1, size=33, skin_protect=0.0)
    assert inspect_lut(lut_paired)["inversion_count"] == 0
    rec = apply_lut(landscape_1, lut_paired)
    de00 = float(np.mean(ciede2000(rgb_to_lab(rec), rgb_to_lab(graded_1))))
    assert de00 < 2.0, f"Expected paired recovery dE00 < 2.0, got {de00:.3f}"


def _inv_slog3(srgb: np.ndarray) -> np.ndarray:
    """Encode display sRGB into synthetic S-Log3 / S-Gamut3.Cine inverting apply_idt(..., 'slog3')."""
    disp = srgb_to_linear(np.clip(srgb, 0.0, 0.96))
    a = 0.12 * (1.0 - disp)
    b = 1.0 - 0.85 * disp
    c = -0.95 * disp
    rec709 = (-b + np.sqrt(np.maximum(b * b - 4.0 * a * c, 0.0))) / np.maximum(2.0 * a, 1e-12)
    sgamut = rec709 @ np.linalg.inv(_M_SGAMUT3CINE_TO_REC709).T
    t = np.where(
        sgamut >= 0.01125,
        420.0 + np.log10(np.maximum((sgamut + 0.01) / 0.19, 1e-9)) * 261.5,
        sgamut * (171.2102946929 - 95.0) / 0.01125 + 95.0,
    )
    return np.clip(t / 1023.0, 0.0, 1.0)


def test_slog3_idt_baked_into_lattice_end_to_end():
    src_srgb = _synthetic_photo(seed=42)
    # Smooth warm film look target derived from the display-referred image
    target_srgb = np.clip(0.92 * src_srgb * np.array([1.06, 1.00, 0.90]) + 0.03, 0.0, 1.0)
    slog3_img = _inv_slog3(src_srgb)

    lut_slog3 = solve_lut(slog3_img, target_srgb, size=33, idt="slog3", skin_protect=0.0)
    lut_srgb_lie = solve_lut(slog3_img, target_srgb, size=33, idt="srgb", skin_protect=0.0)
    lut_rec709 = solve_lut(src_srgb, target_srgb, size=17, idt="rec709", skin_protect=0.0)

    assert inspect_lut(lut_srgb_lie)["inversion_count"] == 0
    assert inspect_lut(lut_rec709)["inversion_count"] == 0

    # Applying the S-Log3 LUT directly to the S-Log3 image must recover target_srgb accurately
    graded_slog3 = apply_lut(slog3_img, lut_slog3)
    graded_lie = apply_lut(slog3_img, lut_srgb_lie)

    de_slog3 = float(np.mean(ciede2000(rgb_to_lab(graded_slog3), rgb_to_lab(target_srgb))))
    de_lie = float(np.mean(ciede2000(rgb_to_lab(graded_lie), rgb_to_lab(target_srgb))))
    assert de_slog3 < 3.0, f"Expected S-Log3 LUT mean dE00 < 3.0, got {de_slog3:.3f}"
    assert de_slog3 < de_lie * 0.5, f"Expected slog3 ({de_slog3:.3f}) to beat lying srgb ({de_lie:.3f})"

    # strength=0.0 with idt='slog3' emits the clean IDT Rec.709 conversion
    lut_zero_slog3 = solve_lut(slog3_img, target_srgb, size=17, strength=0.0, idt="slog3")
    expected_zero = apply_idt(identity_lut(17), idt="slog3")
    assert np.allclose(lut_zero_slog3, expected_zero, atol=1e-12)


def test_all_idts_fine_grid_monotonicity_continuity_and_spec_constants():
    from lut_match.color import SUPPORTED_IDTS

    # Verify official Sony S-Gamut3.Cine -> Rec.709 row 0 (G = 0.225, 0.800)
    assert np.allclose(_M_SGAMUT3CINE_TO_REC709[0], [1.62694741, -0.54013854, -0.08680887], atol=1e-6)
    assert np.allclose(_M_SGAMUT3CINE_TO_REC709.sum(axis=1), [1.0, 1.0, 1.0], atol=1e-9)

    # Fine 2001-point neutral ramp across all 6 IDTs: strictly non-decreasing and C0 continuous
    ramp = np.repeat(np.linspace(0.0, 1.0, 2001)[:, None], 3, axis=1)
    for idt in SUPPORTED_IDTS:
        out = apply_idt(ramp, idt=idt)
        diffs = np.diff(out[:, 0])
        assert float(np.min(diffs)) >= -1e-9, f"IDT '{idt}' dropped backwards: min diff={float(np.min(diffs))}"
        assert float(np.max(diffs)) < 0.025, f"IDT '{idt}' discontinuity: max jump={float(np.max(diffs))}"

    # Explicit Canon C-Log3 upper junction regression check around y = 0.15277891
    clog_pair = apply_idt(np.array([[0.1527] * 3, [0.1528] * 3]), idt="clog3")[:, 0]
    assert clog_pair[1] > clog_pair[0]
    assert (clog_pair[1] - clog_pair[0]) < 2e-3


def test_creative_mode_expressiveness_skin_guard_and_zero_inversions():
    src = _synthetic_photo(seed=301)
    ref = _synthetic_photo(seed=302, tint=(0.72, 1.05, 1.18))
    ref = np.clip(0.88 * np.power(ref, 1.18) + 0.04, 0.0, 1.0)

    lut_bal = solve_lut(src, ref, size=33, skin_protect=0.5, mode="balanced")
    lut_cre = solve_lut(src, ref, size=33, skin_protect=0.5, mode="creative")
    lut_cre_unprot = solve_lut(src, ref, size=33, skin_protect=0.0, mode="creative")

    assert inspect_lut(lut_bal)["inversion_count"] == 0
    assert inspect_lut(lut_cre)["inversion_count"] == 0
    assert inspect_lut(lut_cre_unprot)["inversion_count"] == 0

    ident = identity_lut(33)
    diff_bal = float(np.mean(np.abs(lut_bal - ident)))
    diff_cre = float(np.mean(np.abs(lut_cre - ident)))
    assert diff_cre > diff_bal * 1.25, f"Expected creative ({diff_cre:.4f}) > balanced ({diff_bal:.4f})"

    # Skin protection actively reduces hue shift in creative mode
    skin_patch = np.full((8, 8, 3), [0.76, 0.56, 0.44])
    h_in = np.degrees(np.arctan2(srgb_to_oklab(skin_patch)[0, 0, 2], srgb_to_oklab(skin_patch)[0, 0, 1]))
    ok_p = srgb_to_oklab(apply_lut(skin_patch, lut_cre))
    ok_u = srgb_to_oklab(apply_lut(skin_patch, lut_cre_unprot))
    dh_p = abs(((np.degrees(np.arctan2(ok_p[0, 0, 2], ok_p[0, 0, 1])) - h_in + 180) % 360) - 180)
    dh_u = abs(((np.degrees(np.arctan2(ok_u[0, 0, 2], ok_u[0, 0, 1])) - h_in + 180) % 360) - 180)
    assert dh_p < dh_u

    with pytest.raises(ValueError):
        solve_lut(src, ref, size=17, mode="invalid")


def test_hue_harmonic_captures_selective_filmic_hue_shift():
    from lut_match.color import oklab_to_srgb

    # Build a smooth selective hue-vs-hue & hue-vs-sat film LUT (greens -> olive/cyan, reds -> amber)
    g = identity_lut(33)
    lab = srgb_to_oklab(g)
    a, b = lab[..., 1], lab[..., 2]
    c = np.hypot(a, b)
    th = np.arctan2(b, a)
    gate = (c * c) / (c * c + 0.03 * 0.03)
    dtheta = np.radians(14.0 * np.sin(2.0 * th) - 8.0 * np.cos(th)) * gate
    sat_scale = np.exp((0.18 * np.cos(2.0 * th) + 0.10 * np.sin(th)) * gate)
    cd, sd = np.cos(dtheta), np.sin(dtheta)
    lab_out = lab.copy()
    lab_out[..., 1] = sat_scale * (cd * a - sd * b)
    lab_out[..., 2] = sat_scale * (sd * a + cd * b)
    selective_lut = np.clip(oklab_to_srgb(lab_out), 0.0, 1.0)

    # Rich color wheel + 2D textured synthetic image for paired recovery
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float64) / 127.0
    ang = np.arctan2(yy - 0.5, xx - 0.5)
    rad = np.clip(np.hypot(xx - 0.5, yy - 0.5) * 1.9, 0.0, 1.0)
    tex = 0.72 + 0.28 * np.sin(2.0 * np.pi * 6.0 * xx) * np.cos(2.0 * np.pi * 5.0 * yy)
    src = np.clip(
        np.stack(
            [
                0.5 + 0.42 * rad * np.cos(ang),
                0.5 + 0.42 * rad * np.cos(ang - 2.094),
                0.5 + 0.42 * rad * np.cos(ang + 2.094),
            ],
            axis=-1,
        )
        * (0.45 + 0.55 * yy[..., None])
        * tex[..., None],
        0.02,
        0.98,
    )
    ref = apply_lut(src, selective_lut)
    assert _is_paired_shot(src, ref)
    rec = solve_lut(src, ref, size=33, skin_protect=0.0)
    assert inspect_lut(rec)["inversion_count"] == 0

    test_rgb = _synthetic_photo(seed=777)
    pred = apply_lut(test_rgb, rec)
    gt = apply_lut(test_rgb, selective_lut)
    de_rec = float(np.mean(ciede2000(rgb_to_lab(pred), rgb_to_lab(gt))))
    de_id = float(np.mean(ciede2000(rgb_to_lab(test_rgb), rgb_to_lab(gt))))
    assert de_rec < 1.2, f"Expected selective hue-harmonic paired recovery dE00 < 1.2, got {de_rec:.3f}"
    assert de_rec < de_id * 0.35
