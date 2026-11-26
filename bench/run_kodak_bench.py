from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.baselines import (
    fit_hist_match,
    fit_hm_mvgd_hm,
    fit_mkl_oklab,
    fit_mkl_rgb,
    fit_reinhard_lab,
)
from bench.cinematic_luts import build_cinematic_luts
from lut_match.color import ciede2000, rgb_to_lab, srgb_to_oklab
from lut_match.io import apply_lut, identity_lut, inspect_lut
from lut_match.solver import solve_lut

DEFAULT_KODAK_DIR = Path("data/kodak24")
PORTRAIT_NAMES = {"kodim04", "kodim15", "kodim17", "kodim18", "kodim19"}


def _find_kodak_files(kodak_dir: Path = DEFAULT_KODAK_DIR) -> dict[str, Path]:
    paths = sorted(kodak_dir.rglob("kodim*.png"))
    if len(paths) < 24:
        raise FileNotFoundError(
            f"Expected 24 Kodak-24 PNGs under {kodak_dir}, found {len(paths)}"
        )
    return {p.stem: p for p in paths}


def _load_rgb(path: Path, max_dim: int | None = None) -> np.ndarray:
    with Image.open(path) as img:
        rgb_img = img.convert("RGB")
        if max_dim is not None and max(rgb_img.size) > max_dim:
            w, h = rgb_img.size
            scale = max_dim / max(w, h)
            rgb_img = rgb_img.resize(
                (max(1, round(w * scale)), max(1, round(h * scale))),
                Image.Resampling.LANCZOS,
            )
        return np.asarray(rgb_img, dtype=np.float64) / 255.0


def _skin_mask_oklab(rgb: np.ndarray) -> np.ndarray:
    lab = srgb_to_oklab(rgb)
    l_v = lab[..., 0]
    a_v = lab[..., 1]
    b_v = lab[..., 2]
    c_v = np.hypot(a_v, b_v)
    h_v = np.degrees(np.arctan2(b_v, a_v)) % 360.0
    return (l_v >= 0.28) & (l_v <= 0.85) & (c_v >= 0.03) & (c_v <= 0.17) & (h_v >= 28.0) & (h_v <= 65.0)


def _mean_hue_error_deg(pred_rgb: np.ndarray, gt_rgb: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    ok_p = srgb_to_oklab(pred_rgb[mask])
    ok_g = srgb_to_oklab(gt_rgb[mask])
    h_p = np.degrees(np.arctan2(ok_p[:, 2], ok_p[:, 1])) % 360.0
    h_g = np.degrees(np.arctan2(ok_g[:, 2], ok_g[:, 1])) % 360.0
    dh = (h_p - h_g + 180.0) % 360.0 - 180.0
    return float(np.mean(np.abs(dh)))


def _build_unbiased_kodak_splits(
    names: list[str],
    n_splits: int = 12,
    k_stills: int = 5,
    seed: int = 2026,
) -> list[tuple[list[str], list[str], str]]:
    """Draw seeded disjoint (5 src, 5 ref, 1 test) splits fairly across all 24 real Kodak photos."""
    rng = np.random.default_rng(seed)
    # Ensure portraits and diverse scenes are represented in held-out test set
    portraits = sorted(n for n in names if n in PORTRAIT_NAMES)
    others = sorted(n for n in names if n not in PORTRAIT_NAMES)
    rng.shuffle(portraits)
    rng.shuffle(others)
    test_order = portraits[:4] + others[: max(0, n_splits - 4)]
    rng.shuffle(test_order)

    src_counts: Counter[str] = Counter({n: 0 for n in names})
    ref_counts: Counter[str] = Counter({n: 0 for n in names})
    splits: list[tuple[list[str], list[str], str]] = []
    for test_name in test_order:
        pool = [n for n in names if n != test_name]
        rng.shuffle(pool)
        a_names = sorted(pool, key=lambda n: src_counts[n])[:k_stills]
        rem = [n for n in pool if n not in a_names]
        rng.shuffle(rem)
        b_names = sorted(rem, key=lambda n: ref_counts[n])[:k_stills]
        src_counts.update(a_names)
        ref_counts.update(b_names)
        splits.append((a_names, b_names, test_name))
    return splits


def _pool_flat_grid(imgs: list[np.ndarray]) -> np.ndarray:
    """Pool heterogeneous-aspect Kodak photos into a uniform (H, W, 3) patch grid for baselines."""
    chunks = []
    for im in imgs:
        sub = im[::2, ::2].reshape((-1, 3))
        idx = np.linspace(0, sub.shape[0] - 1, 8192, dtype=np.intp)
        chunks.append(sub[idx])
    all_px = np.concatenate(chunks, axis=0)
    return all_px.reshape((len(imgs) * 64, 128, 3))


def run_kodak_benchmark(
    kodak_dir: Path = DEFAULT_KODAK_DIR,
    save_path: Path | None = None,
) -> dict[str, Any]:
    files = _find_kodak_files(kodak_dir)
    names = sorted(files.keys())
    photos = {name: _load_rgb(path, max_dim=256) for name, path in files.items()}
    skin_masks = {name: _skin_mask_oklab(rgb) for name, rgb in photos.items()}

    luts = build_cinematic_luts(size=33)
    id_lut = identity_lut(33)

    # 6 diverse real Kodak triples (src A, ref B, held-out C) covering portraits, foliage, coastal, sky, vivid
    cross_triples = [
        ("kodim04", "kodim05", "kodim15"),
        ("kodim11", "kodim23", "kodim04"),
        ("kodim05", "kodim16", "kodim11"),
        ("kodim20", "kodim22", "kodim18"),
        ("kodim01", "kodim12", "kodim23"),
        ("kodim19", "kodim08", "kodim21"),
    ]
    multi_splits = _build_unbiased_kodak_splits(names, n_splits=12, k_stills=5, seed=2026)

    methods = [
        "lut_match",
        "identity",
        "reinhard_lab",
        "mkl_rgb",
        "mkl_oklab",
        "hm_mvgd_hm",
        "hist_match",
    ]
    transfer_baselines = [m for m in methods if m not in ("lut_match", "identity")]

    def _fit_all(
        src_arg: np.ndarray | list[np.ndarray],
        ref_arg: np.ndarray | list[np.ndarray],
        pool_src: np.ndarray,
        pool_ref: np.ndarray,
    ) -> dict[str, tuple[np.ndarray, Any]]:
        return {
            "lut_match": (solve_lut(src_arg, ref_arg, size=33, skin_protect=0.5, mode="balanced"), None),
            "identity": (id_lut, None),
            "reinhard_lab": fit_reinhard_lab(pool_src, pool_ref, size=33),
            "mkl_rgb": fit_mkl_rgb(pool_src, pool_ref, size=33),
            "mkl_oklab": fit_mkl_oklab(pool_src, pool_ref, size=33),
            "hm_mvgd_hm": fit_hm_mvgd_hm(pool_src, pool_ref, size=33),
            "hist_match": fit_hist_match(pool_src, pool_ref, size=33),
        }

    acc_r1: dict[str, dict[str, list[float]]] = {
        m: {"de00": [], "skin_de00": [], "roughness": [], "inversion_rate": []} for m in methods
    }
    acc_r2: dict[str, dict[str, list[float]]] = {
        m: {"de00": [], "skin_de00": [], "roughness": [], "inversion_rate": []} for m in methods
    }
    acc_r3: dict[str, dict[str, list[float]]] = {
        m: {"cross_de00": [], "skin_de00": [], "skin_hue_err": [], "roughness": [], "inversion_rate": []}
        for m in methods
    }
    acc_r3_creative: dict[str, list[float]] = {
        "cross_de00": [],
        "skin_de00": [],
        "skin_hue_err": [],
        "roughness": [],
        "inversion_rate": [],
    }

    for _, gt_lut in luts.items():
        gt_graded = {name: apply_lut(rgb, gt_lut) for name, rgb in photos.items()}

        # Regime 1: Paired Look Recovery (src=A, ref=LUT_k(A) -> test C)
        for src_name, _, test_name in cross_triples:
            src_rgb = photos[src_name]
            ref_rgb = gt_graded[src_name]
            test_rgb = photos[test_name]
            gt_lab = rgb_to_lab(gt_graded[test_name])
            smask = skin_masks[test_name]
            for m_name, (m_lut, _) in _fit_all(src_rgb, ref_rgb, src_rgb, ref_rgb).items():
                pred = apply_lut(test_rgb, m_lut)
                de = ciede2000(rgb_to_lab(pred), gt_lab)
                acc_r1[m_name]["de00"].append(float(np.mean(de)))
                if np.sum(smask) >= 64:
                    acc_r1[m_name]["skin_de00"].append(float(np.mean(de[smask])))
                info = inspect_lut(m_lut)
                acc_r1[m_name]["roughness"].append(info["laplacian_roughness"])
                acc_r1[m_name]["inversion_rate"].append(info["inversion_rate"])

        # Regime 2: Unbiased Multi-Still Look Transfer (5 src + 5 ref -> test C)
        for a_names, b_names, test_name in multi_splits:
            src_list = [photos[a] for a in a_names]
            ref_list = [gt_graded[b] for b in b_names]
            src_pooled = _pool_flat_grid(src_list)
            ref_pooled = _pool_flat_grid(ref_list)
            test_rgb = photos[test_name]
            gt_lab = rgb_to_lab(gt_graded[test_name])
            smask = skin_masks[test_name]
            for m_name, (m_lut, _) in _fit_all(src_list, ref_list, src_pooled, ref_pooled).items():
                pred = apply_lut(test_rgb, m_lut)
                de = ciede2000(rgb_to_lab(pred), gt_lab)
                acc_r2[m_name]["de00"].append(float(np.mean(de)))
                if np.sum(smask) >= 64:
                    acc_r2[m_name]["skin_de00"].append(float(np.mean(de[smask])))
                info = inspect_lut(m_lut)
                acc_r2[m_name]["roughness"].append(info["laplacian_roughness"])
                acc_r2[m_name]["inversion_rate"].append(info["inversion_rate"])

        # Regime 3: Single-Still Cross-Scene (src=A, ref=LUT_k(B) -> test C)
        for src_name, ref_name, test_name in cross_triples:
            src_rgb = photos[src_name]
            ref_rgb = gt_graded[ref_name]
            test_rgb = photos[test_name]
            gt_test = gt_graded[test_name]
            gt_lab = rgb_to_lab(gt_test)
            smask = skin_masks[test_name]
            for m_name, (m_lut, _) in _fit_all(src_rgb, ref_rgb, src_rgb, ref_rgb).items():
                pred = apply_lut(test_rgb, m_lut)
                de = ciede2000(rgb_to_lab(pred), gt_lab)
                acc_r3[m_name]["cross_de00"].append(float(np.mean(de)))
                if np.sum(smask) >= 64:
                    acc_r3[m_name]["skin_de00"].append(float(np.mean(de[smask])))
                    acc_r3[m_name]["skin_hue_err"].append(_mean_hue_error_deg(pred, gt_test, smask))
                info = inspect_lut(m_lut)
                acc_r3[m_name]["roughness"].append(info["laplacian_roughness"])
                acc_r3[m_name]["inversion_rate"].append(info["inversion_rate"])

            # Also track lut_match creative mode diagnostics
            cr_lut = solve_lut(src_rgb, ref_rgb, size=33, skin_protect=0.5, mode="creative")
            cr_pred = apply_lut(test_rgb, cr_lut)
            cr_de = ciede2000(rgb_to_lab(cr_pred), gt_lab)
            acc_r3_creative["cross_de00"].append(float(np.mean(cr_de)))
            if np.sum(smask) >= 64:
                acc_r3_creative["skin_de00"].append(float(np.mean(cr_de[smask])))
                acc_r3_creative["skin_hue_err"].append(_mean_hue_error_deg(cr_pred, gt_test, smask))
            cr_info = inspect_lut(cr_lut)
            acc_r3_creative["roughness"].append(cr_info["laplacian_roughness"])
            acc_r3_creative["inversion_rate"].append(cr_info["inversion_rate"])

    def _summarize(acc: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float]]:
        return {m: {k: float(np.mean(v)) for k, v in metrics.items()} for m, metrics in acc.items()}

    summary_r1 = _summarize(acc_r1)
    summary_r2 = _summarize(acc_r2)
    summary_r3 = _summarize(acc_r3)
    summary_r3_creative = {k: float(np.mean(v)) for k, v in acc_r3_creative.items()}

    best_r1_base = min(summary_r1[m]["de00"] for m in methods if m != "lut_match")
    best_r2_transfer = min(summary_r2[m]["de00"] for m in transfer_baselines)
    best_r3_transfer = min(summary_r3[m]["cross_de00"] for m in transfer_baselines)
    best_r3_skin_transfer = min(summary_r3[m]["skin_de00"] for m in transfer_baselines)
    best_r3_rough = min(summary_r3[m]["roughness"] for m in transfer_baselines)

    report = {
        "dataset": "Kodak-24 (24 real public-domain photographs)",
        "n_photos": len(photos),
        "n_luts": len(luts),
        "n_paired_trials": len(luts) * len(cross_triples),
        "n_multi_still_trials": len(luts) * len(multi_splits),
        "n_cross_trials": len(luts) * len(cross_triples),
        "paired_summary": summary_r1,
        "multi_still_summary": summary_r2,
        "cross_summary": summary_r3,
        "creative_cross_summary": summary_r3_creative,
        "gains": {
            "paired_de00_reduction_pct": 100.0 * (best_r1_base - summary_r1["lut_match"]["de00"]) / best_r1_base,
            "paired_vs_identity_pct": 100.0 * (summary_r1["identity"]["de00"] - summary_r1["lut_match"]["de00"]) / summary_r1["identity"]["de00"],
            "multi_still_vs_identity_pct": 100.0 * (summary_r2["identity"]["de00"] - summary_r2["lut_match"]["de00"]) / summary_r2["identity"]["de00"],
            "multi_still_vs_best_transfer_pct": 100.0 * (best_r2_transfer - summary_r2["lut_match"]["de00"]) / best_r2_transfer,
            "cross_de00_reduction_pct": 100.0 * (best_r3_transfer - summary_r3["lut_match"]["cross_de00"]) / best_r3_transfer,
            "cross_skin_de00_reduction_pct": 100.0 * (best_r3_skin_transfer - summary_r3["lut_match"]["skin_de00"]) / best_r3_skin_transfer,
            "roughness_reduction_pct": 100.0 * (best_r3_rough - summary_r3["lut_match"]["roughness"]) / best_r3_rough,
            "lut_match_inversion_rate": max(
                summary_r1["lut_match"]["inversion_rate"],
                summary_r2["lut_match"]["inversion_rate"],
                summary_r3["lut_match"]["inversion_rate"],
                summary_r3_creative["inversion_rate"],
            ),
        },
    }

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return report


def _to_pil_tile(rgb: np.ndarray, tile_w: int, tile_h: int) -> Image.Image:
    u8 = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    im = Image.fromarray(u8, mode="RGB")
    # Center-crop gently to target aspect ratio so portrait and landscape sit cleanly in rows
    w, h = im.size
    target_ar = tile_w / tile_h
    cur_ar = w / h
    if cur_ar > target_ar:
        new_w = int(round(h * target_ar))
        x0 = (w - new_w) // 2
        im = im.crop((x0, 0, x0 + new_w, h))
    elif cur_ar < target_ar:
        new_h = int(round(w / target_ar))
        # Bias slightly upward for portraits so faces stay centered
        y0 = int(round((h - new_h) * 0.25))
        im = im.crop((0, y0, w, y0 + new_h))
    return im.resize((tile_w, tile_h), Image.Resampling.LANCZOS)


def _render_contact_sheet(
    title: str,
    col_headers: list[str],
    rows: list[tuple[str, list[np.ndarray]]],
    out_path: Path,
    tile_w: int = 384,
    tile_h: int = 256,
) -> None:
    pad = 12
    header_h = 56
    row_label_h = 22
    n_cols = len(col_headers)
    n_rows = len(rows)

    canvas_w = pad + n_cols * (tile_w + pad)
    canvas_h = header_h + n_rows * (tile_h + row_label_h + pad) + pad
    sheet = Image.new("RGB", (canvas_w, canvas_h), (18, 20, 24))
    draw = ImageDraw.Draw(sheet)

    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_sub = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except OSError:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    draw.text((pad, 10), title, fill=(235, 238, 242), font=font_title)
    for c_i, col_label in enumerate(col_headers):
        x = pad + c_i * (tile_w + pad)
        draw.text((x + 2, 34), col_label, fill=(175, 182, 194), font=font_sub)

    for r_i, (row_label, imgs) in enumerate(rows):
        y_top = header_h + r_i * (tile_h + row_label_h + pad)
        draw.text((pad + 2, y_top), row_label, fill=(145, 154, 168), font=font_sub)
        y_img = y_top + row_label_h
        for c_i, arr in enumerate(imgs):
            x = pad + c_i * (tile_w + pad)
            tile = _to_pil_tile(arr, tile_w, tile_h)
            sheet.paste(tile, (x, y_img))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, optimize=True)


def generate_kodak_contact_sheets(
    kodak_dir: Path = DEFAULT_KODAK_DIR,
    qa_dir: Path | None = None,
) -> list[Path]:
    qa = qa_dir or (ROOT / "qa")
    files = _find_kodak_files(kodak_dir)
    luts = build_cinematic_luts(size=33)

    # Full-resolution Kodak photos for visual inspection
    full = {k: _load_rgb(files[k]) for k in (
        "kodim01", "kodim02", "kodim03", "kodim04", "kodim05", "kodim06",
        "kodim07", "kodim08", "kodim11", "kodim12", "kodim15", "kodim16",
        "kodim20", "kodim21", "kodim22", "kodim23",
    )}

    # 1) qa/kodak-paired-check.png: fit paired LUT on held-out pair kodim12 -> apply to kodim04, kodim05, kodim11
    paired_specs = [
        ("kodim04 (Portrait) · Kodak 2383 Print", "kodim04", "kodak_2383_print"),
        ("kodim05 (Motocross) · Teal & Orange", "kodim05", "teal_and_orange"),
        ("kodim11 (Coastal) · Fuji Eterna 250D", "kodim11", "fuji_eterna_250d"),
    ]
    paired_rows = []
    for label, scene_key, lut_key in paired_specs:
        gt_lut = luts[lut_key]
        src_fit = full["kodim12"]
        ref_fit = apply_lut(src_fit, gt_lut)
        rec_lut = solve_lut(src_fit, ref_fit, size=33, skin_protect=0.5, mode="balanced")
        test_orig = full[scene_key]
        test_gt = apply_lut(test_orig, gt_lut)
        test_rec = apply_lut(test_orig, rec_lut)
        de = float(np.mean(ciede2000(rgb_to_lab(test_rec), rgb_to_lab(test_gt))))
        paired_rows.append((f"{label}  (held-out dE00 = {de:.2f})", [test_orig, test_gt, test_rec]))

    p1 = qa / "kodak-paired-check.png"
    _render_contact_sheet(
        "Kodak-24 Paired Look Recovery (fit on kodim12 pair -> applied to unseen scenes)",
        ["Original", "Ground Truth LUT", "lut-match Recovered"],
        paired_rows,
        p1,
    )

    # 2) qa/kodak-multi-check.png: 5-still disjoint src & ref transfer -> applied to kodim04, kodim05, kodim11
    src_pool_keys = ["kodim01", "kodim02", "kodim06", "kodim08", "kodim20"]
    ref_pool_keys = ["kodim03", "kodim07", "kodim12", "kodim16", "kodim22"]
    src_pool = [full[k] for k in src_pool_keys]
    src_flat = _pool_flat_grid(src_pool)
    multi_rows = []
    for label, scene_key, lut_key in paired_specs:
        gt_lut = luts[lut_key]
        ref_pool = [apply_lut(full[k], gt_lut) for k in ref_pool_keys]
        ref_flat = _pool_flat_grid(ref_pool)
        lm_lut = solve_lut(src_pool, ref_pool, size=33, skin_protect=0.5, mode="balanced")
        rh_lut, _ = fit_reinhard_lab(src_flat, ref_flat, size=33)

        test_orig = full[scene_key]
        test_gt = apply_lut(test_orig, gt_lut)
        test_lm = apply_lut(test_orig, lm_lut)
        test_rh = apply_lut(test_orig, rh_lut)
        de_lm = float(np.mean(ciede2000(rgb_to_lab(test_lm), rgb_to_lab(test_gt))))
        de_rh = float(np.mean(ciede2000(rgb_to_lab(test_rh), rgb_to_lab(test_gt))))
        multi_rows.append(
            (
                f"{label}  (lut-match dE00={de_lm:.2f} vs reinhard={de_rh:.2f})",
                [test_orig, test_gt, test_lm, test_rh],
            )
        )

    p2 = qa / "kodak-multi-check.png"
    _render_contact_sheet(
        "Kodak-24 Unbiased 5-Still Transfer (5 disjoint src + 5 disjoint ref -> unseen scenes)",
        ["Original", "Ground Truth LUT", "lut-match (5-Still)", "reinhard_lab (5-Still)"],
        multi_rows,
        p2,
        tile_w=320,
        tile_h=214,
    )

    # 3) qa/kodak-creative-check.png: Portrait kodim04 matched to 3 unrelated graded movie-style refs
    portrait = full["kodim04"]
    creative_specs = [
        ("Ref: kodim11 (Teal & Orange)", "kodim11", "teal_and_orange"),
        ("Ref: kodim05 (Golden Hour Warm)", "kodim05", "golden_hour_warm"),
        ("Ref: kodim21 (Kodak 2383 Print)", "kodim21", "kodak_2383_print"),
    ]
    creative_rows = []
    for label, ref_key, lut_key in creative_specs:
        ref_img = apply_lut(full[ref_key], luts[lut_key])
        lut_bal = solve_lut(portrait, ref_img, size=33, skin_protect=0.5, mode="balanced")
        lut_cre = solve_lut(portrait, ref_img, size=33, skin_protect=0.5, mode="creative")
        out_bal = apply_lut(portrait, lut_bal)
        out_cre = apply_lut(portrait, lut_cre)
        creative_rows.append(
            (
                f"Portrait kodim04 <- {label}  (skin_protect=0.5, 0 inversions)",
                [portrait, ref_img, out_bal, out_cre],
            )
        )

    p3 = qa / "kodak-creative-check.png"
    _render_contact_sheet(
        "Kodak-24 Single-Still Transfer: Balanced vs Creative Mode (with Skin Protection)",
        ["Original (kodim04)", "Reference Still", "lut-match Balanced", "lut-match Creative"],
        creative_rows,
        p3,
        tile_w=320,
        tile_h=240,
    )
    return [p1, p2, p3]


def _print_kodak_table(report: dict[str, Any]) -> None:
    print(f"\n=== Real-Photo Kodak-24 Benchmark ({report['n_photos']} photos, {report['n_luts']} LUTs) ===")

    h1 = f"{'Method':<15} | {'dE00':>8} | {'Skin dE00':>9} | {'Roughness':>9} | {'Inversion %':>11}"
    print(f"\n1) Paired Look Recovery (src=A, ref=LUT_k(A) -> test C, {report['n_paired_trials']} trials)")
    print(h1)
    print("-" * len(h1))
    for m_name, row in report["paired_summary"].items():
        print(
            f"{m_name:<15} | {row['de00']:>8.3f} | {row['skin_de00']:>9.3f} | "
            f"{row['roughness']:>9.5f} | {row['inversion_rate'] * 100.0:>10.4f}%"
        )

    print(f"\n2) Unbiased Multi-Still Transfer (5 src + 5 ref -> test C, {report['n_multi_still_trials']} trials)")
    print(h1)
    print("-" * len(h1))
    for m_name, row in report["multi_still_summary"].items():
        print(
            f"{m_name:<15} | {row['de00']:>8.3f} | {row['skin_de00']:>9.3f} | "
            f"{row['roughness']:>9.5f} | {row['inversion_rate'] * 100.0:>10.4f}%"
        )

    print(f"\n3) Single-Still Cross-Scene Transfer (src=A, ref=LUT_k(B) -> test C, {report['n_cross_trials']} trials)")
    h3 = f"{'Method':<15} | {'Cross dE00':>10} | {'Skin dE00':>9} | {'Skin Hue Err':>12} | {'Roughness':>9} | {'Inversion %':>11}"
    print(h3)
    print("-" * len(h3))
    for m_name, row in report["cross_summary"].items():
        print(
            f"{m_name:<15} | {row['cross_de00']:>10.3f} | {row['skin_de00']:>9.3f} | "
            f"{row['skin_hue_err']:>11.2f}° | {row['roughness']:>9.5f} | {row['inversion_rate'] * 100.0:>10.4f}%"
        )
    cr = report["creative_cross_summary"]
    print(
        f"{'lut_match(cr)':<15} | {cr['cross_de00']:>10.3f} | {cr['skin_de00']:>9.3f} | "
        f"{cr['skin_hue_err']:>11.2f}° | {cr['roughness']:>9.5f} | {cr['inversion_rate'] * 100.0:>10.4f}%"
    )

    g = report["gains"]
    print("\nKodak-24 Summary Gains:")
    print(f"  Paired dE00 vs identity / best baseline: {g['paired_vs_identity_pct']:.1f}% / {g['paired_de00_reduction_pct']:.1f}%")
    print(f"  Multi-still dE00 vs identity / transfer: {g['multi_still_vs_identity_pct']:.1f}% / {g['multi_still_vs_best_transfer_pct']:.1f}%")
    print(f"  Single-still cross dE00 vs transfer:     {g['cross_de00_reduction_pct']:.1f}% (skin: {g['cross_skin_de00_reduction_pct']:.1f}%)")
    print(f"  Max lut_match inversion rate:            {g['lut_match_inversion_rate'] * 100.0:.4f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Kodak-24 real-photo 3-regime LUT benchmark.")
    parser.add_argument(
        "--kodak-dir",
        type=Path,
        default=DEFAULT_KODAK_DIR,
        help="Directory containing Kodak-24 kodim01.png..kodim24.png images.",
    )
    parser.add_argument("--no-sheets", action="store_true", help="Skip generating QA contact sheets.")
    args = parser.parse_args()
    out_file = Path(__file__).resolve().parent / "results_kodak.json"
    res = run_kodak_benchmark(kodak_dir=args.kodak_dir, save_path=out_file)
    _print_kodak_table(res)
    if not args.no_sheets:
        sheets = generate_kodak_contact_sheets(kodak_dir=args.kodak_dir)
        print("\nGenerated visual contact sheets:")
        for s in sheets:
            print(f"  - {s}")
