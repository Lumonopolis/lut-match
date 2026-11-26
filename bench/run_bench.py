from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

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
from bench.scenes import build_test_scenes
from lut_match.color import ciede2000, rgb_to_lab, srgb_to_oklab
from lut_match.io import apply_lut, identity_lut, inspect_lut
from lut_match.solver import solve_lut


def _mean_hue_error_deg(pred_rgb: np.ndarray, gt_rgb: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    ok_p = srgb_to_oklab(pred_rgb[mask])
    ok_g = srgb_to_oklab(gt_rgb[mask])
    h_p = np.degrees(np.arctan2(ok_p[:, 2], ok_p[:, 1])) % 360.0
    h_g = np.degrees(np.arctan2(ok_g[:, 2], ok_g[:, 1])) % 360.0
    dh = (h_p - h_g + 180.0) % 360.0 - 180.0
    return float(np.mean(np.abs(dh)))


def _build_unbiased_multi_still_splits(
    natural_names: list[str],
    n_splits: int = 12,
    k_stills: int = 5,
    seed: int = 2026,
) -> list[tuple[list[str], list[str], str]]:
    """Draw seeded disjoint (5 src, 5 ref, 1 test) splits fairly across all natural scenes."""
    rng = np.random.default_rng(seed)
    test_order = list(rng.permutation(natural_names))[:n_splits]
    src_counts: Counter[str] = Counter({n: 0 for n in natural_names})
    ref_counts: Counter[str] = Counter({n: 0 for n in natural_names})
    splits: list[tuple[list[str], list[str], str]] = []
    for test_name in test_order:
        pool = [n for n in natural_names if n != test_name]
        rng.shuffle(pool)
        a_names = sorted(pool, key=lambda n: src_counts[n])[:k_stills]
        rem = [n for n in pool if n not in a_names]
        rng.shuffle(rem)
        b_names = sorted(rem, key=lambda n: ref_counts[n])[:k_stills]
        src_counts.update(a_names)
        ref_counts.update(b_names)
        splits.append((a_names, b_names, test_name))
    return splits


def run_benchmark(save_path: Path | None = None) -> dict[str, Any]:
    luts = build_cinematic_luts(size=33)
    scenes = build_test_scenes()
    id_lut = identity_lut(33)

    # Regime 3: Single-still cross-scene triples (src A, ref B, held-out C) & adversarial triples
    cross_triples = [
        ("portrait_fair_neutral", "foliage_nature", "portrait_medium_warm"),
        ("urban_neutrals", "portrait_deep_cool", "sky_architecture"),
        ("foliage_nature", "colorful_market", "autumn_park"),
        ("sky_architecture", "sunset_coast", "portrait_olive_studio"),
        ("portrait_medium_warm", "high_key_studio", "tungsten_cafe"),
        ("low_key_night", "urban_neutrals", "portrait_fair_neutral"),
    ]

    adv_triples = [
        ("portrait_fair_neutral", "adv_red_wall", "portrait_medium_warm"),
        ("portrait_olive_studio", "adv_cyan_fog", "portrait_deep_cool"),
        ("portrait_fair_neutral", "adv_green_wall", "portrait_olive_studio"),
    ]

    # Regime 2: Unbiased multi-still splits across all 13 natural scenes (5 src, 5 ref, 1 test)
    natural_names = [name for name, sc in scenes.items() if not sc.is_adversarial]
    multi_still_splits = _build_unbiased_multi_still_splits(
        natural_names, n_splits=12, k_stills=5, seed=2026
    )

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
            "lut_match": (solve_lut(src_arg, ref_arg, size=33, skin_protect=0.5), None),
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
        m: {
            "cross_de00": [],
            "cross_skin_de00": [],
            "cross_skin_hue_err": [],
            "adv_de00": [],
            "adv_skin_de00": [],
            "adv_skin_hue_err": [],
            "roughness": [],
            "inversion_rate": [],
            "direct_cross_de00": [],
            "direct_adv_skin_de00": [],
        }
        for m in methods
    }

    for lut_name, gt_lut in luts.items():
        gt_graded = {sname: apply_lut(sc.rgb, gt_lut) for sname, sc in scenes.items()}

        # Regime 1: Paired Look Recovery (src=A, ref=LUT_k(A) -> test C)
        for src_name, _, test_name in cross_triples:
            src_rgb = scenes[src_name].rgb
            ref_rgb = gt_graded[src_name]
            test_scene = scenes[test_name]
            gt_lab = rgb_to_lab(gt_graded[test_name])
            for m_name, (m_lut, _) in _fit_all(src_rgb, ref_rgb, src_rgb, ref_rgb).items():
                pred = apply_lut(test_scene.rgb, m_lut)
                de = ciede2000(rgb_to_lab(pred), gt_lab)
                acc_r1[m_name]["de00"].append(float(np.mean(de)))
                if np.any(test_scene.skin_mask):
                    acc_r1[m_name]["skin_de00"].append(float(np.mean(de[test_scene.skin_mask])))
                info = inspect_lut(m_lut)
                acc_r1[m_name]["roughness"].append(info["laplacian_roughness"])
                acc_r1[m_name]["inversion_rate"].append(info["inversion_rate"])

        # Regime 2: Unbiased Multi-Still Look Transfer (src=[A1..A5], refs=[LUT_k(B1..B5)] -> test C)
        for a_names, b_names, test_name in multi_still_splits:
            src_list = [scenes[a].rgb for a in a_names]
            ref_list = [gt_graded[b] for b in b_names]
            src_pooled = np.concatenate([s[::2, ::2] for s in src_list], axis=0)
            ref_pooled = np.concatenate([r[::2, ::2] for r in ref_list], axis=0)
            test_scene = scenes[test_name]
            gt_lab = rgb_to_lab(gt_graded[test_name])
            for m_name, (m_lut, _) in _fit_all(src_list, ref_list, src_pooled, ref_pooled).items():
                pred = apply_lut(test_scene.rgb, m_lut)
                de = ciede2000(rgb_to_lab(pred), gt_lab)
                acc_r2[m_name]["de00"].append(float(np.mean(de)))
                if np.any(test_scene.skin_mask):
                    acc_r2[m_name]["skin_de00"].append(float(np.mean(de[test_scene.skin_mask])))
                info = inspect_lut(m_lut)
                acc_r2[m_name]["roughness"].append(info["laplacian_roughness"])
                acc_r2[m_name]["inversion_rate"].append(info["inversion_rate"])

        # Regime 3: Single-Still Cross-Scene & Adversarial Dominant-Color Robustness
        for src_name, ref_name, test_name in cross_triples:
            src_rgb = scenes[src_name].rgb
            ref_rgb = gt_graded[ref_name]
            test_scene = scenes[test_name]
            gt_test = gt_graded[test_name]
            gt_lab = rgb_to_lab(gt_test)

            for m_name, (m_lut, direct_fn) in _fit_all(src_rgb, ref_rgb, src_rgb, ref_rgb).items():
                pred = apply_lut(test_scene.rgb, m_lut)
                de = ciede2000(rgb_to_lab(pred), gt_lab)
                acc_r3[m_name]["cross_de00"].append(float(np.mean(de)))
                if np.any(test_scene.skin_mask):
                    acc_r3[m_name]["cross_skin_de00"].append(float(np.mean(de[test_scene.skin_mask])))
                    acc_r3[m_name]["cross_skin_hue_err"].append(
                        _mean_hue_error_deg(pred, gt_test, test_scene.skin_mask)
                    )
                info = inspect_lut(m_lut)
                acc_r3[m_name]["roughness"].append(info["laplacian_roughness"])
                acc_r3[m_name]["inversion_rate"].append(info["inversion_rate"])

                if direct_fn is not None:
                    pred_d = direct_fn(test_scene.rgb)
                    de_d = ciede2000(rgb_to_lab(pred_d), gt_lab)
                    acc_r3[m_name]["direct_cross_de00"].append(float(np.mean(de_d)))
                else:
                    acc_r3[m_name]["direct_cross_de00"].append(float(np.mean(de)))

        for src_name, ref_name, test_name in adv_triples:
            src_rgb = scenes[src_name].rgb
            ref_rgb = gt_graded[ref_name]
            test_scene = scenes[test_name]
            gt_test = gt_graded[test_name]
            gt_lab = rgb_to_lab(gt_test)

            for m_name, (m_lut, direct_fn) in _fit_all(src_rgb, ref_rgb, src_rgb, ref_rgb).items():
                pred = apply_lut(test_scene.rgb, m_lut)
                de = ciede2000(rgb_to_lab(pred), gt_lab)
                acc_r3[m_name]["adv_de00"].append(float(np.mean(de)))
                acc_r3[m_name]["adv_skin_de00"].append(float(np.mean(de[test_scene.skin_mask])))
                acc_r3[m_name]["adv_skin_hue_err"].append(
                    _mean_hue_error_deg(pred, gt_test, test_scene.skin_mask)
                )
                if direct_fn is not None:
                    pred_d = direct_fn(test_scene.rgb)
                    de_d = ciede2000(rgb_to_lab(pred_d), gt_lab)
                    acc_r3[m_name]["direct_adv_skin_de00"].append(float(np.mean(de_d[test_scene.skin_mask])))
                else:
                    acc_r3[m_name]["direct_adv_skin_de00"].append(float(np.mean(de[test_scene.skin_mask])))

    def _summarize(acc: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float]]:
        return {m: {k: float(np.mean(v)) for k, v in metrics.items()} for m, metrics in acc.items()}

    summary_r1 = _summarize(acc_r1)
    summary_r2 = _summarize(acc_r2)
    summary = _summarize(acc_r3)

    best_base_cross = min(summary[m]["cross_de00"] for m in transfer_baselines)
    best_base_adv_skin = min(summary[m]["adv_skin_de00"] for m in transfer_baselines)
    best_base_rough = min(summary[m]["roughness"] for m in transfer_baselines)
    best_r1_base = min(summary_r1[m]["de00"] for m in methods if m != "lut_match")
    best_r2_transfer = min(summary_r2[m]["de00"] for m in transfer_baselines)

    report = {
        "n_luts": len(luts),
        "n_paired_trials": len(luts) * len(cross_triples),
        "n_multi_still_trials": len(luts) * len(multi_still_splits),
        "n_cross_trials": len(luts) * len(cross_triples),
        "n_adv_trials": len(luts) * len(adv_triples),
        "paired_summary": summary_r1,
        "multi_still_summary": summary_r2,
        "summary": summary,
        "gains_vs_best_baseline": {
            "paired_de00_reduction_pct": 100.0 * (best_r1_base - summary_r1["lut_match"]["de00"]) / best_r1_base,
            "paired_vs_identity_pct": 100.0 * (summary_r1["identity"]["de00"] - summary_r1["lut_match"]["de00"]) / summary_r1["identity"]["de00"],
            "multi_still_vs_identity_pct": 100.0 * (summary_r2["identity"]["de00"] - summary_r2["lut_match"]["de00"]) / summary_r2["identity"]["de00"],
            "multi_still_vs_best_transfer_pct": 100.0 * (best_r2_transfer - summary_r2["lut_match"]["de00"]) / best_r2_transfer,
            "cross_de00_reduction_pct": 100.0 * (best_base_cross - summary["lut_match"]["cross_de00"]) / best_base_cross,
            "adv_skin_de00_reduction_pct": 100.0 * (best_base_adv_skin - summary["lut_match"]["adv_skin_de00"]) / best_base_adv_skin,
            "roughness_reduction_pct": 100.0 * (best_base_rough - summary["lut_match"]["roughness"]) / best_base_rough,
            "lut_match_inversion_rate": summary["lut_match"]["inversion_rate"],
        },
    }

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return report


def _print_table(report: dict[str, Any]) -> None:
    print(f"\n1) Paired Look Recovery (src=A, ref=LUT_k(A) -> test C, {report['n_paired_trials']} trials)")
    h1 = f"{'Method':<15} | {'dE00':>8} | {'Skin dE00':>9} | {'Roughness':>9} | {'Inversion %':>11}"
    print(h1)
    print("-" * len(h1))
    for m_name, row in report["paired_summary"].items():
        print(
            f"{m_name:<15} | {row['de00']:>8.3f} | {row['skin_de00']:>9.3f} | "
            f"{row['roughness']:>9.5f} | {row['inversion_rate'] * 100.0:>10.4f}%"
        )

    print(f"\n2) Unbiased Multi-Still Look Transfer (src=[A1..A5], refs=[LUT_k(B1..B5)] -> test C, {report['n_multi_still_trials']} trials)")
    print(h1)
    print("-" * len(h1))
    for m_name, row in report["multi_still_summary"].items():
        print(
            f"{m_name:<15} | {row['de00']:>8.3f} | {row['skin_de00']:>9.3f} | "
            f"{row['roughness']:>9.5f} | {row['inversion_rate'] * 100.0:>10.4f}%"
        )

    print(
        f"\n3) Single-Still Cross-Scene & Adversarial Robustness ({report['n_luts']} LUTs, "
        f"{report['n_cross_trials']} cross trials, {report['n_adv_trials']} adversarial trials)"
    )
    header = (
        f"{'Method':<15} | {'Cross dE00':>10} | {'Skin dE00':>9} | "
        f"{'Adv Skin dE00':>13} | {'Adv Hue Err':>11} | {'Roughness':>9} | {'Inversion %':>11}"
    )
    print(header)
    print("-" * len(header))
    for m_name, row in report["summary"].items():
        print(
            f"{m_name:<15} | {row['cross_de00']:>10.3f} | {row['cross_skin_de00']:>9.3f} | "
            f"{row['adv_skin_de00']:>13.3f} | {row['adv_skin_hue_err']:>10.2f}° | "
            f"{row['roughness']:>9.5f} | {row['inversion_rate'] * 100.0:>10.4f}%"
        )
    gains = report["gains_vs_best_baseline"]
    print("\nSummary Gains:")
    print(f"  Paired dE00 reduction vs best baseline:  {gains['paired_de00_reduction_pct']:.1f}% (vs identity: {gains['paired_vs_identity_pct']:.1f}%)")
    print(f"  Multi-still dE00 vs identity / transfer: {gains['multi_still_vs_identity_pct']:.1f}% / {gains['multi_still_vs_best_transfer_pct']:.1f}%")
    print(f"  Single-still cross dE00 vs transfer:     {gains['cross_de00_reduction_pct']:.1f}%")
    print(f"  Adversarial skin dE00 vs transfer:       {gains['adv_skin_de00_reduction_pct']:.1f}%")
    print(f"  Lattice roughness vs transfer:           {gains['roughness_reduction_pct']:.1f}%")
    print(f"  lut_match inversion rate:                {gains['lut_match_inversion_rate'] * 100.0:.4f}%")


if __name__ == "__main__":
    out_file = Path(__file__).resolve().parent / "results.json"
    res = run_benchmark(save_path=out_file)
    _print_table(res)
