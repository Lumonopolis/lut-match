import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.baselines import fit_hist_match, fit_mkl_rgb, fit_reinhard_lab
from bench.cinematic_luts import build_cinematic_luts
from bench.scenes import build_test_scenes
from lut_match.color import ciede2000, rgb_to_lab
from lut_match.io import apply_lut, inspect_lut
from lut_match.solver import solve_lut


def test_solver_beats_baselines_live_cross_and_adversarial():
    luts = build_cinematic_luts(size=33)
    scenes = build_test_scenes()

    # Run a representative live cross-scene + adversarial verification
    for lut_name in ("teal_and_orange", "kodak_2383_print", "vintage_faded_film"):
        gt_lut = luts[lut_name]
        src = scenes["portrait_fair_neutral"].rgb
        ref_cross = apply_lut(scenes["foliage_nature"].rgb, gt_lut)
        ref_adv = apply_lut(scenes["adv_red_wall"].rgb, gt_lut)
        test_sc = scenes["portrait_medium_warm"]
        gt_test_lab = rgb_to_lab(apply_lut(test_sc.rgb, gt_lut))

        our_cross = solve_lut(src, ref_cross, size=33, skin_protect=0.5)
        our_adv = solve_lut(src, ref_adv, size=33, skin_protect=0.5)
        assert inspect_lut(our_cross)["inversion_rate"] == 0.0
        assert inspect_lut(our_adv)["inversion_rate"] == 0.0

        our_cross_de = float(np.mean(ciede2000(rgb_to_lab(apply_lut(test_sc.rgb, our_cross)), gt_test_lab)))
        our_adv_skin = float(
            np.mean(ciede2000(rgb_to_lab(apply_lut(test_sc.rgb, our_adv)), gt_test_lab)[test_sc.skin_mask])
        )

        for baseline_fn in (fit_reinhard_lab, fit_mkl_rgb, fit_hist_match):
            b_cross, _ = baseline_fn(src, ref_cross, size=33)
            b_adv, _ = baseline_fn(src, ref_adv, size=33)
            b_cross_de = float(np.mean(ciede2000(rgb_to_lab(apply_lut(test_sc.rgb, b_cross)), gt_test_lab)))
            b_adv_skin = float(
                np.mean(ciede2000(rgb_to_lab(apply_lut(test_sc.rgb, b_adv)), gt_test_lab)[test_sc.skin_mask])
            )
            assert our_cross_de < b_cross_de * 0.85, f"{lut_name}: {our_cross_de} vs {b_cross_de}"
            assert our_adv_skin < b_adv_skin * 0.70, f"{lut_name}: {our_adv_skin} vs {b_adv_skin}"


def test_saved_full_benchmark_results_meet_targets():
    results_path = ROOT / "bench" / "results.json"
    assert results_path.exists()
    data = json.loads(results_path.read_text(encoding="utf-8"))
    assert data["n_luts"] == 15
    assert data["n_paired_trials"] == 90
    assert data["n_multi_still_trials"] == 180
    assert data["n_cross_trials"] == 90
    assert data["n_adv_trials"] == 45
    assert "identity" in data["paired_summary"]
    assert "identity" in data["multi_still_summary"]
    assert "identity" in data["summary"]
    gains = data["gains_vs_best_baseline"]
    assert gains["paired_de00_reduction_pct"] > 0.0
    assert gains["paired_vs_identity_pct"] >= 70.0
    assert gains["multi_still_vs_identity_pct"] >= 15.0
    assert gains["multi_still_vs_best_transfer_pct"] >= 40.0
    assert gains["cross_de00_reduction_pct"] >= 50.0
    assert gains["adv_skin_de00_reduction_pct"] >= 50.0
    assert gains["roughness_reduction_pct"] >= 60.0
    assert data["paired_summary"]["lut_match"]["inversion_rate"] == 0.0
    assert data["multi_still_summary"]["lut_match"]["inversion_rate"] == 0.0
    assert gains["lut_match_inversion_rate"] == 0.0
