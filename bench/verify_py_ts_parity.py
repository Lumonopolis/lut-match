from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.scenes import build_test_scenes
from lut_match.io import inspect_lut
from lut_match.solver import solve_lut


def _to_rgba_u8(rgb: np.ndarray) -> np.ndarray:
    h, w, _ = rgb.shape
    u8 = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = u8
    rgba[..., 3] = 255
    return rgba


def main() -> int:
    scenes = build_test_scenes()

    src_portrait = scenes["portrait_fair_neutral"].rgb
    ref_teal = np.clip(scenes["sunset_coast"].rgb * np.array([0.88, 1.02, 1.12]), 0.0, 1.0)
    ref_warm = np.clip(scenes["autumn_park"].rgb * np.array([1.08, 1.00, 0.88]), 0.0, 1.0)
    ref_adv_neon = scenes["adv_red_wall"].rgb

    # Synthetic S-Log3-encoded source image (smooth log ramps + skin patch)
    h, w = src_portrait.shape[:2]
    slog3_src = np.clip(0.32 + 0.36 * src_portrait, 0.0, 1.0)

    cases = [
        {
            "name": "paired srgb",
            "sources": [src_portrait],
            "references": [np.clip(0.92 * src_portrait * np.array([1.06, 1.0, 0.90]) + 0.03, 0.0, 1.0)],
            "idt": "srgb",
            "skin_protect": 0.5,
            "shadow_lift": 0.0,
            "highlight_rolloff": 0.0,
            "smoothness": 0.35,
            "strength": 1.0,
        },
        {
            "name": "normal srgb",
            "sources": [src_portrait],
            "references": [ref_teal],
            "idt": "srgb",
            "mode": "balanced",
            "skin_protect": 0.5,
            "shadow_lift": 0.0,
            "highlight_rolloff": 0.0,
            "smoothness": 0.35,
            "strength": 1.0,
        },
        {
            "name": "creative srgb",
            "sources": [src_portrait],
            "references": [ref_teal],
            "idt": "srgb",
            "mode": "creative",
            "skin_protect": 0.5,
            "shadow_lift": 0.0,
            "highlight_rolloff": 0.0,
            "smoothness": 0.35,
            "strength": 1.0,
        },
        {
            "name": "multi-ref",
            "sources": [src_portrait],
            "references": [ref_teal, ref_warm],
            "idt": "srgb",
            "skin_protect": 0.35,
            "shadow_lift": 0.04,
            "highlight_rolloff": 0.15,
            "smoothness": 0.35,
            "strength": 0.9,
        },
        {
            "name": "multi-src+multi-ref",
            "sources": [src_portrait, scenes["urban_neutrals"].rgb],
            "references": [ref_teal, ref_warm],
            "idt": "srgb",
            "skin_protect": 0.5,
            "shadow_lift": 0.02,
            "highlight_rolloff": 0.10,
            "smoothness": 0.35,
            "strength": 1.0,
        },
        {
            "name": "slog3 IDT",
            "sources": [slog3_src],
            "references": [ref_teal],
            "idt": "slog3",
            "skin_protect": 0.5,
            "shadow_lift": 0.05,
            "highlight_rolloff": 0.2,
            "smoothness": 0.35,
            "strength": 1.0,
        },
        {
            "name": "slog3 strength<1",
            "sources": [slog3_src],
            "references": [ref_teal],
            "idt": "slog3",
            "skin_protect": 0.5,
            "shadow_lift": 0.05,
            "highlight_rolloff": 0.2,
            "smoothness": 0.35,
            "strength": 0.75,
        },
        {
            "name": "rec709 IDT",
            "sources": [src_portrait],
            "references": [ref_teal],
            "idt": "rec709",
            "skin_protect": 0.5,
            "shadow_lift": 0.0,
            "highlight_rolloff": 0.0,
            "smoothness": 0.35,
            "strength": 1.0,
        },
        {
            "name": "apple-log IDT",
            "sources": [slog3_src],
            "references": [ref_teal],
            "idt": "apple-log",
            "skin_protect": 0.5,
            "shadow_lift": 0.0,
            "highlight_rolloff": 0.0,
            "smoothness": 0.35,
            "strength": 1.0,
        },
        {
            "name": "vlog IDT",
            "sources": [slog3_src],
            "references": [ref_teal],
            "idt": "vlog",
            "skin_protect": 0.5,
            "shadow_lift": 0.0,
            "highlight_rolloff": 0.0,
            "smoothness": 0.35,
            "strength": 1.0,
        },
        {
            "name": "clog3 IDT",
            "sources": [slog3_src],
            "references": [ref_teal],
            "idt": "clog3",
            "skin_protect": 0.5,
            "shadow_lift": 0.0,
            "highlight_rolloff": 0.0,
            "smoothness": 0.35,
            "strength": 1.0,
        },
        {
            "name": "adversarial neon-red",
            "sources": [src_portrait],
            "references": [ref_adv_neon],
            "idt": "srgb",
            "skin_protect": 0.5,
            "shadow_lift": 0.0,
            "highlight_rolloff": 0.0,
            "smoothness": 0.35,
            "strength": 1.0,
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        manifest = []
        py_luts = []
        py_inversions = []

        for idx, case in enumerate(cases):
            src_bins = []
            src_exact_list = []
            for s_i, src_arr in enumerate(case["sources"]):
                src_rgba = _to_rgba_u8(src_arr)
                src_exact_list.append(src_rgba[..., :3].astype(np.float64) / 255.0)
                s_bin = tmp / f"case_{idx}_src_{s_i}.bin"
                s_bin.write_bytes(src_rgba.tobytes())
                src_bins.append(
                    {
                        "path": str(s_bin),
                        "width": int(src_rgba.shape[1]),
                        "height": int(src_rgba.shape[0]),
                    }
                )

            ref_bins = []
            ref_exact_list = []
            for r_i, ref_arr in enumerate(case["references"]):
                ref_rgba = _to_rgba_u8(ref_arr)
                ref_exact_list.append(ref_rgba[..., :3].astype(np.float64) / 255.0)
                r_bin = tmp / f"case_{idx}_ref_{r_i}.bin"
                r_bin.write_bytes(ref_rgba.tobytes())
                ref_bins.append(
                    {
                        "path": str(r_bin),
                        "width": int(ref_rgba.shape[1]),
                        "height": int(ref_rgba.shape[0]),
                    }
                )

            src_arg = src_exact_list[0] if len(src_exact_list) == 1 else src_exact_list
            mode = case.get("mode", "balanced")
            py_lut = solve_lut(
                src_rgb=src_arg,
                ref_rgbs=ref_exact_list,
                size=33,
                strength=case["strength"],
                skin_protect=case["skin_protect"],
                shadow_lift=case["shadow_lift"],
                highlight_rolloff=case["highlight_rolloff"],
                smoothness=case["smoothness"],
                idt=case["idt"],
                mode=mode,
            )
            py_info = inspect_lut(py_lut.astype(np.float32).astype(np.float64))
            py_luts.append(np.transpose(py_lut, (2, 1, 0, 3)).reshape(-1))
            py_inversions.append(py_info)

            out_bin = tmp / f"case_{idx}_ts.bin"
            manifest.append(
                {
                    "name": case["name"],
                    "sources": src_bins,
                    "references": ref_bins,
                    "idt": case["idt"],
                    "mode": mode,
                    "skinProtect": case["skin_protect"],
                    "shadowLift": case["shadow_lift"],
                    "highlightRolloff": case["highlight_rolloff"],
                    "smoothness": case["smoothness"],
                    "strength": case["strength"],
                    "outPath": str(out_bin),
                }
            )

        manifest_path = tmp / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        core_entry = (ROOT / "packages" / "lut-match-core" / "dist" / "index.js").resolve()
        node_script = tmp / "run_ts.mjs"
        node_script.write_text(
            f"""
import fs from 'node:fs';
import {{ fitLut, inspectLut }} from {json.dumps(core_entry.as_uri())};

const manifest = JSON.parse(fs.readFileSync({json.dumps(str(manifest_path))}, 'utf8'));
const loadImg = (spec) => ({{
  width: spec.width,
  height: spec.height,
  data: new Uint8ClampedArray(fs.readFileSync(spec.path)),
}});

// Warm up V8 JIT across all cases (3 passes so TurboFan compiles all loops)
for (let w = 0; w < 3; w++) {{
  for (const c of manifest) {{
    const srcImgs = c.sources.map(loadImg);
    fitLut({{
      source: srcImgs.length === 1 ? srcImgs[0] : srcImgs,
      references: c.references.map(loadImg),
      size: 33,
      idt: c.idt,
      mode: c.mode,
      skinProtect: c.skinProtect,
      shadowLift: c.shadowLift,
      highlightRolloff: c.highlightRolloff,
      smoothness: c.smoothness,
      strength: c.strength,
    }});
  }}
}}

const results = [];
for (const c of manifest) {{
  const srcImgs = c.sources.map(loadImg);
  const source = srcImgs.length === 1 ? srcImgs[0] : srcImgs;
  const references = c.references.map(loadImg);
  const times = [];
  let lut = null;
  for (let r = 0; r < 5; r++) {{
    const t0 = performance.now();
    lut = fitLut({{
      source,
      references,
      size: 33,
      idt: c.idt,
      mode: c.mode,
      skinProtect: c.skinProtect,
      shadowLift: c.shadowLift,
      highlightRolloff: c.highlightRolloff,
      smoothness: c.smoothness,
      strength: c.strength,
    }});
    times.push(performance.now() - t0);
  }}
  const solveMs = Math.min(...times);
  const stats = inspectLut(lut);
  fs.writeFileSync(c.outPath, Buffer.from(lut.data.buffer, lut.data.byteOffset, lut.data.byteLength));
  results.push({{
    name: c.name,
    solveMs,
    inversionCount: stats.inversionCount,
    inversionRate: stats.inversionRate,
    laplacianRoughness: stats.laplacianRoughness,
    maxStep: stats.maxStep,
    clippingFraction: stats.clippingFraction,
    entries: lut.data.length,
  }});
}}
console.log(JSON.stringify(results));
""",
            encoding="utf-8",
        )

        proc = subprocess.run(
            ["node", str(node_script)],
            check=True,
            capture_output=True,
            text=True,
        )
        ts_results = json.loads(proc.stdout.strip().splitlines()[-1])

        print(f"{'Case':<22} | {'Max Abs Diff':>12} | {'Mean Abs Diff':>13} | {'Insp Diff':>10} | {'Py Inv':>6} | {'TS Inv':>6} | {'TS ms':>8}")
        print("-" * 92)

        all_ok = True
        for idx, ts_res in enumerate(ts_results):
            out_bin = Path(manifest[idx]["outPath"])
            ts_arr = np.frombuffer(out_bin.read_bytes(), dtype=np.float32).astype(np.float64)
            py_arr = py_luts[idx]
            assert ts_arr.shape == py_arr.shape == (33 * 33 * 33 * 3,)

            abs_diff = np.abs(ts_arr - py_arr)
            max_diff = float(np.max(abs_diff))
            mean_diff = float(np.mean(abs_diff))
            py_info = py_inversions[idx]
            py_inv_cnt = int(py_info["inversion_count"])
            ts_inv_cnt = int(ts_res["inversionCount"])
            insp_diff = max(
                abs(py_info["laplacian_roughness"] - float(ts_res["laplacianRoughness"])),
                abs(py_info["max_step"] - float(ts_res["maxStep"])),
                abs(py_info["clipping_fraction"] - float(ts_res["clippingFraction"])),
                abs(py_info["inversion_rate"] - float(ts_res["inversionRate"])),
            )
            solve_ms = float(ts_res["solveMs"])

            print(
                f"{ts_res['name']:<22} | {max_diff:>12.2e} | {mean_diff:>13.2e} | "
                f"{insp_diff:>10.2e} | {py_inv_cnt:>6d} | {ts_inv_cnt:>6d} | {solve_ms:>6.2f}ms"
            )

            if max_diff >= 1e-3:
                print(f"FAIL: {ts_res['name']} max_abs_diff={max_diff} >= 1e-3", file=sys.stderr)
                all_ok = False
            if insp_diff >= 1e-4:
                print(f"FAIL: {ts_res['name']} inspect_lut parity diff={insp_diff} >= 1e-4", file=sys.stderr)
                all_ok = False
            if manifest[idx]["idt"] in ("srgb", "rec709") and (py_inv_cnt != 0 or ts_inv_cnt != 0):
                print(f"FAIL: {ts_res['name']} non-zero inversions py={py_inv_cnt} ts={ts_inv_cnt}", file=sys.stderr)
                all_ok = False
            if solve_ms >= 150.0:
                print(f"FAIL: {ts_res['name']} TS solve_ms={solve_ms:.2f} >= 150", file=sys.stderr)
                all_ok = False

        return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
