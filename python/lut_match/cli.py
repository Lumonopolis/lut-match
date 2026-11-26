from __future__ import annotations

import argparse
import json
import sys

from lut_match.color import SUPPORTED_IDTS
from lut_match.io import (
    apply_lut,
    inspect_lut,
    load_cube,
    load_image,
    save_cube,
    save_hald_clut,
    save_image,
)
from lut_match.solver import solve_lut


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lut-match",
        description="Fit, apply, and inspect smooth monotone .cube 3D color-grade LUTs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fit = sub.add_parser("fit", help="Fit a .cube 3D LUT from source and reference still(s).")
    p_fit.add_argument("--ref", action="append", required=True, help="Reference graded still path (repeatable).")
    p_fit.add_argument("--src", action="append", required=True, help="Source still path (repeatable).")
    p_fit.add_argument("-o", "--out", required=True, help="Output .cube file path.")
    p_fit.add_argument("--size", type=int, default=33, choices=(17, 33, 65), help="3D LUT grid size.")
    p_fit.add_argument("--strength", type=float, default=1.0, help="LUT strength in [0, 1].")
    p_fit.add_argument("--skin-protect", type=float, default=0.5, help="Skin-tone protection in [0, 1].")
    p_fit.add_argument("--shadow-lift", type=float, default=0.0, help="Shadow lift adjustment.")
    p_fit.add_argument("--highlight-rolloff", type=float, default=0.0, help="Highlight rolloff adjustment.")
    p_fit.add_argument("--smoothness", type=float, default=0.35, help="3D lattice smoothing factor.")
    p_fit.add_argument(
        "--mode",
        default="balanced",
        choices=("balanced", "creative"),
        help="Transfer mode: 'balanced' (Empirical-Bayes) or 'creative' (expressive single-still).",
    )
    p_fit.add_argument("--idt", default="srgb", choices=SUPPORTED_IDTS, help="Source input device transform.")
    p_fit.add_argument("--hald", default=None, help="Optional Level-8 HaldCLUT PNG output path.")
    p_fit.add_argument("--title", default="lut-match", help="TITLE metadata written to .cube.")

    p_apply = sub.add_parser("apply", help="Apply a .cube 3D LUT to an image via tetrahedral interpolation.")
    p_apply.add_argument("--lut", required=True, help="Input .cube LUT path.")
    p_apply.add_argument("--in", dest="inp", required=True, help="Input image path.")
    p_apply.add_argument("--out", required=True, help="Output graded image path.")

    p_inspect = sub.add_parser("inspect", help="Inspect structural quality metrics of a .cube LUT.")
    p_inspect.add_argument("--lut", required=True, help="Input .cube LUT path.")
    p_inspect.add_argument("--json", action="store_true", help="Emit JSON output.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "fit":
            src_imgs = [load_image(p) for p in args.src]
            src_arg = src_imgs[0] if len(src_imgs) == 1 else src_imgs
            ref_imgs = [load_image(p) for p in args.ref]
            lut = solve_lut(
                src_rgb=src_arg,
                ref_rgbs=ref_imgs,
                size=args.size,
                strength=args.strength,
                skin_protect=args.skin_protect,
                shadow_lift=args.shadow_lift,
                highlight_rolloff=args.highlight_rolloff,
                smoothness=args.smoothness,
                idt=args.idt,
                mode=args.mode,
            )
            save_cube(lut, args.out, title=args.title)
            if args.hald:
                save_hald_clut(lut, args.hald)
            return 0

        if args.command == "apply":
            lut, _ = load_cube(args.lut)
            img = load_image(args.inp)
            graded = apply_lut(img, lut)
            save_image(graded, args.out)
            return 0

        if args.command == "inspect":
            lut, meta = load_cube(args.lut)
            metrics = inspect_lut(lut)
            payload = {"size": meta["size"], "title": meta["title"], **metrics}
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print(f"Title: {payload['title']} (size {payload['size']}^3)")
                print(f"Inversions:          {payload['inversion_count']} ({payload['inversion_rate'] * 100.0:.4f}%)")
                print(f"Laplacian roughness: {payload['laplacian_roughness']:.6f}")
                print(f"Max step:            {payload['max_step']:.6f}")
                print(f"Clipping fraction:   {payload['clipping_fraction'] * 100.0:.2f}%")
                print(f"Skin hue shift:      {payload['skin_hue_shift_deg']:.3f} deg")
            return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
