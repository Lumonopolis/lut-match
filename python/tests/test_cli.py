import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


def _write_png(path: Path, rgb: np.ndarray) -> None:
    u8 = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(u8, mode="RGB").save(path)


def test_cli_fit_apply_inspect_and_error_path(tmp_path: Path):
    rng = np.random.default_rng(77)
    src = rng.uniform(0.1, 0.9, size=(48, 48, 3))
    ref = np.clip(src * np.array([1.05, 0.96, 0.88]), 0.0, 1.0)

    src_p = tmp_path / "src.png"
    ref_p = tmp_path / "ref.png"
    cube_p = tmp_path / "look.cube"
    hald_p = tmp_path / "hald.png"
    out_p = tmp_path / "graded.png"
    _write_png(src_p, src)
    _write_png(ref_p, ref)

    fit_res = subprocess.run(
        [
            sys.executable,
            "-m",
            "lut_match.cli",
            "fit",
            "--ref",
            str(ref_p),
            "--src",
            str(src_p),
            "-o",
            str(cube_p),
            "--size",
            "17",
            "--strength",
            "0.9",
            "--skin-protect",
            "0.6",
            "--idt",
            "srgb",
            "--hald",
            str(hald_p),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert fit_res.returncode == 0
    assert cube_p.exists()
    assert hald_p.exists()
    with Image.open(hald_p) as im:
        assert im.size == (512, 512)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "lut_match.cli",
            "apply",
            "--lut",
            str(cube_p),
            "--in",
            str(src_p),
            "--out",
            str(out_p),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out_p.exists()

    insp_res = subprocess.run(
        [
            sys.executable,
            "-m",
            "lut_match.cli",
            "inspect",
            "--lut",
            str(cube_p),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(insp_res.stdout)
    assert payload["size"] == 17
    assert payload["inversion_rate"] == 0.0

    # Creative mode CLI flag
    cube_cre = tmp_path / "look_creative.cube"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "lut_match.cli",
            "fit",
            "--ref",
            str(ref_p),
            "--src",
            str(src_p),
            "-o",
            str(cube_cre),
            "--size",
            "17",
            "--mode",
            "creative",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert cube_cre.exists()

    # Error path: missing file must fail with non-zero code
    err_res = subprocess.run(
        [
            sys.executable,
            "-m",
            "lut_match.cli",
            "inspect",
            "--lut",
            str(tmp_path / "does_not_exist.cube"),
        ],
        capture_output=True,
        text=True,
    )
    assert err_res.returncode != 0
    assert "not found" in err_res.stderr.lower()
