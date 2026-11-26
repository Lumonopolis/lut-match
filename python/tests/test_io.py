from pathlib import Path

import numpy as np

from lut_match.io import (
    apply_lut,
    identity_lut,
    inspect_lut,
    load_cube,
    load_hald_clut,
    save_cube,
    save_hald_clut,
)


def test_cube_roundtrip_and_r_fastest_order(tmp_path: Path):
    # Asymmetric non-identity LUT to catch any axis-transposition bug
    lut = identity_lut(17).copy()
    lut[16, 0, 0] = [0.91, 0.12, 0.03]  # pure R=1, G=0, B=0 -> should be index 16 in .cube
    lut[0, 16, 0] = [0.04, 0.88, 0.11]  # R=0, G=1, B=0 -> index 16*17
    lut[0, 0, 16] = [0.05, 0.15, 0.94]  # R=0, G=0, B=1 -> index 16*17*17

    cube_path = tmp_path / "test_order.cube"
    save_cube(lut, cube_path, title="OrderCheck")

    data_lines = [
        ln.strip()
        for ln in cube_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith(("TITLE", "LUT_3D_SIZE", "DOMAIN_", "#"))
    ]
    assert len(data_lines) == 17 ** 3
    assert [float(x) for x in data_lines[16].split()] == [0.91, 0.12, 0.03]
    assert [float(x) for x in data_lines[16 * 17].split()] == [0.04, 0.88, 0.11]
    assert [float(x) for x in data_lines[16 * 17 * 17].split()] == [0.05, 0.15, 0.94]

    loaded, meta = load_cube(cube_path)
    assert meta["title"] == "OrderCheck"
    assert meta["size"] == 17
    assert np.allclose(loaded, lut, atol=1e-6)


def test_hald_clut_roundtrip(tmp_path: Path):
    ident = identity_lut(33)
    # Smooth nonlinear color grade
    lut = np.clip(0.85 * ident + 0.05, 0.0, 1.0)
    hald_path = tmp_path / "hald.png"
    save_hald_clut(lut, hald_path, level=8)
    loaded = load_hald_clut(hald_path, target_size=33)
    assert loaded.shape == (33, 33, 33, 3)
    assert np.max(np.abs(loaded - lut)) < 1.0 / 255.0 + 1e-3


def test_tetrahedral_interpolation_exactness():
    ident = identity_lut(17)
    rng = np.random.default_rng(123)
    pts = rng.uniform(0.0, 1.0, size=(500, 3))
    out_ident = apply_lut(pts, ident)
    assert np.allclose(out_ident, pts, atol=1e-12)

    # Exact recovery on lattice nodes for an arbitrary non-linear LUT
    custom = np.clip(ident ** 1.3 * 0.9 + 0.03, 0.0, 1.0)
    out_nodes = apply_lut(ident, custom)
    assert np.allclose(out_nodes, custom, atol=1e-12)


def test_inspect_lut_detects_inversions_and_roughness():
    ident = identity_lut(17)
    m_id = inspect_lut(ident)
    assert m_id["inversion_count"] == 0
    assert m_id["inversion_rate"] == 0.0
    assert m_id["laplacian_roughness"] < 1e-12
    assert m_id["skin_hue_shift_deg"] == 0.0

    broken = ident.copy()
    broken[5:8, 5, 5, 0] = [0.9, 0.2, 0.1]
    m_br = inspect_lut(broken)
    assert m_br["inversion_count"] > 0
    assert m_br["inversion_rate"] > 0.0
    assert m_br["laplacian_roughness"] > 0.01


def test_save_cube_sanitizes_title(tmp_path: Path):
    ident = identity_lut(17)
    cube_path = tmp_path / "sanitized.cube"
    save_cube(ident, cube_path, title='Bad "Title"\r\nInjected')
    lines = cube_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == 'TITLE "Bad  Title   Injected"'
    assert lines[1] == "LUT_3D_SIZE 17"
