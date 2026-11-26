from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image

from lut_match.color import srgb_to_oklab


def identity_lut(size: int = 33) -> np.ndarray:
    """Create an identity 3D LUT of shape (size, size, size, 3) with [R, G, B] axes."""
    if size < 2:
        raise ValueError(f"LUT size must be >= 2, got {size}")
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([r, g, b], axis=-1)


def load_cube(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Parse an Adobe .cube 3D LUT (R-fastest, B-slowest) into (N, N, N, 3) [R, G, B]."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"LUT file not found: {p}")

    title = ""
    size: int | None = None
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    entries: list[list[float]] = []

    with p.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            key = parts[0].upper()
            if key == "TITLE":
                title = line[len("TITLE") :].strip().strip('"')
            elif key == "LUT_3D_SIZE":
                if len(parts) != 2:
                    raise ValueError(f"Invalid LUT_3D_SIZE line: {raw_line.strip()}")
                size = int(parts[1])
            elif key == "LUT_1D_SIZE":
                raise ValueError("1D .cube LUTs are not supported; expected LUT_3D_SIZE")
            elif key == "DOMAIN_MIN":
                if len(parts) != 4:
                    raise ValueError(f"Invalid DOMAIN_MIN line: {raw_line.strip()}")
                domain_min = np.array([float(v) for v in parts[1:4]], dtype=np.float64)
            elif key == "DOMAIN_MAX":
                if len(parts) != 4:
                    raise ValueError(f"Invalid DOMAIN_MAX line: {raw_line.strip()}")
                domain_max = np.array([float(v) for v in parts[1:4]], dtype=np.float64)
            elif key == "LUT_3D_INPUT_RANGE":
                continue
            else:
                if len(parts) != 3:
                    raise ValueError(f"Invalid LUT data entry: {raw_line.strip()}")
                entries.append([float(parts[0]), float(parts[1]), float(parts[2])])

    if size is None or size < 2:
        raise ValueError("Missing or invalid LUT_3D_SIZE header")
    expected = size ** 3
    if len(entries) != expected:
        raise ValueError(f"Expected {expected} entries for size {size}, found {len(entries)}")

    # Spec 1.0: R varies fastest, then G, then B -> reshape (B, G, R, 3) then transpose to (R, G, B, 3)
    arr = np.asarray(entries, dtype=np.float64).reshape((size, size, size, 3))
    lut = np.transpose(arr, (2, 1, 0, 3)).copy()
    meta = {
        "title": title,
        "size": size,
        "domain_min": domain_min.tolist(),
        "domain_max": domain_max.tolist(),
    }
    return lut, meta


def save_cube(
    lut: np.ndarray,
    path: str | Path,
    title: str = "lut-match",
    domain_min: tuple[float, float, float] = (0.0, 0.0, 0.0),
    domain_max: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    """Serialize a (N, N, N, 3) [R, G, B] LUT in Adobe .cube Spec 1.0 format (R-fastest)."""
    arr = np.asarray(lut, dtype=np.float64)
    if arr.ndim != 4 or arr.shape[0] != arr.shape[1] or arr.shape[1] != arr.shape[2] or arr.shape[3] != 3:
        raise ValueError(f"Expected LUT shape (N, N, N, 3), got {arr.shape}")
    size = arr.shape[0]

    # Transpose (R, G, B, 3) -> (B, G, R, 3) so R varies fastest in row-major C order
    ordered = np.transpose(arr, (2, 1, 0, 3)).reshape((-1, 3))

    safe_title = re.sub(r'[\r\n"]', " ", title or "lut-match").strip()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        fh.write(f'TITLE "{safe_title}"\n')
        fh.write(f"LUT_3D_SIZE {size}\n")
        fh.write(f"DOMAIN_MIN {domain_min[0]:.6f} {domain_min[1]:.6f} {domain_min[2]:.6f}\n")
        fh.write(f"DOMAIN_MAX {domain_max[0]:.6f} {domain_max[1]:.6f} {domain_max[2]:.6f}\n")
        for r, g, b in ordered:
            fh.write(f"{r:.6f} {g:.6f} {b:.6f}\n")


def apply_lut(image_rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a 3D LUT (N, N, N, 3) to an RGB array using exact tetrahedral interpolation."""
    img = np.asarray(image_rgb, dtype=np.float64)
    table = np.asarray(lut, dtype=np.float64)
    orig_shape = img.shape
    if orig_shape[-1] != 3:
        raise ValueError(f"Expected last dimension 3, got {orig_shape}")

    n = table.shape[0]
    flat = np.clip(img.reshape((-1, 3)), 0.0, 1.0)
    scaled = flat * (n - 1)

    i0 = np.minimum(np.floor(scaled).astype(np.intp), n - 2)
    f = scaled - i0
    r0, g0, b0 = i0[:, 0], i0[:, 1], i0[:, 2]
    r1, g1, b1 = r0 + 1, g0 + 1, b0 + 1
    fr, fg, fb = f[:, 0:1], f[:, 1:2], f[:, 2:3]

    c000 = table[r0, g0, b0]
    c111 = table[r1, g1, b1]

    # 6 tetrahedra partitioning the unit cube along main diagonal (0,0,0)->(1,1,1)
    cond_rg = fr[:, 0] >= fg[:, 0]
    cond_gb = fg[:, 0] >= fb[:, 0]
    cond_br = fb[:, 0] >= fr[:, 0]

    out = np.empty_like(flat)

    m1 = cond_rg & cond_gb  # fr >= fg >= fb
    if np.any(m1):
        out[m1] = (
            (1.0 - fr[m1]) * c000[m1]
            + (fr[m1] - fg[m1]) * table[r1[m1], g0[m1], b0[m1]]
            + (fg[m1] - fb[m1]) * table[r1[m1], g1[m1], b0[m1]]
            + fb[m1] * c111[m1]
        )

    m2 = cond_rg & (~cond_gb) & (~cond_br)  # fr >= fb > fg
    if np.any(m2):
        out[m2] = (
            (1.0 - fr[m2]) * c000[m2]
            + (fr[m2] - fb[m2]) * table[r1[m2], g0[m2], b0[m2]]
            + (fb[m2] - fg[m2]) * table[r1[m2], g0[m2], b1[m2]]
            + fg[m2] * c111[m2]
        )

    m3 = cond_rg & cond_br  # fb >= fr >= fg
    if np.any(m3):
        out[m3] = (
            (1.0 - fb[m3]) * c000[m3]
            + (fb[m3] - fr[m3]) * table[r0[m3], g0[m3], b1[m3]]
            + (fr[m3] - fg[m3]) * table[r1[m3], g0[m3], b1[m3]]
            + fg[m3] * c111[m3]
        )

    m4 = (~cond_rg) & cond_br  # fg > fr, fb >= fr -> fb >= fg > fr or fg > fb >= fr
    m4_a = (~cond_rg) & (~cond_gb) & cond_br  # fb >= fg > fr
    if np.any(m4_a):
        out[m4_a] = (
            (1.0 - fb[m4_a]) * c000[m4_a]
            + (fb[m4_a] - fg[m4_a]) * table[r0[m4_a], g0[m4_a], b1[m4_a]]
            + (fg[m4_a] - fr[m4_a]) * table[r0[m4_a], g1[m4_a], b1[m4_a]]
            + fr[m4_a] * c111[m4_a]
        )

    m5 = (~cond_rg) & cond_gb & cond_br  # fg > fb >= fr
    if np.any(m5):
        out[m5] = (
            (1.0 - fg[m5]) * c000[m5]
            + (fg[m5] - fb[m5]) * table[r0[m5], g1[m5], b0[m5]]
            + (fb[m5] - fr[m5]) * table[r0[m5], g1[m5], b1[m5]]
            + fr[m5] * c111[m5]
        )

    m6 = (~cond_rg) & (~cond_br)  # fg > fr > fb
    if np.any(m6):
        out[m6] = (
            (1.0 - fg[m6]) * c000[m6]
            + (fg[m6] - fr[m6]) * table[r0[m6], g1[m6], b0[m6]]
            + (fr[m6] - fb[m6]) * table[r1[m6], g1[m6], b0[m6]]
            + fb[m6] * c111[m6]
        )

    return out.reshape(orig_shape)


def generate_hald_clut(lut: np.ndarray, level: int = 8) -> np.ndarray:
    """Render a 3D LUT onto a Level-L HaldCLUT uint8 image (for level=8: 512x512 = 64^3)."""
    cube_size = level * level
    side = level ** 3
    if side * side != cube_size ** 3:
        raise ValueError(f"Invalid Hald level {level}")
    # Standard HaldCLUT ordering: R fastest, then G, then B (identical to .cube lattice order)
    axis = np.linspace(0.0, 1.0, cube_size, dtype=np.float64)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    grid_rgb = np.stack([r, g, b], axis=-1).reshape((-1, 3))
    graded = np.clip(apply_lut(grid_rgb, lut), 0.0, 1.0)
    img_u8 = np.rint(graded * 255.0).astype(np.uint8).reshape((side, side, 3))
    return img_u8


def save_hald_clut(lut: np.ndarray, path: str | Path, level: int = 8) -> None:
    arr = generate_hald_clut(lut, level=level)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(p)


def load_hald_clut(path: str | Path, target_size: int | None = None) -> np.ndarray:
    """Parse a HaldCLUT PNG into a (N, N, N, 3) [R, G, B] LUT."""
    img = load_image(path)
    h, w, _ = img.shape
    if h != w:
        raise ValueError(f"HaldCLUT image must be square, got {w}x{h}")
    total = h * w
    cube_size = round(total ** (1.0 / 3.0))
    if cube_size ** 3 != total:
        raise ValueError(f"Image pixel count {total} is not a cube")
    # Stored as (B, G, R, 3) -> transpose to (R, G, B, 3)
    lut = np.transpose(img.reshape((cube_size, cube_size, cube_size, 3)), (2, 1, 0, 3)).copy()
    if target_size is not None and target_size != cube_size:
        ident = identity_lut(target_size)
        return apply_lut(ident, lut)
    return lut


def load_image(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {p}")
    with Image.open(p) as im:
        rgb = im.convert("RGB")
        return np.asarray(rgb, dtype=np.float64) / 255.0


def save_image(image_rgb: np.ndarray, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.rint(np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(u8, mode="RGB").save(p)


def inspect_lut(lut: np.ndarray) -> dict[str, float]:
    """Compute structural quality metrics for a (N, N, N, 3) 3D LUT."""
    arr = np.asarray(lut, dtype=np.float64)
    n = arr.shape[0]
    eps = 1e-6

    # Monotonicity inversions: per-axis self-channel AND Rec.709 luma along all 3 axes
    luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    inv_r = np.sum(np.diff(arr[..., 0], axis=0) < -eps) + np.sum(np.diff(luma, axis=0) < -eps)
    inv_g = np.sum(np.diff(arr[..., 1], axis=1) < -eps) + np.sum(np.diff(luma, axis=1) < -eps)
    inv_b = np.sum(np.diff(arr[..., 2], axis=2) < -eps) + np.sum(np.diff(luma, axis=2) < -eps)
    total_checks = 6 * (n - 1) * n * n
    inversion_count = int(inv_r + inv_g + inv_b)
    inversion_rate = float(inversion_count / total_checks)

    # 3D discrete Laplacian roughness on interior nodes
    center = arr[1:-1, 1:-1, 1:-1]
    lap = (
        arr[2:, 1:-1, 1:-1]
        + arr[:-2, 1:-1, 1:-1]
        + arr[1:-1, 2:, 1:-1]
        + arr[1:-1, :-2, 1:-1]
        + arr[1:-1, 1:-1, 2:]
        + arr[1:-1, 1:-1, :-2]
        - 6.0 * center
    )
    laplacian_roughness = float(np.sqrt(np.mean(np.sum(lap ** 2, axis=-1))))

    # Max Euclidean step across adjacent grid nodes
    step_r = np.linalg.norm(np.diff(arr, axis=0), axis=-1)
    step_g = np.linalg.norm(np.diff(arr, axis=1), axis=-1)
    step_b = np.linalg.norm(np.diff(arr, axis=2), axis=-1)
    max_step = float(max(np.max(step_r), np.max(step_g), np.max(step_b)))

    # Boundary clipping fraction (fraction of lattice nodes with any channel < 1e-6 or > 1 - 1e-6)
    clipped = np.any((arr < 1e-6) | (arr > 1.0 - 1e-6), axis=-1)
    clipping_fraction = float(np.mean(clipped))

    # Mean absolute hue shift on skin-tone lattice nodes
    ident = identity_lut(n)
    ok_in = srgb_to_oklab(ident)
    ok_out = srgb_to_oklab(np.clip(arr, 0.0, 1.0))
    c_in = np.hypot(ok_in[..., 1], ok_in[..., 2])
    h_in = np.degrees(np.arctan2(ok_in[..., 2], ok_in[..., 1])) % 360.0
    h_out = np.degrees(np.arctan2(ok_out[..., 2], ok_out[..., 1])) % 360.0
    skin_mask = (
        (ok_in[..., 0] >= 0.25)
        & (ok_in[..., 0] <= 0.85)
        & (c_in >= 0.02)
        & (c_in <= 0.18)
        & (h_in >= 30.0)
        & (h_in <= 65.0)
    )
    if np.any(skin_mask):
        dh = (h_out[skin_mask] - h_in[skin_mask] + 180.0) % 360.0 - 180.0
        skin_hue_shift_deg = float(np.mean(np.abs(dh)))
    else:
        skin_hue_shift_deg = 0.0

    return {
        "inversion_count": inversion_count,
        "inversion_rate": inversion_rate,
        "laplacian_roughness": laplacian_roughness,
        "max_step": max_step,
        "clipping_fraction": clipping_fraction,
        "skin_hue_shift_deg": skin_hue_shift_deg,
    }
