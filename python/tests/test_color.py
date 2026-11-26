import numpy as np
import pytest

from lut_match.color import (
    SUPPORTED_IDTS,
    apply_idt,
    ciede2000,
    linear_to_oklab,
    linear_to_srgb,
    oklab_to_linear,
    rgb_to_lab,
    srgb_to_linear,
)


def test_srgb_linear_roundtrip_and_continuity():
    x = np.linspace(0.0, 1.0, 1001)
    lin = srgb_to_linear(x)
    rec = linear_to_srgb(lin)
    assert np.allclose(rec, x, atol=1e-7)
    # Check continuity across piecewise threshold 0.04045
    lo = srgb_to_linear(np.array([0.0404499]))[0]
    hi = srgb_to_linear(np.array([0.0404501]))[0]
    assert abs(hi - lo) < 1e-6


def test_oklab_roundtrip_and_achromatic():
    rng = np.random.default_rng(7)
    rgb = rng.uniform(0.0, 1.0, size=(200, 3))
    lab = linear_to_oklab(rgb)
    rec = oklab_to_linear(lab)
    assert np.allclose(rec, rgb, atol=1e-6)

    grays = np.repeat(np.linspace(0.0, 1.0, 21)[:, None], 3, axis=1)
    g_lab = linear_to_oklab(grays)
    assert np.allclose(g_lab[:, 1:], 0.0, atol=1e-6)
    assert np.all(np.diff(g_lab[:, 0]) > 0.0)


@pytest.mark.parametrize("idt", SUPPORTED_IDTS)
def test_idts_preserve_neutrals_and_monotonicity(idt: str):
    ramp = np.repeat(np.linspace(0.15, 0.85, 32)[:, None], 3, axis=1)
    out = apply_idt(ramp, idt=idt)
    assert out.shape == ramp.shape
    assert np.all((out >= 0.0) & (out <= 1.0))
    # D65 neutral ramp stays nearly neutral and strictly monotone
    assert np.max(np.ptp(out, axis=1)) < 0.02
    assert np.all(np.diff(np.mean(out, axis=1)) >= -1e-7)


def test_idt_invalid_raises():
    with pytest.raises(ValueError, match="Unsupported IDT"):
        apply_idt(np.zeros((2, 2, 3)), idt="bogus-log")


def test_sharma_2005_ciede2000_vectors():
    # Exact pairs from Sharma, Wu, Dalal (2005)
    pairs = [
        ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
        ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
        ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
        # Pure neutral / zero-chroma edge cases
        ((50.0, 0.0, 0.0), (50.0, 0.0, 0.0), 0.0),
        ((50.0, 0.0, 0.0), (60.0, 0.0, 0.0), 9.470579),
        # Hue wrap-around symmetry
        ((50.0, 20.0, 0.1), (50.0, 20.0, -0.1), None),
    ]
    for lab1, lab2, expected in pairs:
        de = float(ciede2000(np.array(lab1), np.array(lab2)))
        de_rev = float(ciede2000(np.array(lab2), np.array(lab1)))
        assert np.isclose(de, de_rev, atol=1e-10)
        if expected is not None:
            assert np.isclose(de, expected, atol=1e-4)
        else:
            assert 0.0 < de < 0.5


def test_rgb_to_lab_d65_white_and_black():
    wb = rgb_to_lab(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    assert np.allclose(wb[0], [0.0, 0.0, 0.0], atol=1e-4)
    assert np.allclose(wb[1], [100.0, 0.0, 0.0], atol=1e-2)
