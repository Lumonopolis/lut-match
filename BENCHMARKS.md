# lut-match Benchmarks & Verification

Full quantitative evaluation, visual contact sheets, and Python ↔ TypeScript parity verification for `lut-match`.

## 1. Real-Photo Kodak-24 Benchmark (24 Photographs × 15 Cinematic LUTs)

Evaluated on held-out real photographs from the public-domain Kodak-24 suite using OKLab Luma + Tone + Hue-Harmonic (`k=1..3` Fourier) + Chroma transport with smooth 3D lattice projection ([bench/run_kodak_bench.py](bench/run_kodak_bench.py), [bench/results_kodak.json](bench/results_kodak.json)). Every table includes the no-op `identity` LUT explicitly.

### Paired Look Recovery (`src = A, ref = LUT_k(A) → held-out photo C`, 90 trials)

| Method | ΔE00 ↓ | Skin ΔE00 ↓ | Roughness ↓ | Inversion % |
| --- | ---: | ---: | ---: | ---: |
| **lut-match** | **1.430** | 2.480 | 0.01732 | **0.0000%** |
| mkl_oklab | 1.502 | 1.410 | 0.01955 | 4.0357% |
| hm_mvgd_hm | 1.528 | **1.271** | 0.01098 | 2.5637% |
| mkl_rgb | 1.827 | 1.586 | **0.00819** | 6.8060% |
| hist_match | 2.419 | 2.226 | 0.01014 | 0.0000% |
| reinhard_lab | 2.777 | 2.477 | 0.02160 | 4.4853% |
| identity | 5.987 | 5.588 | 0.00000 | 0.0000% |

### Unbiased Multi-Still Look Transfer (`src = [A1..A5], refs = [LUT_k(B1..B5)] → held-out photo C`, 180 trials)

Evaluated across 12 seeded random disjoint splits of Kodak-24 (5 ungraded source photos, 5 graded reference photos, 1 held-out test photo), with every baseline pooling the exact same 5 source and 5 reference photos symmetrically:

| Method | ΔE00 ↓ | Skin ΔE00 ↓ | Roughness ↓ | Inversion % |
| --- | ---: | ---: | ---: | ---: |
| **lut-match** | **4.164** | **4.207** | **0.00479** | **0.0000%** |
| identity | 6.099 | 5.511 | 0.00000 | 0.0000% |
| reinhard_lab | 7.189 | 8.100 | 0.03267 | 5.9579% |
| mkl_oklab | 7.302 | 8.417 | 0.02972 | 5.9420% |
| mkl_rgb | 7.983 | 9.129 | 0.01022 | 7.8260% |
| hist_match | 9.132 | 8.174 | 0.04025 | 0.0000% |
| hm_mvgd_hm | 9.584 | 9.708 | 0.03549 | 5.1348% |

### Single-Still Cross-Scene Robustness (`src = A, ref = LUT_k(B) → held-out photo C`, 90 trials)

| Method | Cross ΔE00 ↓ | Skin ΔE00 ↓ | Skin Hue Err ↓ | Roughness ↓ | Inversion % |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity | 5.987 | 5.588 | 7.82° | 0.00000 | 0.0000% |
| **lut-match (`balanced`)** | **6.006** | **5.978** | **12.06°** | **0.00644** | **0.0000%** |
| `lut-match (creative)` | 7.593 | 7.514 | 15.74° | 0.00811 | **0.0000%** |
| reinhard_lab | 12.590 | 15.600 | 32.10° | 0.05564 | 8.4572% |
| mkl_oklab | 12.886 | 15.638 | 34.52° | 0.05576 | 10.5218% |
| hist_match | 13.652 | 14.779 | 32.94° | 0.05681 | 0.0000% |
| hm_mvgd_hm | 13.673 | 14.360 | 34.81° | 0.05297 | 5.4336% |
| mkl_rgb | 13.834 | 16.169 | 35.85° | 0.02004 | 11.8536% |

### Visual Contact Sheets

- [Paired look recovery contact sheet](qa/kodak-paired-check.png)
- [Unbiased 5-still transfer vs Reinhard contact sheet](qa/kodak-multi-check.png)
- [Single-still Balanced vs Creative transfer contact sheet](qa/kodak-creative-check.png)

## 2. Synthetic 16-Scene Reproducible Suite (`bench/run_bench.py`)

Self-contained benchmark requiring no external image downloads ([bench/results.json](bench/results.json)):

- **Paired look recovery (90 trials):** `1.289` ΔE00 (`-77.7%` vs `identity` `5.770`, beats `mkl_oklab` `1.292` with `0.0000%` inversions vs `3.78%`).
- **Unbiased 5-still transfer (180 trials):** `4.562` ΔE00 (`-22.8%` vs `identity` `5.911`, `-51.7%` vs best pooled baseline `reinhard_lab` `9.451`, `0.0000%` inversions).
- **Single-still cross-scene & adversarial robustness (90 cross + 45 adversarial trials):** `7.173` cross ΔE00 (`-64.7%` vs `hist_match` `20.328`), `5.714` adversarial skin ΔE00 (`-68.3%` vs `reinhard_lab` `18.050`), `0.00618` roughness (`-74.9%` vs `mkl_rgb`), `0.0000%` inversions.

## 3. Python ↔ TypeScript Parity (All 6 IDTs & Both Modes)

Verified by `python bench/verify_py_ts_parity.py` across 12 test cases (`srgb`, `rec709`, `apple-log`, `slog3`, `vlog`, `clog3`, `balanced`, `creative`, multi-still, and adversarial neon-red):

- **Max entry abs diff:** `2.98e-08` to `5.88e-08` (all `< 1e-7`)
- **Steady-state Node V8 `33³` solve latency:** `8–30 ms`
- **sRGB & Rec.709 monotonicity inversions:** `0` across all cases

## Reproducing All Checks

```bash
pytest python/tests -q
cd packages/lut-match-core && npm run build && npm test && cd ../..
python bench/verify_py_ts_parity.py
python bench/run_bench.py
python bench/run_kodak_bench.py --kodak-dir path/to/kodak24
```
