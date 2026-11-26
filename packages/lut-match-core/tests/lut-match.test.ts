import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  applyLutCpu,
  createIdentityLut,
  evaluateLutPoint,
  fitLut,
  generateHaldClut,
  IDT_LIST,
  idtChannelToLinear,
  inspectLut,
  linearRgbToOklab,
  lutIndex,
  oklabToLinearRgb,
  parseCube,
  serializeCube,
  srgbToLinear,
  linearToSrgb,
  type ImageDataLike,
} from '../dist/index.js';

function makeSyntheticScene(
  width: number,
  height: number,
  mode: 'portrait' | 'teal-orange' | 'neon-red' | 'slog3-flat'
): ImageDataLike {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    const v = y / Math.max(1, height - 1);
    for (let x = 0; x < width; x++) {
      const u = x / Math.max(1, width - 1);
      const p = (y * width + x) * 4;
      let r = 0;
      let g = 0;
      let b = 0;

      if (mode === 'portrait') {
        // Rich gradients + warm skin patch in center
        const dist = Math.hypot(u - 0.5, v - 0.5);
        if (dist < 0.25) {
          r = 0.82 - v * 0.12;
          g = 0.61 - v * 0.1;
          b = 0.49 - v * 0.08;
        } else {
          r = 0.15 + 0.7 * u * (1 - 0.4 * v);
          g = 0.22 + 0.6 * (1 - v);
          b = 0.35 + 0.55 * (1 - u) * (1 - 0.5 * v);
        }
      } else if (mode === 'teal-orange') {
        r = 0.06 + 0.85 * Math.pow(u, 1.3);
        g = 0.28 + 0.42 * (1 - v);
        b = 0.42 + 0.45 * (1 - u);
      } else if (mode === 'neon-red') {
        // Adversarial monochromatic / hyper-saturated neon red
        r = 0.25 + 0.75 * u;
        g = 0.02 + 0.06 * v;
        b = 0.03 + 0.05 * (1 - v);
      } else {
        // Compressed log-like flat scene
        r = 0.32 + 0.24 * u;
        g = 0.34 + 0.22 * (1 - v);
        b = 0.36 + 0.2 * u * (1 - v);
      }

      data[p] = Math.round(Math.max(0, Math.min(1, r)) * 255);
      data[p + 1] = Math.round(Math.max(0, Math.min(1, g)) * 255);
      data[p + 2] = Math.round(Math.max(0, Math.min(1, b)) * 255);
      data[p + 3] = 255;
    }
  }
  return { width, height, data };
}

describe('color transforms', () => {
  it('round-trips sRGB <-> linear and linear <-> OKLab within 1e-6', () => {
    const lab: [number, number, number] = [0, 0, 0];
    const rgbOut: [number, number, number] = [0, 0, 0];
    for (let i = 0; i <= 10; i++) {
      const s = i / 10;
      const lin = srgbToLinear(s);
      const back = linearToSrgb(lin);
      assert.ok(Math.abs(back - s) < 1e-6, `sRGB round-trip failed for ${s}`);
    }

    const samples: [number, number, number][] = [
      [0, 0, 0],
      [1, 1, 1],
      [0.8, 0.2, 0.1],
      [0.12, 0.55, 0.72],
      [0.45, 0.45, 0.45],
    ];
    for (const [r, g, b] of samples) {
      linearRgbToOklab(r, g, b, lab);
      oklabToLinearRgb(lab[0], lab[1], lab[2], rgbOut);
      assert.ok(Math.abs(rgbOut[0] - r) < 1e-6);
      assert.ok(Math.abs(rgbOut[1] - g) < 1e-6);
      assert.ok(Math.abs(rgbOut[2] - b) < 1e-6);
    }
  });

  it('keeps all IDTs monotonic across [0, 1]', () => {
    for (const idt of IDT_LIST) {
      let prev = -Infinity;
      for (let i = 0; i <= 100; i++) {
        const val = idtChannelToLinear(i / 100, idt);
        assert.ok(val >= prev - 1e-12, `IDT ${idt} non-monotonic at ${i}%`);
        prev = val;
      }
    }
  });
});

describe('.cube serialization and R-fastest indexing', () => {
  it('writes R-fastest / B-slowest order and round-trips accurately', () => {
    const lut = createIdentityLut(17, 'Test "RoundTrip"');
    const cubeText = serializeCube(lut);
    const lines = cubeText.trim().split('\n');
    assert.equal(lines[0], 'TITLE "Test  RoundTrip"');
    assert.equal(lines[1], 'LUT_3D_SIZE 17');

    // First entry (0,0,0), second entry must be (r=1, g=0, b=0) -> R fastest!
    assert.equal(lines[4], '0.000000 0.000000 0.000000');
    const step1 = (1 / 16).toFixed(6);
    assert.equal(lines[5], `${step1} 0.000000 0.000000`);

    // Entry at index 17 (r=0, g=1, b=0) -> G middle
    assert.equal(lines[4 + 17], `0.000000 ${step1} 0.000000`);

    // Entry at index 17*17 (r=0, g=0, b=1) -> B slowest
    assert.equal(lines[4 + 17 * 17], `0.000000 0.000000 ${step1}`);

    const parsed = parseCube(cubeText);
    assert.equal(parsed.size, 17);
    assert.equal(parsed.data.length, 17 * 17 * 17 * 3);
    const idx = lutIndex(3, 9, 12, 17);
    assert.ok(Math.abs(parsed.data[idx]! - 3 / 16) < 1e-5);
    assert.ok(Math.abs(parsed.data[idx + 1]! - 9 / 16) < 1e-5);
    assert.ok(Math.abs(parsed.data[idx + 2]! - 12 / 16) < 1e-5);
  });
});

describe('HaldCLUT integrity', () => {
  it('generates exact Level-8 512x512 HaldCLUT for identity and graded LUTs', () => {
    const idLut = createIdentityLut(33);
    const hald = generateHaldClut(idLut);
    assert.equal(hald.width, 512);
    assert.equal(hald.height, 512);
    assert.equal(hald.data.length, 512 * 512 * 4);

    // Pixel 0 -> (0, 0, 0, 255)
    assert.deepEqual(Array.from(hald.data.slice(0, 4)), [0, 0, 0, 255]);
    // Pixel 63 -> (255, 0, 0, 255)
    assert.deepEqual(Array.from(hald.data.slice(63 * 4, 63 * 4 + 4)), [255, 0, 0, 255]);
    // Pixel 64 -> (0, 4, 0, 255)
    assert.deepEqual(Array.from(hald.data.slice(64 * 4, 64 * 4 + 4)), [0, 4, 0, 255]);
    // Last pixel -> (255, 255, 255, 255)
    const last = (512 * 512 - 1) * 4;
    assert.deepEqual(Array.from(hald.data.slice(last, last + 4)), [255, 255, 255, 255]);
  });
});

describe('fitLut solver guarantees & latency', () => {
  it('guarantees 0.00% inversions on adversarial monochromatic neon-red reference', () => {
    const source = makeSyntheticScene(320, 180, 'portrait');
    const refNeon = makeSyntheticScene(320, 180, 'neon-red');

    const lut = fitLut({
      source,
      references: [refNeon],
      size: 33,
      strength: 1.0,
      skinProtect: 0.35,
      smoothness: 0.35,
      highlightRolloff: 0.15,
      shadowLift: 0.0,
    });

    const inspection = inspectLut(lut);
    assert.equal(inspection.inversionRate, 0);
    assert.ok(inspection.laplacianRoughness < 0.25);

    const out = applyLutCpu(source, lut);
    assert.equal(out.width, source.width);
    assert.equal(out.height, source.height);
  });

  it('preserves skin tone hue closer to source when skinProtect = 1 vs 0', () => {
    const source = makeSyntheticScene(240, 160, 'portrait');
    const refTeal = makeSyntheticScene(240, 160, 'teal-orange');

    const lutNoProtect = fitLut({
      source,
      references: [refTeal],
      size: 33,
      skinProtect: 0,
    });
    const lutProtected = fitLut({
      source,
      references: [refTeal],
      size: 33,
      skinProtect: 1,
    });

    // Test a canonical skin tone RGB (0.78, 0.56, 0.44)
    const a: [number, number, number] = [0, 0, 0];
    const b: [number, number, number] = [0, 0, 0];
    evaluateLutPoint(lutNoProtect, 0.78, 0.56, 0.44, a);
    evaluateLutPoint(lutProtected, 0.78, 0.56, 0.44, b);

    // Protected skin retains higher R-B warmth ratio than unprotected teal-shifted output
    assert.ok(b[0] - b[2] >= a[0] - a[2] - 1e-4);
  });

  it('solves 33^3 LUT under 150 ms and supports multi-reference + Log IDTs', () => {
    const source = makeSyntheticScene(640, 360, 'slog3-flat');
    const ref1 = makeSyntheticScene(640, 360, 'teal-orange');
    const ref2 = makeSyntheticScene(640, 360, 'portrait');

    // Warm-up JIT so benchmark reflects steady browser execution
    for (let w = 0; w < 2; w++) {
      fitLut({
        source,
        references: [ref1, ref2],
        size: 33,
        idt: 'slog3',
        highlightRolloff: 0.2,
        shadowLift: 0.05,
      });
    }

    let elapsed = Infinity;
    let lut = fitLut({
      source,
      references: [ref1, ref2],
      size: 33,
      idt: 'slog3',
      highlightRolloff: 0.2,
      shadowLift: 0.05,
    });
    elapsed = Math.min(elapsed, lut.diagnostics?.solveMs ?? Infinity);
    for (let r = 0; r < 2; r++) {
      const t0 = performance.now();
      lut = fitLut({
        source,
        references: [ref1, ref2],
        size: 33,
        idt: 'slog3',
        highlightRolloff: 0.2,
        shadowLift: 0.05,
      });
      elapsed = Math.min(elapsed, performance.now() - t0);
    }

    console.log(`33^3 fitLut multi-ref solveMs=${elapsed.toFixed(2)}ms`);
    assert.ok(elapsed < 150, `Expected <150ms solve time, got ${elapsed.toFixed(2)}ms`);
    const stats = inspectLut(lut);
    assert.ok(stats.laplacianRoughness < 0.5);
    const srgbLut = fitLut({ source: ref2, references: [ref1], size: 33, idt: 'srgb' });
    assert.equal(inspectLut(srgbLut).inversionCount, 0);
  });

  it('bakes slog3 IDT into 3D LUT lattice so direct apply recovers graded display sRGB', () => {
    const rec709ToSgamut3 = [
      0.645679477609992, 0.25911454699971487, 0.0952059753902934,
      0.08752999151186004, 0.7596995625540706, 0.15277044593406927,
      0.03695741988752367, 0.1292809047663383, 0.8337616753461381,
    ] as const;

    const invSlog3Pixel = (r: number, g: number, b: number): [number, number, number] => {
      const invTone = (s: number) => {
        const y = Math.min(0.96, Math.max(0, srgbToLinear(s)));
        const a = 0.12 * (1 - y);
        const bb = 1 - 0.85 * y;
        const c = -0.95 * y;
        return (-bb + Math.sqrt(Math.max(0, bb * bb - 4 * a * c))) / Math.max(2 * a, 1e-12);
      };
      const r7 = invTone(r);
      const g7 = invTone(g);
      const b7 = invTone(b);
      const sr = rec709ToSgamut3[0] * r7 + rec709ToSgamut3[1] * g7 + rec709ToSgamut3[2] * b7;
      const sg = rec709ToSgamut3[3] * r7 + rec709ToSgamut3[4] * g7 + rec709ToSgamut3[5] * b7;
      const sb = rec709ToSgamut3[6] * r7 + rec709ToSgamut3[7] * g7 + rec709ToSgamut3[8] * b7;
      const linToSlog3 = (lin: number) => {
        const t =
          lin >= 0.01125
            ? 420.0 + Math.log10(Math.max((lin + 0.01) / 0.19, 1e-9)) * 261.5
            : (lin * (171.2102946929 - 95.0)) / 0.01125 + 95.0;
        return Math.min(1, Math.max(0, t / 1023.0));
      };
      return [linToSlog3(sr), linToSlog3(sg), linToSlog3(sb)];
    };

    const srgbScene = makeSyntheticScene(128, 96, 'portrait');
    const nPix = srgbScene.width * srgbScene.height;
    const slog3Data = new Float32Array(nPix * 4);
    const targetData = new Float32Array(nPix * 4);
    for (let i = 0; i < nPix; i++) {
      const p = i * 4;
      const sr = (srgbScene.data[p] ?? 0) / 255;
      const sg = (srgbScene.data[p + 1] ?? 0) / 255;
      const sb = (srgbScene.data[p + 2] ?? 0) / 255;
      const [lr, lg, lb] = invSlog3Pixel(sr, sg, sb);
      slog3Data[p] = lr;
      slog3Data[p + 1] = lg;
      slog3Data[p + 2] = lb;
      slog3Data[p + 3] = 1;

      targetData[p] = Math.min(1, 0.92 * sr * 1.06 + 0.03);
      targetData[p + 1] = Math.min(1, 0.92 * sg * 1.0 + 0.03);
      targetData[p + 2] = Math.min(1, 0.92 * sb * 0.9 + 0.03);
      targetData[p + 3] = 1;
    }

    const slog3Img: ImageDataLike = { width: srgbScene.width, height: srgbScene.height, data: slog3Data };
    const targetImg: ImageDataLike = { width: srgbScene.width, height: srgbScene.height, data: targetData };

    const lutSlog3 = fitLut({ source: slog3Img, references: [targetImg], size: 33, idt: 'slog3', skinProtect: 0 });
    const lutSrgbLie = fitLut({ source: slog3Img, references: [targetImg], size: 33, idt: 'srgb', skinProtect: 0 });
    assert.equal(inspectLut(lutSrgbLie).inversionCount, 0);

    const outSlog3 = applyLutCpu(slog3Img, lutSlog3);
    const outLie = applyLutCpu(slog3Img, lutSrgbLie);

    let errSlog3 = 0;
    let errLie = 0;
    for (let i = 0; i < nPix; i++) {
      const p = i * 4;
      const dr1 = (outSlog3.data[p] ?? 0) - (targetData[p] ?? 0);
      const dg1 = (outSlog3.data[p + 1] ?? 0) - (targetData[p + 1] ?? 0);
      const db1 = (outSlog3.data[p + 2] ?? 0) - (targetData[p + 2] ?? 0);
      errSlog3 += Math.hypot(dr1, dg1, db1);

      const dr2 = (outLie.data[p] ?? 0) - (targetData[p] ?? 0);
      const dg2 = (outLie.data[p + 1] ?? 0) - (targetData[p + 1] ?? 0);
      const db2 = (outLie.data[p + 2] ?? 0) - (targetData[p + 2] ?? 0);
      errLie += Math.hypot(dr2, dg2, db2);
    }
    const meanRgbErrSlog3 = errSlog3 / nPix;
    const meanRgbErrLie = errLie / nPix;
    assert.ok(meanRgbErrSlog3 < 0.02, `Expected S-Log3 mean RGB error < 0.02, got ${meanRgbErrSlog3}`);
    assert.ok(meanRgbErrSlog3 < meanRgbErrLie * 0.3, `Expected S-Log3 (${meanRgbErrSlog3}) to beat sRGB lie (${meanRgbErrLie})`);
  });

  it('rejects unrelated landscapes, 1D ramps, and aspect-ratio mismatches from paired mode with 0 inversions, while recovering true pairs < 2.0 dE00', () => {
    const makeImg = (
      width: number,
      height: number,
      fn: (u: number, v: number) => [number, number, number]
    ): ImageDataLike => {
      const data = new Uint8ClampedArray(width * height * 4);
      for (let y = 0; y < height; y++) {
        const v = y / Math.max(1, height - 1);
        for (let x = 0; x < width; x++) {
          const u = x / Math.max(1, width - 1);
          const [r, g, b] = fn(u, v);
          const p = (y * width + x) * 4;
          data[p] = Math.round(Math.max(0, Math.min(1, r)) * 255);
          data[p + 1] = Math.round(Math.max(0, Math.min(1, g)) * 255);
          data[p + 2] = Math.round(Math.max(0, Math.min(1, b)) * 255);
          data[p + 3] = 255;
        }
      }
      return { width, height, data };
    };

    const srgbToCieLab = (r: number, g: number, b: number): [number, number, number] => {
      const lr = srgbToLinear(r);
      const lg = srgbToLinear(g);
      const lb = srgbToLinear(b);
      const x = (0.4124564 * lr + 0.3575761 * lg + 0.1804375 * lb) / 0.95047;
      const y = (0.2126729 * lr + 0.7151522 * lg + 0.0721750 * lb) / 1.00000;
      const z = (0.0193339 * lr + 0.1191920 * lg + 0.9503041 * lb) / 1.08883;
      const f = (t: number) => (t > 0.008856451679 ? Math.cbrt(t) : 7.787037037 * t + 16 / 116);
      const fx = f(x), fy = f(y), fz = f(z);
      return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
    };

    const ciede2000Pixel = (lab1: [number, number, number], lab2: [number, number, number]): number => {
      const [L1, a1, b1] = lab1;
      const [L2, a2, b2] = lab2;
      const C1 = Math.hypot(a1, b1), C2 = Math.hypot(a2, b2);
      const Cbar = 0.5 * (C1 + C2);
      const Cbar7 = Math.pow(Cbar, 7);
      const G = 0.5 * (1 - Math.sqrt(Cbar7 / (Cbar7 + Math.pow(25, 7))));
      const a1p = (1 + G) * a1, a2p = (1 + G) * a2;
      const C1p = Math.hypot(a1p, b1), C2p = Math.hypot(a2p, b2);
      const h1p = (C1p === 0) ? 0 : ((Math.atan2(b1, a1p) * 180) / Math.PI + 360) % 360;
      const h2p = (C2p === 0) ? 0 : ((Math.atan2(b2, a2p) * 180) / Math.PI + 360) % 360;
      const dLp = L2 - L1;
      const dCp = C2p - C1p;
      let dhp = 0;
      if (C1p * C2p !== 0) {
        const diff = h2p - h1p;
        dhp = Math.abs(diff) <= 180 ? diff : diff > 180 ? diff - 360 : diff + 360;
      }
      const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin((dhp * Math.PI) / 360);
      const Lbarp = 0.5 * (L1 + L2);
      const Cbarp = 0.5 * (C1p + C2p);
      let hbarp = h1p + h2p;
      if (C1p * C2p !== 0) {
        hbarp = Math.abs(h1p - h2p) <= 180 ? 0.5 * (h1p + h2p) : (h1p + h2p < 360 ? 0.5 * (h1p + h2p + 360) : 0.5 * (h1p + h2p - 360));
      }
      const T =
        1 -
        0.17 * Math.cos(((hbarp - 30) * Math.PI) / 180) +
        0.24 * Math.cos((2 * hbarp * Math.PI) / 180) +
        0.32 * Math.cos(((3 * hbarp + 6) * Math.PI) / 180) -
        0.20 * Math.cos(((4 * hbarp - 63) * Math.PI) / 180);
      const dTheta = 30 * Math.exp(-Math.pow((hbarp - 275) / 25, 2));
      const Cbarp7 = Math.pow(Cbarp, 7);
      const RC = 2 * Math.sqrt(Cbarp7 / (Cbarp7 + Math.pow(25, 7)));
      const L50 = Math.pow(Lbarp - 50, 2);
      const SL = 1 + (0.015 * L50) / Math.sqrt(20 + L50);
      const SC = 1 + 0.045 * Cbarp;
      const SH = 1 + 0.015 * Cbarp * T;
      const RT = -Math.sin((2 * dTheta * Math.PI) / 180) * RC;
      return Math.sqrt(
        Math.pow(dLp / SL, 2) +
          Math.pow(dCp / SC, 2) +
          Math.pow(dHp / SH, 2) +
          RT * (dCp / SC) * (dHp / SH)
      );
    };

    // 1) Two unrelated horizon landscapes (blue sky/green hills vs warm sunset/red canyon)
    const landscape1 = makeImg(128, 128, (u, v) => {
      const sky = v < 0.45 + 0.05 * Math.sin(2 * Math.PI * 2 * u);
      const tex = 0.5 + 0.5 * Math.sin(2 * Math.PI * 5 * u) * Math.cos(2 * Math.PI * 4 * v);
      return sky
        ? [0.30 + 0.25 * v, 0.55 + 0.20 * v, 0.88 - 0.10 * v]
        : [0.20 + 0.12 * tex, 0.45 + 0.25 * tex, 0.18 + 0.08 * tex];
    });
    const landscape2 = makeImg(128, 128, (u, v) => {
      const sky = v < 0.46 + 0.05 * Math.cos(2 * Math.PI * 3 * u + 1.1);
      const tex = 0.5 + 0.5 * Math.sin(2 * Math.PI * 7 * u + 0.7) * Math.cos(2 * Math.PI * 6 * v + 0.4);
      return sky
        ? [0.90 - 0.12 * v, 0.52 + 0.18 * v, 0.28 + 0.14 * v]
        : [0.62 + 0.22 * tex, 0.24 + 0.12 * tex, 0.16 + 0.06 * tex];
    });

    const lutLand = fitLut({ source: landscape1, references: [landscape2], size: 17 });
    assert.equal(inspectLut(lutLand).inversionCount, 0);
    // Regularized unpaired solver sets sourceSamples = 16384 (128*128), NOT 4096 from paired mode
    assert.equal(lutLand.diagnostics?.sourceSamples, 128 * 128);

    // 2) Unrelated linear ramps & mismatched aspect ratios (16:9 vs 4:5)
    const ramp16x9 = makeImg(160, 90, (u) => {
      const val = 0.05 + 0.90 * u;
      return [val, 0.5 * val, 0.25 * val];
    });
    const ramp4x5 = makeImg(96, 120, (u) => {
      const val = 0.10 + 0.80 * u;
      return [0.2 * val, 0.8 * val, val];
    });
    const lutAspect = fitLut({ source: ramp16x9, references: [ramp4x5], size: 17 });
    assert.equal(inspectLut(lutAspect).inversionCount, 0);
    assert.equal(lutAspect.diagnostics?.sourceSamples, 160 * 90);

    // Same-aspect 1D smooth ramps (no 2D gradients) must also reject paired mode
    const rampSqA = makeImg(128, 128, (_u, v) => {
      const val = 0.05 + 0.90 * v;
      return [val, val, val];
    });
    const rampSqB = makeImg(128, 128, (_u, v) => {
      const val = 0.05 + 0.90 * v;
      return [Math.min(1, val * 1.05), val * 0.95, val * 0.85];
    });
    const lutRamp = fitLut({ source: rampSqA, references: [rampSqB], size: 17 });
    assert.equal(inspectLut(lutRamp).inversionCount, 0);
    assert.equal(lutRamp.diagnostics?.sourceSamples, 128 * 128);

    // 3) True same-scene graded pair (A vs LUT(A)) triggers paired mode and recovers dE00 < 2.0
    const graded1 = makeImg(128, 128, (u, v) => {
      const p = (yIdx: number) => (landscape1.data[yIdx]! / 255);
      const baseIdx = (Math.round(v * 127) * 128 + Math.round(u * 127)) * 4;
      return [
        Math.min(1, 0.90 * p(baseIdx) * 1.08 + 0.03),
        Math.min(1, 0.90 * p(baseIdx + 1) * 0.98 + 0.03),
        Math.min(1, 0.90 * p(baseIdx + 2) * 0.88 + 0.03),
      ];
    });
    const lutPaired = fitLut({ source: landscape1, references: [graded1], size: 33, skinProtect: 0 });
    assert.equal(inspectLut(lutPaired).inversionCount, 0);
    assert.equal(lutPaired.diagnostics?.sourceSamples, 4096);

    const outPaired = applyLutCpu(landscape1, lutPaired);
    const nPix = 128 * 128;
    let totalDe00 = 0;
    for (let i = 0; i < nPix; i++) {
      const p = i * 4;
      const labP = srgbToCieLab(outPaired.data[p]! / 255, outPaired.data[p + 1]! / 255, outPaired.data[p + 2]! / 255);
      const labG = srgbToCieLab(graded1.data[p]! / 255, graded1.data[p + 1]! / 255, graded1.data[p + 2]! / 255);
      totalDe00 += ciede2000Pixel(labP, labG);
    }
    const meanDe00 = totalDe00 / nPix;
    assert.ok(meanDe00 < 2.0, `Expected paired recovery dE00 < 2.0, got ${meanDe00.toFixed(3)}`);

    // Multi-source pooling works cleanly with 0 inversions
    const lutMultiSource = fitLut({
      source: [landscape1, landscape2],
      references: [graded1, landscape2],
      size: 17,
    });
    assert.equal(inspectLut(lutMultiSource).inversionCount, 0);
  });

  it('keeps all 6 IDTs continuous and monotonically non-decreasing on a 2001-point neutral ramp', () => {
    const n = 2001;
    const dummy = makeSyntheticScene(16, 16, 'portrait');
    for (const idt of IDT_LIST) {
      let prev = -Infinity;
      for (let i = 0; i < n; i++) {
        const v = i / (n - 1);
        const cur = idtChannelToLinear(v, idt);
        assert.ok(cur >= prev - 1e-12, `${idt} channel curve inverted at v=${v}: ${prev} -> ${cur}`);
        prev = cur;
      }
      const idtLut = fitLut({ source: dummy, references: [dummy], size: 33, strength: 0, idt });
      for (let i = 1; i < 33; i++) {
        const idxPrev = ((i - 1) * 33 * 33 + (i - 1) * 33 + (i - 1)) * 3;
        const idxCurr = (i * 33 * 33 + i * 33 + i) * 3;
        assert.ok(idtLut.data[idxCurr]! >= idtLut.data[idxPrev]! - 1e-6, `${idt} neutral ramp inverted at ${i}`);
      }
    }
    // Explicit Canon C-Log3 upper junction continuity check around y = 0.15277891
    const cLo = idtChannelToLinear(0.1527, 'clog3');
    const cHi = idtChannelToLinear(0.1528, 'clog3');
    assert.ok(cHi > cLo, `Expected clog3 monotone across 0.15278 junction: ${cLo} -> ${cHi}`);
    assert.ok(cHi - cLo < 2e-4, `Expected clog3 continuous across 0.15278 junction: diff=${cHi - cLo}`);
  });

  it('supports creative transfer mode with stronger look transfer and zero sRGB inversions', () => {
    const src = makeSyntheticScene(160, 100, 'portrait');
    const ref = makeSyntheticScene(160, 100, 'teal-orange');

    const lutBal = fitLut({ source: src, references: [ref], size: 33, mode: 'balanced', skinProtect: 0.5 });
    const lutCre = fitLut({ source: src, references: [ref], size: 33, mode: 'creative', skinProtect: 0.5 });
    assert.equal(inspectLut(lutBal).inversionCount, 0);
    assert.equal(inspectLut(lutCre).inversionCount, 0);

    const ident = createIdentityLut(33);
    let diffBal = 0;
    let diffCre = 0;
    for (let i = 0; i < ident.data.length; i++) {
      diffBal += Math.abs(lutBal.data[i]! - ident.data[i]!);
      diffCre += Math.abs(lutCre.data[i]! - ident.data[i]!);
    }
    assert.ok(diffCre > diffBal * 1.2, `Expected creative transfer (${diffCre}) > balanced (${diffBal})`);

    assert.throws(() =>
      fitLut({ source: src, references: [ref], size: 17, mode: 'bogus' as never })
    );
  });
});
