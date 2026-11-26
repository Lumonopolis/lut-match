import {
  clamp,
  clamp01,
  clog3ToLinear,
  appleLogToLinear,
  linearToSrgb,
  rec709ToLinear,
  slog3ToLinear,
  srgbToLinear,
  vlogToLinear,
} from './color.js';
import { evaluateLutPoint, lutIndex } from './io.js';
import type {
  FitLutOptions,
  IdtName,
  ImageDataLike,
  Lut3D,
  LutSize,
  TransferMode,
} from './types.js';

const MAX_SAMPLES_PER_IMAGE = 32768;
const BAND_CENTERS = [0.14, 0.32, 0.50, 0.68, 0.86] as const;
const BAND_SIGMA = 0.16;
const BAND_STEP = 0.18;
const INV_2_BAND_SIGMA2 = 1.0 / (2.0 * BAND_SIGMA * BAND_SIGMA);
const BAND_EXP_STEP2 = 2.0 * BAND_STEP * INV_2_BAND_SIGMA2;
const BAND_Q1 = Math.exp(-(BAND_STEP * BAND_STEP) * INV_2_BAND_SIGMA2);
const BAND_Q2 = Math.exp(-(4.0 * BAND_STEP * BAND_STEP) * INV_2_BAND_SIGMA2);
const QUANTILES = [
  0.01, 0.03, 0.07, 0.12, 0.20, 0.30, 0.42, 0.55, 0.68, 0.80, 0.90, 0.96, 0.99,
] as const;
const VOXEL_SPACE = 26 * 33 * 33;

const HUE_BINS = 72;
const HUE_ANCHOR_COUNT = 24;
const HUE_SIGMA = (28.0 * Math.PI) / 180.0;
const INV_2_HUE_SIGMA2 = 1.0 / (2.0 * HUE_SIGMA * HUE_SIGMA);
const HUE_BIN_CENTERS = new Float64Array(HUE_BINS);
for (let b = 0; b < HUE_BINS; b++) {
  HUE_BIN_CENTERS[b] = -Math.PI + (2.0 * Math.PI * (b + 0.5)) / HUE_BINS;
}
const HUE_ANCHORS = new Float64Array(HUE_ANCHOR_COUNT);
const HUE_BASIS = new Float64Array(HUE_ANCHOR_COUNT * 6);
const HUE_DIFF_MB = new Float64Array(HUE_ANCHOR_COUNT * HUE_BINS);
const HUE_KERNEL_MB = new Float64Array(HUE_ANCHOR_COUNT * HUE_BINS);
for (let m = 0; m < HUE_ANCHOR_COUNT; m++) {
  const phi = -Math.PI + (2.0 * Math.PI * (m + 0.5)) / HUE_ANCHOR_COUNT;
  HUE_ANCHORS[m] = phi;
  HUE_BASIS[m * 6] = Math.cos(phi);
  HUE_BASIS[m * 6 + 1] = Math.sin(phi);
  HUE_BASIS[m * 6 + 2] = Math.cos(2.0 * phi);
  HUE_BASIS[m * 6 + 3] = Math.sin(2.0 * phi);
  HUE_BASIS[m * 6 + 4] = Math.cos(3.0 * phi);
  HUE_BASIS[m * 6 + 5] = Math.sin(3.0 * phi);
  const rowOff = m * HUE_BINS;
  for (let b = 0; b < HUE_BINS; b++) {
    let d = HUE_BIN_CENTERS[b]! - phi;
    if (d > Math.PI) d -= 2.0 * Math.PI;
    else if (d < -Math.PI) d += 2.0 * Math.PI;
    HUE_DIFF_MB[rowOff + b] = d;
    HUE_KERNEL_MB[rowOff + b] = Math.exp(-d * d * INV_2_HUE_SIGMA2);
  }
}
const HUE_REG_DIAG = [0.06, 0.06, 0.12, 0.12, 0.24, 0.24] as const;
const PAIRED_REG_DIAG = [
  1e-6, 1e-5, 2e-3, 2e-3, 2e-3, 2e-2, 2e-2,
  5e-3, 8e-3, 8e-3, 1.5e-2, 1.5e-2, 3e-2, 3e-2,
] as const;

interface ModeConfig {
  lumaAlpha: number;
  mklAlpha: number;
  shiftAlpha: number;
  shiftMaxScale: number;
  shiftMaxBase: number;
  hueAlpha: number;
  satAlpha: number;
  maxHueRad: number;
  maxLogSat: number;
}

const MODE_CONFIG: Record<TransferMode, ModeConfig> = {
  balanced: {
    lumaAlpha: 0.35,
    mklAlpha: 0.45,
    shiftAlpha: 0.55,
    shiftMaxScale: 0.065,
    shiftMaxBase: 0.015,
    hueAlpha: 0.35,
    satAlpha: 0.35,
    maxHueRad: (10.0 * Math.PI) / 180.0,
    maxLogSat: 0.22,
  },
  creative: {
    lumaAlpha: 0.70,
    mklAlpha: 0.75,
    shiftAlpha: 0.75,
    shiftMaxScale: 0.075,
    shiftMaxBase: 0.018,
    hueAlpha: 0.55,
    satAlpha: 0.55,
    maxHueRad: (14.0 * Math.PI) / 180.0,
    maxLogSat: 0.28,
  },
};

// Reusable scratch buffers for zero-GC 33^3 solves
let scratchCubeOrig = new Float64Array(33 * 33 * 33 * 3);
let scratchCubeCur = new Float64Array(33 * 33 * 33 * 3);
let scratchCubeWork = new Float64Array(33 * 33 * 33 * 3);
let scratchLuma = new Float64Array(33 * 33 * 33);
let scratchViolMask = new Uint8Array(33 * 33);
let scratchSortV = new Float64Array(MAX_SAMPLES_PER_IMAGE);
let scratchSortW = new Float64Array(MAX_SAMPLES_PER_IMAGE);
let scratchCdf = new Float64Array(MAX_SAMPLES_PER_IMAGE);
let scratchVoxelCounts = new Int32Array(VOXEL_SPACE);
let scratchSampleKeys = new Int32Array(MAX_SAMPLES_PER_IMAGE);
let scratchSrcL = new Float64Array(MAX_SAMPLES_PER_IMAGE);
let scratchSrcA = new Float64Array(MAX_SAMPLES_PER_IMAGE);
let scratchSrcB = new Float64Array(MAX_SAMPLES_PER_IMAGE);
let scratchSrcW = new Float64Array(MAX_SAMPLES_PER_IMAGE);
let scratchRefL = new Float64Array(MAX_SAMPLES_PER_IMAGE);
let scratchRefA = new Float64Array(MAX_SAMPLES_PER_IMAGE);
let scratchRefB = new Float64Array(MAX_SAMPLES_PER_IMAGE);
let scratchRefW = new Float64Array(MAX_SAMPLES_PER_IMAGE);
let scratchSrcNeu = new Float64Array(MAX_SAMPLES_PER_IMAGE);
let scratchRefNeu = new Float64Array(MAX_SAMPLES_PER_IMAGE);
let scratchSrcBandW = new Float64Array(MAX_SAMPLES_PER_IMAGE);
let scratchRefBandW = new Float64Array(MAX_SAMPLES_PER_IMAGE);
let scratchSrcMklA = new Float64Array(MAX_SAMPLES_PER_IMAGE);
let scratchSrcMklB = new Float64Array(MAX_SAMPLES_PER_IMAGE);

let scratchPavaLine = new Float64Array(65);
let scratchPavaCols = new Float64Array(65 * 3);
let scratchBlockVal = new Float64Array(65);
let scratchBlockCount = new Int32Array(65);

function ensureCubeScratch(size: number): void {
  const n3 = size * size * size;
  const len = n3 * 3;
  if (scratchCubeWork.length < len) {
    scratchCubeOrig = new Float64Array(len);
    scratchCubeCur = new Float64Array(len);
    scratchCubeWork = new Float64Array(len);
    scratchLuma = new Float64Array(n3);
  }
  if (scratchViolMask.length < size * size) {
    scratchViolMask = new Uint8Array(size * size);
  }
  if (scratchPavaLine.length < size) {
    scratchPavaLine = new Float64Array(size);
    scratchPavaCols = new Float64Array(size * 3);
    scratchBlockVal = new Float64Array(size);
    scratchBlockCount = new Int32Array(size);
  }
}

function ensureSrcScratch(n: number): void {
  if (scratchSrcL.length < n) {
    scratchSrcL = new Float64Array(n);
    scratchSrcA = new Float64Array(n);
    scratchSrcB = new Float64Array(n);
    scratchSrcW = new Float64Array(n);
    scratchSrcNeu = new Float64Array(n);
    scratchSrcBandW = new Float64Array(n);
    scratchSrcMklA = new Float64Array(n);
    scratchSrcMklB = new Float64Array(n);
  }
}

function ensureRefScratch(n: number): void {
  if (scratchRefL.length < n) {
    scratchRefL = new Float64Array(n);
    scratchRefA = new Float64Array(n);
    scratchRefB = new Float64Array(n);
    scratchRefW = new Float64Array(n);
    scratchRefNeu = new Float64Array(n);
    scratchRefBandW = new Float64Array(n);
  }
}

function ensureSampleScratch(n: number): void {
  if (scratchSortV.length < n) {
    scratchSortV = new Float64Array(n);
    scratchSortW = new Float64Array(n);
    scratchCdf = new Float64Array(n);
    scratchSampleKeys = new Int32Array(n);
  }
}

interface OklabSample {
  count: number;
  L: Float64Array;
  a: Float64Array;
  b: Float64Array;
  weights: Float64Array;
}

interface PchipSpline {
  xs: Float64Array;
  ys: Float64Array;
  ms: Float64Array;
}

interface BandTransform {
  muSrcA: number;
  muSrcB: number;
  muRefA: number;
  muRefB: number;
  t00: number;
  t01: number;
  t10: number;
  t11: number;
}

function validateImage(img: ImageDataLike, label: string): void {
  if (img.width <= 0 || img.height <= 0) {
    throw new Error(`${label} has invalid dimensions ${img.width}x${img.height}`);
  }
  const required = img.width * img.height * 4;
  if (img.data.length < required) {
    throw new Error(
      `${label} buffer too short: expected >= ${required}, got ${img.data.length}`
    );
  }
}

function clampInt(v: number, lo: number, hi: number): number {
  return v <= lo ? lo : v >= hi ? hi : v | 0;
}

function computeDensityWeightsInto(
  L: Float64Array,
  a: Float64Array,
  b: Float64Array,
  count: number,
  normalizeSum: boolean,
  weights: Float64Array
): Float64Array {
  ensureSampleScratch(count);
  const counts = scratchVoxelCounts;
  counts.fill(0);
  const sampleKeys = scratchSampleKeys;
  for (let i = 0; i < count; i++) {
    const ql = clampInt(Math.floor(L[i]! / 0.04), 0, 25);
    const qa = clampInt(Math.floor((a[i]! + 0.40) / 0.025), 0, 32);
    const qb = clampInt(Math.floor((b[i]! + 0.40) / 0.025), 0, 32);
    const key = (ql * 33 + qa) * 33 + qb;
    sampleKeys[i] = key;
    counts[key]!++;
  }

  let sum = 0;
  for (let i = 0; i < count; i++) {
    const w = Math.pow(counts[sampleKeys[i]!]!, -0.75);
    weights[i] = w;
    sum += w;
  }

  if (normalizeSum) {
    const inv = 1.0 / Math.max(sum, 1e-12);
    for (let i = 0; i < count; i++) {
      weights[i]! *= inv;
    }
  }
  return weights;
}

const IDENTITY_3X3 = [1, 0, 0, 0, 1, 0, 0, 0, 1] as const;
const REC2020_TO_709 = [
  1.6604910021084343, -0.5876411387885495, -0.07284986331988486,
  -0.1245504745215906, 1.13289989712596, -0.00834942260436947,
  -0.01815076335490522, -0.10057889800800743, 1.118729661362913,
] as const;
const SGAMUT3_TO_709 = [
  1.6269474097290801, -0.5401385388696353, -0.0868088708594452,
  -0.17851552711487662, 1.4179409274640795, -0.23942540034920257,
  -0.04443611500929753, -0.1959199661720156, 1.240356081181313,
] as const;
const VGAMUT_TO_709 = [
  1.806576, -0.695697, -0.110879,
  -0.170090, 1.305955, -0.135865,
  -0.025206, -0.154468, 1.179674,
] as const;
const CGAMUT_TO_709 = [
  1.9238612958985506, -0.7987606632484301, -0.12510063265012045,
  -0.2043108482009208, 1.4958985097509383, -0.29158766155001764,
  -0.02368502097034144, -0.4201270110302702, 1.4438120320006116,
] as const;

function rec709ToWorkingSrgb(v: number): number {
  const lin = v < 0.0812428582986315
    ? v / 4.5
    : Math.pow((Math.max(v, 0.0812428582986315) + 0.09929682680944) / 1.09929682680944, 1.0 / 0.45);
  return clamp01(linearToSrgb(clamp01(lin)));
}

function rawIdtChannel(v: number, idt: IdtName | 'srgb-ref'): number {
  if (idt === 'srgb' || idt === 'srgb-ref') return srgbToLinear(clamp01(v));
  if (idt === 'rec709') {
    return srgbToLinear(rec709ToWorkingSrgb(v));
  }
  if (idt === 'apple-log') {
    const r0 = -0.05641088;
    const pt = 47.28711236 * (0.01 - r0) * (0.01 - r0);
    if (v >= pt) return Math.pow(2.0, (v - 0.69336945) / 0.08550479) - 0.00964052;
    return v > 0.0 ? Math.sqrt(Math.max(v, 0.0) / 47.28711236) + r0 : r0;
  }
  if (idt === 'slog3') {
    const t = v * 1023.0;
    const cut = 171.2102946929;
    return t >= cut
      ? Math.pow(10.0, (t - 420.0) / 261.5) * 0.19 - 0.01
      : ((t - 95.0) * 0.01125) / (cut - 95.0);
  }
  if (idt === 'vlog') {
    return v >= 0.181 ? Math.pow(10.0, (v - 0.598206) / 0.241514) - 0.00873 : (v - 0.125) / 5.6;
  }
  // clog3
  if (v >= 0.15277891) return (Math.pow(10.0, (v - 0.12240537) / 0.36726845) - 1.0) / 14.98325;
  if (v <= 0.09746547) return -(Math.pow(10.0, (0.12783901 - v) / 0.36726845) - 1.0) / 14.98325;
  return (v - 0.12512219) / 1.9754798;
}

const BYTE_IDT_TABLES = new Map<string, Float64Array>();
function getByteIdtTable(idt: IdtName | 'srgb-ref'): Float64Array {
  let tbl = BYTE_IDT_TABLES.get(idt);
  if (!tbl) {
    tbl = new Float64Array(256);
    for (let i = 0; i < 256; i++) {
      tbl[i] = rawIdtChannel(i / 255, idt);
    }
    BYTE_IDT_TABLES.set(idt, tbl);
  }
  return tbl;
}

// Keep unused reference silenced cleanly if needed — use direct helpers
void rec709ToLinear; void appleLogToLinear; void slog3ToLinear; void vlogToLinear; void clog3ToLinear;

function sampleSingleImageInto(
  img: ImageDataLike,
  maxSamples: number,
  idt: IdtName | 'srgb-ref',
  normalizeWeights: boolean,
  L: Float64Array,
  a: Float64Array,
  b: Float64Array,
  weights: Float64Array,
  offset = 0
): number {
  const n = img.width * img.height;
  const count = n <= maxSamples ? n : maxSamples;
  const data = img.data;
  const isFloat = data instanceof Float32Array;
  const byteTbl = getByteIdtTable(idt);
  const useLinspace = n > maxSamples;
  const step = useLinspace ? (n - 1) / (maxSamples - 1) : 1;

  if (idt === 'srgb' || idt === 'srgb-ref' || idt === 'rec709') {
    for (let i = 0; i < count; i++) {
      const p = useLinspace ? (i === count - 1 ? n - 1 : Math.floor(i * step)) : i;
      const idx = p * 4;
      const lr = isFloat ? rawIdtChannel(data[idx]!, idt) : byteTbl[data[idx]!]!;
      const lg = isFloat ? rawIdtChannel(data[idx + 1]!, idt) : byteTbl[data[idx + 1]!]!;
      const lb = isFloat ? rawIdtChannel(data[idx + 2]!, idt) : byteTbl[data[idx + 2]!]!;

      const l_ = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
      const m_ = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
      const s_ = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
      const dst = offset + i;
      L[dst] = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
      a[dst] = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
      b[dst] = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;
    }
  } else {
    const mat =
      idt === 'apple-log'
        ? REC2020_TO_709
        : idt === 'slog3'
          ? SGAMUT3_TO_709
          : idt === 'vlog'
            ? VGAMUT_TO_709
            : CGAMUT_TO_709;
    const m0 = mat[0], m1 = mat[1], m2 = mat[2];
    const m3 = mat[3], m4 = mat[4], m5 = mat[5];
    const m6 = mat[6], m7 = mat[7], m8 = mat[8];

    for (let i = 0; i < count; i++) {
      const p = useLinspace ? (i === count - 1 ? n - 1 : Math.floor(i * step)) : i;
      const idx = p * 4;
      const rLog = isFloat ? rawIdtChannel(data[idx]!, idt) : byteTbl[data[idx]!]!;
      const gLog = isFloat ? rawIdtChannel(data[idx + 1]!, idt) : byteTbl[data[idx + 1]!]!;
      const bLog = isFloat ? rawIdtChannel(data[idx + 2]!, idt) : byteTbl[data[idx + 2]!]!;

      const r7 = m0 * rLog + m1 * gLog + m2 * bLog;
      const g7 = m3 * rLog + m4 * gLog + m5 * bLog;
      const b7 = m6 * rLog + m7 * gLog + m8 * bLog;
      const rx = r7 > 0 ? r7 : 0;
      const gx = g7 > 0 ? g7 : 0;
      const bx = b7 > 0 ? b7 : 0;
      const lr = clamp01((rx * (1.0 + 0.12 * rx)) / (rx * (0.85 + 0.12 * rx) + 0.95));
      const lg = clamp01((gx * (1.0 + 0.12 * gx)) / (gx * (0.85 + 0.12 * gx) + 0.95));
      const lb = clamp01((bx * (1.0 + 0.12 * bx)) / (bx * (0.85 + 0.12 * bx) + 0.95));

      const l_ = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
      const m_ = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
      const s_ = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
      const dst = offset + i;
      L[dst] = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
      a[dst] = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
      b[dst] = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;
    }
  }

  computeDensityWeightsInto(
    offset === 0 ? L : L.subarray(offset, offset + count),
    offset === 0 ? a : a.subarray(offset, offset + count),
    offset === 0 ? b : b.subarray(offset, offset + count),
    count,
    normalizeWeights,
    offset === 0 ? weights : weights.subarray(offset, offset + count)
  );
  return count;
}

function sampleSources(sources: ImageDataLike[], idt: IdtName, maxSamples: number): OklabSample {
  const perSrcBudget = Math.max(2048, Math.floor(maxSamples / sources.length));
  const maxTotal = perSrcBudget * sources.length;
  ensureSrcScratch(maxTotal);
  let totalCount = 0;
  for (const src of sources) {
    totalCount += sampleSingleImageInto(
      src,
      perSrcBudget,
      idt,
      true,
      scratchSrcL,
      scratchSrcA,
      scratchSrcB,
      scratchSrcW,
      totalCount
    );
  }
  return {
    count: totalCount,
    L: scratchSrcL,
    a: scratchSrcA,
    b: scratchSrcB,
    weights: scratchSrcW,
  };
}

function sampleReferences(refs: ImageDataLike[], maxSamples: number): OklabSample {
  const perRefBudget = Math.max(2048, Math.floor(maxSamples / refs.length));
  const maxTotal = perRefBudget * refs.length;
  ensureRefScratch(maxTotal);
  let totalCount = 0;
  for (const ref of refs) {
    totalCount += sampleSingleImageInto(
      ref,
      perRefBudget,
      'srgb-ref',
      true,
      scratchRefL,
      scratchRefA,
      scratchRefB,
      scratchRefW,
      totalCount
    );
  }
  return {
    count: totalCount,
    L: scratchRefL,
    a: scratchRefA,
    b: scratchRefB,
    weights: scratchRefW,
  };
}

function sortPairsRange(v: Float64Array, w: Float64Array, lo: number, hi: number): void {
  while (lo < hi) {
    if (hi - lo <= 16) {
      for (let i = lo + 1; i <= hi; i++) {
        const kv = v[i]!;
        const kw = w[i]!;
        let j = i - 1;
        while (j >= lo && v[j]! > kv) {
          v[j + 1] = v[j]!;
          w[j + 1] = w[j]!;
          j--;
        }
        v[j + 1] = kv;
        w[j + 1] = kw;
      }
      return;
    }

    const mid = (lo + hi) >> 1;
    if (v[mid]! < v[lo]!) {
      const tv = v[lo]!; v[lo] = v[mid]!; v[mid] = tv;
      const tw = w[lo]!; w[lo] = w[mid]!; w[mid] = tw;
    }
    if (v[hi]! < v[lo]!) {
      const tv = v[lo]!; v[lo] = v[hi]!; v[hi] = tv;
      const tw = w[lo]!; w[lo] = w[hi]!; w[hi] = tw;
    }
    if (v[hi]! < v[mid]!) {
      const tv = v[mid]!; v[mid] = v[hi]!; v[hi] = tv;
      const tw = w[mid]!; w[mid] = w[hi]!; w[hi] = tw;
    }
    const pivot = v[mid]!;

    let i = lo;
    let j = hi;
    while (i <= j) {
      while (v[i]! < pivot) i++;
      while (v[j]! > pivot) j--;
      if (i <= j) {
        const tv = v[i]!; v[i] = v[j]!; v[j] = tv;
        const tw = w[i]!; w[i] = w[j]!; w[j] = tw;
        i++;
        j--;
      }
    }

    if (j - lo < hi - i) {
      if (lo < j) sortPairsRange(v, w, lo, j);
      lo = i;
    } else {
      if (i < hi) sortPairsRange(v, w, i, hi);
      hi = j;
    }
  }
}

function sortPairs(v: Float64Array, w: Float64Array, n: number): void {
  if (n > 1) sortPairsRange(v, w, 0, n - 1);
}

function computeWeightedQuantiles(
  values: Float64Array,
  weights: Float64Array,
  n: number,
  probs: readonly number[]
): Float64Array {
  const out = new Float64Array(probs.length);
  if (n === 0) return out;

  ensureSampleScratch(n);
  const v = scratchSortV;
  const w = scratchSortW;
  v.set(values.subarray(0, n));
  w.set(weights.subarray(0, n));
  sortPairs(v, w, n);

  if (n === 1) {
    out.fill(v[0]!);
    return out;
  }

  const cdf = scratchCdf;
  let cum = 0;
  for (let i = 0; i < n; i++) {
    cum += w[i]!;
  }

  if (cum <= 0) {
    for (let k = 0; k < probs.length; k++) {
      const pos = probs[k]! * (n - 1);
      const lo = Math.floor(pos);
      const hi = Math.min(n - 1, lo + 1);
      const frac = pos - lo;
      out[k] = v[lo]! + frac * (v[hi]! - v[lo]!);
    }
    return out;
  }

  let running = 0;
  const invTotal = 1.0 / cum;
  for (let i = 0; i < n; i++) {
    const wi = w[i]!;
    running += wi;
    cdf[i] = (running - 0.5 * wi) * invTotal;
  }

  for (let k = 0; k < probs.length; k++) {
    const p = probs[k]!;
    if (p <= cdf[0]!) {
      out[k] = v[0]!;
      continue;
    }
    if (p >= cdf[n - 1]!) {
      out[k] = v[n - 1]!;
      continue;
    }
    let lo = 0;
    let hi = n - 1;
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1;
      if (cdf[mid]! <= p) lo = mid;
      else hi = mid;
    }
    const c0 = cdf[lo]!;
    const c1 = cdf[hi]!;
    if (c1 <= c0) {
      out[k] = v[lo]!;
    } else {
      out[k] = v[lo]! + ((p - c0) / (c1 - c0)) * (v[hi]! - v[lo]!);
    }
  }
  return out;
}

function buildPchip(xs: Float64Array, ys: Float64Array): PchipSpline {
  const n = xs.length;
  const ms = new Float64Array(n);
  const hs = new Float64Array(n - 1);
  const deltas = new Float64Array(n - 1);

  for (let k = 0; k < n - 1; k++) {
    hs[k] = xs[k + 1]! - xs[k]!;
    deltas[k] = (ys[k + 1]! - ys[k]!) / hs[k]!;
  }

  for (let k = 1; k < n - 1; k++) {
    const d0 = deltas[k - 1]!;
    const d1 = deltas[k]!;
    if (d0 === 0 || d1 === 0 || (d0 > 0) !== (d1 > 0)) {
      ms[k] = 0;
    } else {
      const w1 = 2 * hs[k]! + hs[k - 1]!;
      const w2 = hs[k]! + 2 * hs[k - 1]!;
      ms[k] = (w1 + w2) / (w1 / d0 + w2 / d1);
    }
  }

  if (n === 2) {
    ms[0] = deltas[0]!;
    ms[1] = deltas[0]!;
  } else {
    let m0 =
      ((2 * hs[0]! + hs[1]!) * deltas[0]! - hs[0]! * deltas[1]!) /
      (hs[0]! + hs[1]!);
    if ((m0 > 0) !== (deltas[0]! > 0) && m0 !== 0) m0 = 0;
    if (deltas[0] === 0) m0 = 0;
    else if (Math.abs(m0) > 3 * Math.abs(deltas[0]!)) m0 = 3 * deltas[0]!;
    ms[0] = m0;

    const last = n - 1;
    let mn =
      ((2 * hs[last - 1]! + hs[last - 2]!) * deltas[last - 1]! -
        hs[last - 1]! * deltas[last - 2]!) /
      (hs[last - 1]! + hs[last - 2]!);
    if ((mn > 0) !== (deltas[last - 1]! > 0) && mn !== 0) mn = 0;
    if (deltas[last - 1] === 0) mn = 0;
    else if (Math.abs(mn) > 3 * Math.abs(deltas[last - 1]!)) {
      mn = 3 * deltas[last - 1]!;
    }
    ms[last] = mn;
  }

  return { xs, ys, ms };
}

function evalPchip(spline: PchipSpline, x: number): number {
  const { xs, ys, ms } = spline;
  const n = xs.length;
  let lo = 0;
  if (x <= xs[0]!) {
    lo = 0;
  } else if (x >= xs[n - 1]!) {
    lo = n - 2;
  } else {
    let hi = n - 2;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (x < xs[mid]!) hi = mid - 1;
      else if (x > xs[mid + 1]!) lo = mid + 1;
      else {
        lo = mid;
        break;
      }
    }
  }

  const h = xs[lo + 1]! - xs[lo]!;
  const t = (x - xs[lo]!) / h;
  const t2 = t * t;
  const t3 = t2 * t;
  const h00 = 2 * t3 - 3 * t2 + 1;
  const h10 = t3 - 2 * t2 + t;
  const h01 = -2 * t3 + 3 * t2;
  const h11 = t3 - t2;

  return h00 * ys[lo]! + h10 * h * ms[lo]! + h01 * ys[lo + 1]! + h11 * h * ms[lo + 1]!;
}

function fitLuminanceCurve(
  lSrc: Float64Array,
  wSrc: Float64Array,
  nSrc: number,
  lRef: Float64Array,
  wRef: Float64Array,
  nRef: number,
  shadowLift: number,
  highlightRolloff: number
): PchipSpline {
  const qSrc = computeWeightedQuantiles(lSrc, wSrc, nSrc, QUANTILES);
  const qRef = computeWeightedQuantiles(lRef, wRef, nRef, QUANTILES);

  const qLen = QUANTILES.length;
  const medSrc = qSrc[6]!;
  const medRef = qRef[6]!;
  const spreadSrc = Math.max(qSrc[qLen - 1]! - qSrc[0]!, 0.05);
  const spreadRef = Math.max(qRef[qLen - 1]! - qRef[0]!, 0.05);
  const scale = clamp(spreadRef / spreadSrc, 0.55, 1.65);
  const dMed = clamp(medRef - medSrc, -0.18, 0.18);
  const ratioScale = clamp((scale * spreadSrc) / spreadRef, 0.5, 1.0);

  const qTarget = new Float64Array(qLen);
  for (let i = 0; i < qLen; i++) {
    const rawTarget = medSrc + dMed + (qRef[i]! - medRef) * ratioScale;
    qTarget[i] = 0.78 * qRef[i]! + 0.22 * rawTarget;
  }

  const xKnots = new Float64Array(qLen + 2);
  const yKnots = new Float64Array(qLen + 2);
  xKnots[0] = 0.0;
  xKnots[qLen + 1] = 1.0;
  for (let i = 0; i < qLen; i++) {
    xKnots[i + 1] = clamp(qSrc[i]!, 1e-4, 1.0 - 1e-4);
  }
  for (let i = 1; i < xKnots.length; i++) {
    if (xKnots[i]! <= xKnots[i - 1]! + 1e-4) {
      xKnots[i] = Math.min(1.0, xKnots[i - 1]! + 1e-4);
    }
  }
  if (xKnots[qLen]! >= 1.0 - 1e-4) {
    for (let i = 0; i < qLen; i++) {
      xKnots[i + 1] = 0.01 + (0.98 * i) / (qLen - 1);
    }
  }

  const slopeLow = clamp(
    (qTarget[1]! - qTarget[0]!) / Math.max(xKnots[2]! - xKnots[1]!, 1e-3),
    0.2,
    1.8
  );
  const slopeHigh = clamp(
    (qTarget[qLen - 1]! - qTarget[qLen - 2]!) /
      Math.max(xKnots[qLen]! - xKnots[qLen - 1]!, 1e-3),
    0.2,
    1.8
  );
  yKnots[0] = clamp(qTarget[0]! - slopeLow * xKnots[1]!, 0.0, 0.25);
  yKnots[qLen + 1] = clamp(
    qTarget[qLen - 1]! + slopeHigh * (1.0 - xKnots[qLen]!),
    0.75,
    1.0
  );
  for (let i = 0; i < qLen; i++) {
    yKnots[i + 1] = clamp01(qTarget[i]!);
  }

  for (let i = 1; i < yKnots.length; i++) {
    const dx = xKnots[i]! - xKnots[i - 1]!;
    const minY = yKnots[i - 1]! + 0.05 * dx;
    const maxY = yKnots[i - 1]! + 2.8 * dx;
    yKnots[i] = clamp(yKnots[i]!, minY, maxY);
  }
  for (let i = 0; i < yKnots.length; i++) {
    yKnots[i] = clamp01(yKnots[i]!);
  }
  for (let i = 1; i < yKnots.length; i++) {
    if (yKnots[i]! <= yKnots[i - 1]!) {
      yKnots[i] = Math.min(1.0, yKnots[i - 1]! + 1e-5);
    }
  }

  const pchip = buildPchip(xKnots, yKnots);
  if (shadowLift === 0.0 && highlightRolloff === 0.0) {
    return pchip;
  }

  const gridLen = 65;
  const lGrid = new Float64Array(gridLen);
  const lOut = new Float64Array(gridLen);
  let prev = 0;
  for (let i = 0; i < gridLen; i++) {
    const x = i / (gridLen - 1);
    lGrid[i] = x;
    let y = clamp01(evalPchip(pchip, x));
    if (shadowLift !== 0.0) {
      const om = 1.0 - y;
      y += shadowLift * 0.14 * om * om * om;
    }
    if (highlightRolloff !== 0.0) {
      y -= highlightRolloff * 0.14 * y * y * y;
    }
    y = clamp01(y);
    if (i === 0) prev = y;
    else prev = Math.max(prev, y);
    lOut[i] = prev;
  }
  return buildPchip(lGrid, lOut);
}

interface HuberResult {
  muA: number;
  muB: number;
  c00: number;
  c01: number;
  c11: number;
  neutralCoverage: number;
}

function huberMeanCovBand(
  L: Float64Array,
  a: Float64Array,
  b: Float64Array,
  baseWeights: Float64Array,
  neuDecay: Float64Array,
  bandWeightsScratch: Float64Array,
  ck: number,
  inv2Sigma2: number,
  n: number
): HuberResult {
  let wSum = 0;
  let wNeuSum = 0;
  let neuA = 0;
  let neuB = 0;
  let sumA = 0;
  let sumB = 0;

  for (let i = 0; i < n; i++) {
    const d = L[i]! - ck;
    const w = baseWeights[i]! * Math.exp(-d * d * inv2Sigma2);
    bandWeightsScratch[i] = w;
    const av = a[i]!;
    const bv = b[i]!;
    wSum += w;
    sumA += w * av;
    sumB += w * bv;

    const wn = w * neuDecay[i]!;
    wNeuSum += wn;
    neuA += wn * av;
    neuB += wn * bv;
  }

  if (wSum <= 1e-12) {
    return { muA: 0, muB: 0, c00: 1e-3, c01: 0, c11: 1e-3, neutralCoverage: 0 };
  }

  const neutralCoverage = wNeuSum / wSum;
  const muA = wNeuSum > 1e-12 ? neuA / wNeuSum : sumA / wSum;
  const muB = wNeuSum > 1e-12 ? neuB / wNeuSum : sumB / wSum;

  let muCovA = sumA / wSum;
  let muCovB = sumB / wSum;
  const invDeltaSq = 1.0 / (0.12 * 0.12);

  for (let iter = 0; iter < 3; iter++) {
    let rwSum = 0;
    let raSum = 0;
    let rbSum = 0;
    for (let i = 0; i < n; i++) {
      const av = a[i]!;
      const bv = b[i]!;
      const da = av - muCovA;
      const db = bv - muCovB;
      const wRob =
        bandWeightsScratch[i]! / Math.sqrt(1.0 + (da * da + db * db) * invDeltaSq);
      rwSum += wRob;
      raSum += wRob * av;
      rbSum += wRob * bv;
    }
    if (rwSum > 1e-12) {
      muCovA = raSum / rwSum;
      muCovB = rbSum / rwSum;
    }
  }

  let rwSum = 0;
  let c00 = 0;
  let c01 = 0;
  let c11 = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i]! - muCovA;
    const db = b[i]! - muCovB;
    const wRob =
      bandWeightsScratch[i]! / Math.sqrt(1.0 + (da * da + db * db) * invDeltaSq);
    rwSum += wRob;
    c00 += wRob * da * da;
    c01 += wRob * da * db;
    c11 += wRob * db * db;
  }

  if (rwSum <= 1e-12) {
    return { muA, muB, c00: 1e-3, c01: 0, c11: 1e-3, neutralCoverage };
  }

  const invRw = 1.0 / rwSum;
  return {
    muA,
    muB,
    c00: c00 * invRw,
    c01: c01 * invRw,
    c11: c11 * invRw,
    neutralCoverage,
  };
}

interface Sym2x2Eig {
  l1: number;
  l2: number;
  vx1: number;
  vy1: number;
  vx2: number;
  vy2: number;
}

function eigSym2x2(a: number, b: number, d: number): Sym2x2Eig {
  const trace = a + d;
  const diff = a - d;
  const disc = Math.sqrt(diff * diff + 4 * b * b);
  const l1 = 0.5 * (trace + disc);
  const l2 = 0.5 * (trace - disc);

  let vx1 = 1;
  let vy1 = 0;
  if (Math.abs(b) > 1e-12) {
    vx1 = l1 - d;
    vy1 = b;
    const len = Math.sqrt(vx1 * vx1 + vy1 * vy1);
    vx1 /= len;
    vy1 /= len;
  } else if (a < d) {
    vx1 = 0;
    vy1 = 1;
  }

  return { l1, l2, vx1, vy1, vx2: -vy1, vy2: vx1 };
}

function sym2x2FromEig(e: Sym2x2Eig, f1: number, f2: number): [number, number, number] {
  return [
    f1 * e.vx1 * e.vx1 + f2 * e.vx2 * e.vx2,
    f1 * e.vx1 * e.vy1 + f2 * e.vx2 * e.vy2,
    f1 * e.vy1 * e.vy1 + f2 * e.vy2 * e.vy2,
  ];
}

function mul2x2(
  a00: number,
  a01: number,
  a10: number,
  a11: number,
  b00: number,
  b01: number,
  b10: number,
  b11: number
): [number, number, number, number] {
  return [
    a00 * b00 + a01 * b10,
    a00 * b01 + a01 * b11,
    a10 * b00 + a11 * b10,
    a10 * b01 + a11 * b11,
  ];
}

function regularizedMkl2x2(
  srcC00: number,
  srcC01: number,
  srcC11: number,
  refC00: number,
  refC01: number,
  refC11: number,
  reg = 2e-4
): [number, number, number, number] {
  const eigS = eigSym2x2(srcC00 + reg, srcC01, srcC11 + reg);
  const eigS1 = Math.max(eigS.l1, 1e-8);
  const eigS2 = Math.max(eigS.l2, 1e-8);
  const sHalf = sym2x2FromEig(eigS, Math.sqrt(eigS1), Math.sqrt(eigS2));
  const sInvHalf = sym2x2FromEig(eigS, 1.0 / Math.sqrt(eigS1), 1.0 / Math.sqrt(eigS2));

  const r00 = refC00 + reg;
  const r01 = refC01;
  const r11 = refC11 + reg;

  const [sr00, sr01, sr10, sr11] = mul2x2(
    sHalf[0], sHalf[1], sHalf[1], sHalf[2],
    r00, r01, r01, r11
  );
  const [m00, m01, , m11] = mul2x2(
    sr00, sr01, sr10, sr11,
    sHalf[0], sHalf[1], sHalf[1], sHalf[2]
  );

  const eigM = eigSym2x2(m00, m01, m11);
  const midHalf = sym2x2FromEig(
    eigM,
    Math.sqrt(Math.max(eigM.l1, 1e-10)),
    Math.sqrt(Math.max(eigM.l2, 1e-10))
  );

  const [a00, a01, a10, a11] = mul2x2(
    sInvHalf[0], sInvHalf[1], sInvHalf[1], sInvHalf[2],
    midHalf[0], midHalf[1], midHalf[1], midHalf[2]
  );
  const [mkl00, mkl01, mkl10, mkl11] = mul2x2(
    a00, a01, a10, a11,
    sInvHalf[0], sInvHalf[1], sInvHalf[1], sInvHalf[2]
  );

  const eigSRaw = eigSym2x2(srcC00, srcC01, srcC11);
  const eigRRaw = eigSym2x2(refC00, refC01, refC11);
  const es1 = Math.max(eigSRaw.l1, 0);
  const es2 = Math.max(eigSRaw.l2, 0);
  const er1 = Math.max(eigRRaw.l1, 0);
  const er2 = Math.max(eigRRaw.l2, 0);

  const sumS = es1 + es2;
  const sumR = er1 + er2;
  const effS = (sumS * sumS) / Math.max(es1 * es1 + es2 * es2, 1e-12);
  const effR = (sumR * sumR) / Math.max(er1 * er1 + er2 * er2, 1e-12);

  const alphaRank =
    clamp01((effS - 1.05) / 0.45) * clamp01((effR - 1.05) / 0.45);
  const sIso = clamp(
    Math.sqrt((sumR + 2.0 * reg) / (sumS + 2.0 * reg)),
    0.55,
    1.45
  );

  const t00 = alphaRank * mkl00 + (1.0 - alphaRank) * sIso;
  const t01 = alphaRank * mkl01;
  const t10 = alphaRank * mkl10;
  const t11 = alphaRank * mkl11 + (1.0 - alphaRank) * sIso;

  // SVD of 2x2 matrix T via T^T T
  const b00 = t00 * t00 + t10 * t10;
  const b01 = t00 * t01 + t10 * t11;
  const b11 = t01 * t01 + t11 * t11;
  const eigB = eigSym2x2(b00, b01, b11);
  const sv1 = Math.sqrt(Math.max(eigB.l1, 0));
  const sv2 = Math.sqrt(Math.max(eigB.l2, 0));

  const tv1x = t00 * eigB.vx1 + t01 * eigB.vy1;
  const tv1y = t10 * eigB.vx1 + t11 * eigB.vy1;
  const u1x = sv1 > 1e-12 ? tv1x / sv1 : 1;
  const u1y = sv1 > 1e-12 ? tv1y / sv1 : 0;

  let u2x: number;
  let u2y: number;
  if (sv2 > 1e-12) {
    u2x = (t00 * eigB.vx2 + t01 * eigB.vy2) / sv2;
    u2y = (t10 * eigB.vx2 + t11 * eigB.vy2) / sv2;
  } else {
    u2x = -u1y;
    u2y = u1x;
  }

  const c1 = clamp(sv1, 0.35, 1.95);
  const c2 = clamp(sv2, 0.35, 1.95);

  return [
    c1 * u1x * eigB.vx1 + c2 * u2x * eigB.vx2,
    c1 * u1x * eigB.vy1 + c2 * u2x * eigB.vy2,
    c1 * u1y * eigB.vx1 + c2 * u2y * eigB.vx2,
    c1 * u1y * eigB.vy1 + c2 * u2y * eigB.vy2,
  ];
}

function smoothstepBand(x: number, low: number, high: number, edge: number): number {
  const denom = Math.max(2.0 * edge, 1e-6);
  const tLo = clamp01((x - (low - edge)) / denom);
  const tHi = clamp01((high + edge - x) / denom);
  return tLo * tLo * (3.0 - 2.0 * tLo) * tHi * tHi * (3.0 - 2.0 * tHi);
}

function posMod360(deg: number): number {
  const m = deg % 360.0;
  return m < 0 ? m + 360.0 : m;
}

function applySkinGuard(
  lIn: number,
  aIn: number,
  bIn: number,
  aOut: number,
  bOut: number,
  skinProtect: number,
  abOut: [number, number]
): void {
  if (skinProtect <= 0.0 || lIn <= 0.15 || lIn >= 0.95) {
    abOut[0] = aOut;
    abOut[1] = bOut;
    return;
  }

  const cInSq = aIn * aIn + bIn * bIn;
  if (cInSq >= 0.215 * 0.215) {
    abOut[0] = aOut;
    abOut[1] = bOut;
    return;
  }

  const mL = smoothstepBand(lIn, 0.25, 0.85, 0.10);
  const cIn = Math.sqrt(cInSq);
  const mC = smoothstepBand(cIn, 0.015, 0.19, 0.025);
  if (mL <= 0.0 || mC <= 0.0) {
    abOut[0] = aOut;
    abOut[1] = bOut;
    return;
  }

  const hIn = posMod360((Math.atan2(bIn, aIn) * 180.0) / Math.PI);
  const dhLocus = posMod360(hIn - 45.0 + 180.0) - 180.0;
  const mH = Math.exp(-0.5 * (dhLocus / 22.0) * (dhLocus / 22.0));
  const sp = 1.0 - Math.pow(1.0 - skinProtect, 1.8);
  const wProt = clamp01(sp * mL * mC * mH);
  if (wProt <= 1e-9) {
    abOut[0] = aOut;
    abOut[1] = bOut;
    return;
  }

  const cOut = Math.sqrt(aOut * aOut + bOut * bOut);
  const hOut = posMod360((Math.atan2(bOut, aOut) * 180.0) / Math.PI);

  const dh = posMod360(hOut - hIn + 180.0) - 180.0;
  const dhSafe = dh * (1.0 - 0.94 * wProt);

  const cRatio = cIn > 1e-6 ? cOut / Math.max(cIn, 1e-6) : 1.0;
  const cRatioSafe = 1.0 + (cRatio - 1.0) * (1.0 - 0.55 * wProt);
  const cSafe = cIn > 1e-6 ? cIn * cRatioSafe : cOut;

  const hSafeRad = ((hIn + dhSafe) * Math.PI) / 180.0;
  abOut[0] = cSafe * Math.cos(hSafeRad);
  abOut[1] = cSafe * Math.sin(hSafeRad);
}

function laplacianSmooth3d(data: Float64Array, size: number, smoothness: number): void {
  if (smoothness <= 0.0 || size < 3) return;
  const len = size * size * size * 3;
  ensureCubeScratch(size);
  const nIters = Math.max(1, Math.round(smoothness * 10.0));
  const alpha = 0.14 * Math.min(smoothness, 1.0);
  const anchor = 0.06;
  const orig = scratchCubeOrig;
  const cur = scratchCubeCur;
  orig.set(data.subarray(0, len));
  cur.set(data.subarray(0, len));
  const strideG = size * 3;
  const strideB = size * size * 3;
  const inv6 = 1.0 / 6.0;

  for (let iter = 0; iter < nIters; iter++) {
    for (let b = 1; b < size - 1; b++) {
      for (let g = 1; g < size - 1; g++) {
        const rowBase = b * strideB + g * strideG;
        for (let r = 1; r < size - 1; r++) {
          const p0 = rowBase + r * 3;
          for (let c = 0; c < 3; c++) {
            const p = p0 + c;
            const avg =
              (cur[p - 3]! +
                cur[p + 3]! +
                cur[p - strideG]! +
                cur[p + strideG]! +
                cur[p - strideB]! +
                cur[p + strideB]!) *
              inv6;
            const curV = cur[p]!;
            data[p] =
              (1.0 - alpha) * curV + alpha * avg + anchor * (orig[p]! - curV);
          }
        }
      }
    }
    cur.set(data.subarray(0, len));
  }

  for (let i = 0; i < len; i++) {
    data[i] = clamp01(data[i]!);
  }
}

function pava1d(
  line: Float64Array,
  n: number,
  blockVal: Float64Array,
  blockCount: Int32Array
): void {
  let numBlocks = 0;
  for (let i = 0; i < n; i++) {
    blockVal[numBlocks] = line[i]!;
    blockCount[numBlocks] = 1;
    numBlocks++;
    while (numBlocks >= 2 && blockVal[numBlocks - 2]! > blockVal[numBlocks - 1]!) {
      const c1 = blockCount[numBlocks - 2]!;
      const c2 = blockCount[numBlocks - 1]!;
      blockVal[numBlocks - 2] =
        (blockVal[numBlocks - 2]! * c1 + blockVal[numBlocks - 1]! * c2) / (c1 + c2);
      blockCount[numBlocks - 2] = c1 + c2;
      numBlocks--;
    }
  }

  let outIdx = 0;
  for (let b = 0; b < numBlocks; b++) {
    const val = blockVal[b]!;
    const cnt = blockCount[b]!;
    for (let k = 0; k < cnt; k++) {
      line[outIdx++] = val;
    }
  }
}

function waterfillLine(
  cols: Float64Array,
  targetLuma: Float64Array,
  n: number
): void {
  for (let iter = 0; iter < 4; iter++) {
    let maxAbsErr = 0;
    for (let i = 0; i < n; i++) {
      const p = i * 3;
      const curLuma =
        0.2126 * cols[p]! + 0.7152 * cols[p + 1]! + 0.0722 * cols[p + 2]!;
      const err = targetLuma[i]! - curLuma;
      const absErr = Math.abs(err);
      if (absErr > maxAbsErr) maxAbsErr = absErr;

      let denom = 0;
      let f0 = 0;
      let f1 = 0;
      let f2 = 0;
      if (err >= 0.0) {
        if (cols[p]! < 1.0 - 1e-12) { f0 = 1; denom += 0.2126; }
        if (cols[p + 1]! < 1.0 - 1e-12) { f1 = 1; denom += 0.7152; }
        if (cols[p + 2]! < 1.0 - 1e-12) { f2 = 1; denom += 0.0722; }
      } else {
        if (cols[p]! > 1e-12) { f0 = 1; denom += 0.2126; }
        if (cols[p + 1]! > 1e-12) { f1 = 1; denom += 0.7152; }
        if (cols[p + 2]! > 1e-12) { f2 = 1; denom += 0.0722; }
      }

      if (denom > 1e-9) {
        const step = err / denom;
        if (f0) cols[p] = clamp01(cols[p]! + step);
        if (f1) cols[p + 1] = clamp01(cols[p + 1]! + step);
        if (f2) cols[p + 2] = clamp01(cols[p + 2]! + step);
      }
    }
    if (maxAbsErr < 1e-10) break;
  }
}

function projectMonotoneLattice(data: Float64Array, size: number, maxIters = 32): void {
  const n3 = size * size * size;
  const len = n3 * 3;
  ensureCubeScratch(size);
  for (let i = 0; i < len; i++) {
    data[i] = clamp01(data[i]!);
  }

  const tol = -1e-8;
  const line = scratchPavaLine;
  const cols = scratchPavaCols;
  const blockVal = scratchBlockVal;
  const blockCount = scratchBlockCount;
  const luma = scratchLuma;
  const violMask = scratchViolMask;
  const strideG = size * 3;
  const strideB = size * size * 3;
  const planeCells = size * size;

  for (let iter = 0; iter < maxIters; iter++) {
    let changed = false;

    // Self-channel R along axis 0
    for (let b = 0; b < size; b++) {
      for (let g = 0; g < size; g++) {
        const base = b * strideB + g * strideG;
        let viol = false;
        let prev = data[base]!;
        for (let r = 1; r < size; r++) {
          const v = data[base + r * 3]!;
          if (v - prev < tol) {
            viol = true;
            break;
          }
          prev = v;
        }
        if (viol) {
          changed = true;
          for (let r = 0; r < size; r++) line[r] = data[base + r * 3]!;
          pava1d(line, size, blockVal, blockCount);
          for (let r = 0; r < size; r++) data[base + r * 3] = line[r]!;
        }
      }
    }

    // Self-channel G along axis 1
    for (let b = 0; b < size; b++) {
      const planeBase = b * strideB;
      violMask.fill(0, 0, size);
      let anyViol = false;
      for (let g = 1; g < size; g++) {
        const rowBase = planeBase + g * strideG + 1;
        const prevBase = rowBase - strideG;
        for (let r = 0; r < size; r++) {
          if (data[rowBase + r * 3]! - data[prevBase + r * 3]! < tol) {
            violMask[r] = 1;
            anyViol = true;
          }
        }
      }
      if (anyViol) {
        changed = true;
        for (let r = 0; r < size; r++) {
          if (violMask[r]) {
            const colOff = planeBase + r * 3 + 1;
            for (let g = 0; g < size; g++) line[g] = data[colOff + g * strideG]!;
            pava1d(line, size, blockVal, blockCount);
            for (let g = 0; g < size; g++) data[colOff + g * strideG] = line[g]!;
          }
        }
      }
    }

    // Self-channel B along axis 2
    violMask.fill(0, 0, planeCells);
    let anyBViol = false;
    for (let b = 1; b < size; b++) {
      const planeBase = b * strideB + 2;
      const prevPlane = planeBase - strideB;
      for (let g = 0; g < size; g++) {
        const rowOff = g * strideG;
        const mOff = g * size;
        for (let r = 0; r < size; r++) {
          const p = rowOff + r * 3;
          if (data[planeBase + p]! - data[prevPlane + p]! < tol) {
            violMask[mOff + r] = 1;
            anyBViol = true;
          }
        }
      }
    }
    if (anyBViol) {
      changed = true;
      for (let g = 0; g < size; g++) {
        const mOff = g * size;
        for (let r = 0; r < size; r++) {
          if (violMask[mOff + r]) {
            const colOff = g * strideG + r * 3 + 2;
            for (let b = 0; b < size; b++) line[b] = data[colOff + b * strideB]!;
            pava1d(line, size, blockVal, blockCount);
            for (let b = 0; b < size; b++) data[colOff + b * strideB] = line[b]!;
          }
        }
      }
    }

    // Compute luma sequentially once, then update only changed lines
    for (let i = 0; i < n3; i++) {
      const p = i * 3;
      luma[i] = 0.2126 * data[p]! + 0.7152 * data[p + 1]! + 0.0722 * data[p + 2]!;
    }

    // Luma along R (axis 0)
    for (let b = 0; b < size; b++) {
      for (let g = 0; g < size; g++) {
        const lBase = (b * size + g) * size;
        let viol = false;
        let prevL = luma[lBase]!;
        for (let r = 1; r < size; r++) {
          const lv = luma[lBase + r]!;
          if (lv - prevL < tol) {
            viol = true;
            break;
          }
          prevL = lv;
        }
        if (viol) {
          changed = true;
          const base = b * strideB + g * strideG;
          for (let r = 0; r < size; r++) {
            const p = base + r * 3;
            line[r] = luma[lBase + r]!;
            cols[r * 3] = data[p]!;
            cols[r * 3 + 1] = data[p + 1]!;
            cols[r * 3 + 2] = data[p + 2]!;
          }
          pava1d(line, size, blockVal, blockCount);
          waterfillLine(cols, line, size);
          for (let r = 0; r < size; r++) {
            const p = base + r * 3;
            const nr = cols[r * 3]!;
            const ng = cols[r * 3 + 1]!;
            const nb = cols[r * 3 + 2]!;
            data[p] = nr;
            data[p + 1] = ng;
            data[p + 2] = nb;
            luma[lBase + r] = 0.2126 * nr + 0.7152 * ng + 0.0722 * nb;
          }
        }
      }
    }

    // Luma along G (axis 1)
    for (let b = 0; b < size; b++) {
      const lPlane = b * planeCells;
      const planeBase = b * strideB;
      violMask.fill(0, 0, size);
      let anyViol = false;
      for (let g = 1; g < size; g++) {
        const lRow = lPlane + g * size;
        const lPrev = lRow - size;
        for (let r = 0; r < size; r++) {
          if (luma[lRow + r]! - luma[lPrev + r]! < tol) {
            violMask[r] = 1;
            anyViol = true;
          }
        }
      }
      if (anyViol) {
        changed = true;
        for (let r = 0; r < size; r++) {
          if (violMask[r]) {
            const colOff = planeBase + r * 3;
            const lCol = lPlane + r;
            for (let g = 0; g < size; g++) {
              const p = colOff + g * strideG;
              line[g] = luma[lCol + g * size]!;
              cols[g * 3] = data[p]!;
              cols[g * 3 + 1] = data[p + 1]!;
              cols[g * 3 + 2] = data[p + 2]!;
            }
            pava1d(line, size, blockVal, blockCount);
            waterfillLine(cols, line, size);
            for (let g = 0; g < size; g++) {
              const p = colOff + g * strideG;
              const nr = cols[g * 3]!;
              const ng = cols[g * 3 + 1]!;
              const nb = cols[g * 3 + 2]!;
              data[p] = nr;
              data[p + 1] = ng;
              data[p + 2] = nb;
              luma[lCol + g * size] = 0.2126 * nr + 0.7152 * ng + 0.0722 * nb;
            }
          }
        }
      }
    }

    // Luma along B (axis 2)
    violMask.fill(0, 0, planeCells);
    let anyLumaBViol = false;
    for (let b = 1; b < size; b++) {
      const lPlane = b * planeCells;
      const lPrev = lPlane - planeCells;
      for (let i = 0; i < planeCells; i++) {
        if (luma[lPlane + i]! - luma[lPrev + i]! < tol) {
          violMask[i] = 1;
          anyLumaBViol = true;
        }
      }
    }
    if (anyLumaBViol) {
      changed = true;
      for (let g = 0; g < size; g++) {
        const mOff = g * size;
        for (let r = 0; r < size; r++) {
          if (violMask[mOff + r]) {
            const colOff = g * strideG + r * 3;
            const lCol = mOff + r;
            for (let b = 0; b < size; b++) {
              const p = colOff + b * strideB;
              line[b] = luma[lCol + b * planeCells]!;
              cols[b * 3] = data[p]!;
              cols[b * 3 + 1] = data[p + 1]!;
              cols[b * 3 + 2] = data[p + 2]!;
            }
            pava1d(line, size, blockVal, blockCount);
            waterfillLine(cols, line, size);
            for (let b = 0; b < size; b++) {
              const p = colOff + b * strideB;
              data[p] = cols[b * 3]!;
              data[p + 1] = cols[b * 3 + 1]!;
              data[p + 2] = cols[b * 3 + 2]!;
            }
          }
        }
      }
    }

    if (!changed) break;
  }

  for (let i = 0; i < len; i++) {
    data[i] = clamp01(data[i]!);
  }
}

function monotoneSeparableSmooth3d(
  data: Float64Array,
  size: number,
  smoothness: number
): void {
  if (smoothness <= 0.0 || size < 3) return;
  const len = size * size * size * 3;
  ensureCubeScratch(size);
  const nPasses = Math.max(2, Math.round(smoothness * 14.0));
  const w = Math.min(0.24 * Math.min(smoothness / 0.35, 1.0), 0.28);
  const wMid = 1.0 - 2.0 * w;
  const nxt = scratchCubeCur;
  const strideG = size * 3;
  const strideB = size * size * 3;

  for (let pass = 0; pass < nPasses; pass++) {
    // Axis R interior 1..size-2
    nxt.set(data.subarray(0, len));
    for (let b = 0; b < size; b++) {
      for (let g = 0; g < size; g++) {
        const base = b * strideB + g * strideG;
        for (let r = 1; r < size - 1; r++) {
          const p = base + r * 3;
          nxt[p] = w * data[p - 3]! + wMid * data[p]! + w * data[p + 3]!;
          nxt[p + 1] = w * data[p - 2]! + wMid * data[p + 1]! + w * data[p + 4]!;
          nxt[p + 2] = w * data[p - 1]! + wMid * data[p + 2]! + w * data[p + 5]!;
        }
      }
    }
    data.set(nxt.subarray(0, len));

    // Axis G interior 1..size-2
    for (let b = 0; b < size; b++) {
      const planeBase = b * strideB;
      for (let g = 1; g < size - 1; g++) {
        const rowBase = planeBase + g * strideG;
        for (let r = 0; r < size; r++) {
          const p = rowBase + r * 3;
          nxt[p] = w * data[p - strideG]! + wMid * data[p]! + w * data[p + strideG]!;
          nxt[p + 1] = w * data[p - strideG + 1]! + wMid * data[p + 1]! + w * data[p + strideG + 1]!;
          nxt[p + 2] = w * data[p - strideG + 2]! + wMid * data[p + 2]! + w * data[p + strideG + 2]!;
        }
      }
    }
    data.set(nxt.subarray(0, len));

    // Axis B interior 1..size-2
    for (let b = 1; b < size - 1; b++) {
      const planeBase = b * strideB;
      for (let g = 0; g < size; g++) {
        const rowBase = planeBase + g * strideG;
        for (let r = 0; r < size; r++) {
          const p = rowBase + r * 3;
          nxt[p] = w * data[p - strideB]! + wMid * data[p]! + w * data[p + strideB]!;
          nxt[p + 1] = w * data[p - strideB + 1]! + wMid * data[p + 1]! + w * data[p + strideB + 1]!;
          nxt[p + 2] = w * data[p - strideB + 2]! + wMid * data[p + 2]! + w * data[p + strideB + 2]!;
        }
      }
    }
    data.set(nxt.subarray(0, len));
  }
}

function solveLinearSystem(
  aMat: Float64Array,
  bMat: Float64Array,
  n: number,
  cols: number
): Float64Array {
  const a = new Float64Array(aMat);
  const b = new Float64Array(bMat);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    let maxVal = Math.abs(a[col * n + col]!);
    for (let row = col + 1; row < n; row++) {
      const v = Math.abs(a[row * n + col]!);
      if (v > maxVal) {
        maxVal = v;
        pivot = row;
      }
    }
    if (pivot !== col) {
      for (let j = col; j < n; j++) {
        const t = a[col * n + j]!;
        a[col * n + j] = a[pivot * n + j]!;
        a[pivot * n + j] = t;
      }
      for (let c = 0; c < cols; c++) {
        const t = b[col * cols + c]!;
        b[col * cols + c] = b[pivot * cols + c]!;
        b[pivot * cols + c] = t;
      }
    }
    const diag = a[col * n + col]!;
    const invDiag = Math.abs(diag) > 1e-15 ? 1.0 / diag : 0.0;
    for (let j = col; j < n; j++) a[col * n + j]! *= invDiag;
    for (let c = 0; c < cols; c++) b[col * cols + c]! *= invDiag;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = a[row * n + col]!;
      if (factor === 0.0) continue;
      for (let j = col; j < n; j++) {
        a[row * n + j]! -= factor * a[col * n + j]!;
      }
      for (let c = 0; c < cols; c++) {
        b[row * cols + c]! -= factor * b[col * cols + c]!;
      }
    }
  }
  return b;
}

function computeHueSectorStats(
  aArr: Float64Array,
  bArr: Float64Array,
  weights: Float64Array,
  count: number,
  outP: Float64Array,
  outDelta: Float64Array,
  outMeanC: Float64Array
): void {
  const histW = new Float64Array(HUE_BINS);
  const histC = new Float64Array(HUE_BINS);
  const histDt = new Float64Array(HUE_BINS);
  let tot = 0;
  const cGateSq = 0.035 * 0.035;
  const scaleBin = HUE_BINS / (2.0 * Math.PI);

  for (let i = 0; i < count; i++) {
    const av = aArr[i]!;
    const bv = bArr[i]!;
    const cSq = av * av + bv * bv;
    if (cSq <= 1e-12) continue;
    const gc = cSq / (cSq + cGateSq);
    const wc = weights[i]! * gc;
    if (wc <= 1e-12) continue;
    tot += wc;
    const theta = Math.atan2(bv, av);
    const bin = clampInt(Math.floor((theta + Math.PI) * scaleBin), 0, HUE_BINS - 1);
    histW[bin]! += wc;
    histC[bin]! += wc * Math.sqrt(cSq);
    histDt[bin]! += wc * (theta - HUE_BIN_CENTERS[bin]!);
  }

  if (tot <= 1e-12) {
    outP.fill(0);
    outDelta.fill(0);
    outMeanC.fill(0.05);
    return;
  }

  const invTot = 1.0 / tot;
  for (let m = 0; m < HUE_ANCHOR_COUNT; m++) {
    const rowOff = m * HUE_BINS;
    let mm = 0;
    let ang = 0;
    let chr = 0;
    for (let b = 0; b < HUE_BINS; b++) {
      const kmb = HUE_KERNEL_MB[rowOff + b]!;
      const wb = histW[b]!;
      mm += kmb * wb;
      ang += kmb * (HUE_DIFF_MB[rowOff + b]! * wb + histDt[b]!);
      chr += kmb * histC[b]!;
    }
    outP[m] = mm * invTot;
    if (mm > 1e-12) {
      outDelta[m] = ang / mm;
      outMeanC[m] = chr / mm;
    } else {
      outDelta[m] = 0.0;
      outMeanC[m] = 0.05;
    }
  }
}

function fitHueHarmonicResidual(
  srcMklA: Float64Array,
  srcMklB: Float64Array,
  srcW: Float64Array,
  srcCount: number,
  refA: Float64Array,
  refB: Float64Array,
  refW: Float64Array,
  refCount: number,
  cfg: ModeConfig,
  coefHue: Float64Array,
  coefSat: Float64Array
): void {
  const pS = new Float64Array(HUE_ANCHOR_COUNT);
  const deltaS = new Float64Array(HUE_ANCHOR_COUNT);
  const cS = new Float64Array(HUE_ANCHOR_COUNT);
  const pR = new Float64Array(HUE_ANCHOR_COUNT);
  const deltaR = new Float64Array(HUE_ANCHOR_COUNT);
  const cR = new Float64Array(HUE_ANCHOR_COUNT);

  computeHueSectorStats(srcMklA, srcMklB, srcW, srcCount, pS, deltaS, cS);
  computeHueSectorStats(refA, refB, refW, refCount, pR, deltaR, cR);

  const wm = new Float64Array(HUE_ANCHOR_COUNT);
  let wmSum = 0;
  let psSum = 0;
  let prSum = 0;
  for (let m = 0; m < HUE_ANCHOR_COUNT; m++) {
    const ps = pS[m]!;
    const pr = pR[m]!;
    const w = ps < pr ? ps : pr;
    wm[m] = w;
    wmSum += w;
    psSum += ps;
    prSum += pr;
  }

  const denomOv = Math.max(psSum, prSum, 1e-12);
  const overlapRatio = wmSum / denomOv;
  const hueConf = clamp01((overlapRatio - 0.18) / 0.52);

  if (wmSum <= 1e-12 || hueConf <= 1e-9) {
    coefHue.fill(0);
    coefSat.fill(0);
    return;
  }

  const invWm = 1.0 / wmSum;
  const lhs = new Float64Array(36);
  const rhs = new Float64Array(12);
  for (let m = 0; m < HUE_ANCHOR_COUNT; m++) {
    const wNorm = wm[m]! * invWm;
    const yHue = clamp(deltaR[m]! - deltaS[m]!, -cfg.maxHueRad, cfg.maxHueRad);
    const ySat = clamp(Math.log((cR[m]! + 0.025) / (cS[m]! + 0.025)), -cfg.maxLogSat, cfg.maxLogSat);
    const bOff = m * 6;
    for (let i = 0; i < 6; i++) {
      const wbi = wNorm * HUE_BASIS[bOff + i]!;
      for (let j = 0; j < 6; j++) {
        lhs[i * 6 + j]! += wbi * HUE_BASIS[bOff + j]!;
      }
      rhs[i * 2]! += wbi * yHue;
      rhs[i * 2 + 1]! += wbi * ySat;
    }
  }
  for (let i = 0; i < 6; i++) {
    lhs[i * 6 + i]! += HUE_REG_DIAG[i]!;
  }

  const sol = solveLinearSystem(lhs, rhs, 6, 2);
  const scaleH = cfg.hueAlpha * hueConf;
  const scaleS = cfg.satAlpha * hueConf;
  for (let i = 0; i < 6; i++) {
    coefHue[i] = sol[i * 2]! * scaleH;
    coefSat[i] = sol[i * 2 + 1]! * scaleS;
  }
}

function evaluateLattice(
  workData: Float64Array,
  size: number,
  lSpline: PchipSpline,
  trFlat: Float64Array,
  skinProtect: number,
  shadowLift: number,
  highlightRolloff: number,
  inv2Sigma2: number,
  lumaAlpha: number,
  coefHue: Float64Array,
  coefSat: Float64Array,
  maxHueRad: number,
  maxLogSat: number
): void {
  const scale = 1.0 / (size - 1);
  const tickLin = new Float64Array(size);
  for (let i = 0; i < size; i++) {
    tickLin[i] = srgbToLinear(i * scale);
  }

  const abSafe: [number, number] = [0, 0];
  const c2 = BAND_CENTERS[2];

  const t0_sa = trFlat[0]!, t0_sb = trFlat[1]!, t0_ra = trFlat[2]!, t0_rb = trFlat[3]!;
  const t0_00 = trFlat[4]!, t0_01 = trFlat[5]!, t0_10 = trFlat[6]!, t0_11 = trFlat[7]!;
  const t1_sa = trFlat[8]!, t1_sb = trFlat[9]!, t1_ra = trFlat[10]!, t1_rb = trFlat[11]!;
  const t1_00 = trFlat[12]!, t1_01 = trFlat[13]!, t1_10 = trFlat[14]!, t1_11 = trFlat[15]!;
  const t2_sa = trFlat[16]!, t2_sb = trFlat[17]!, t2_ra = trFlat[18]!, t2_rb = trFlat[19]!;
  const t2_00 = trFlat[20]!, t2_01 = trFlat[21]!, t2_10 = trFlat[22]!, t2_11 = trFlat[23]!;
  const t3_sa = trFlat[24]!, t3_sb = trFlat[25]!, t3_ra = trFlat[26]!, t3_rb = trFlat[27]!;
  const t3_00 = trFlat[28]!, t3_01 = trFlat[29]!, t3_10 = trFlat[30]!, t3_11 = trFlat[31]!;
  const t4_sa = trFlat[32]!, t4_sb = trFlat[33]!, t4_ra = trFlat[34]!, t4_rb = trFlat[35]!;
  const t4_00 = trFlat[36]!, t4_01 = trFlat[37]!, t4_10 = trFlat[38]!, t4_11 = trFlat[39]!;

  const ch0 = coefHue[0]!, ch1 = coefHue[1]!, ch2 = coefHue[2]!;
  const ch3 = coefHue[3]!, ch4 = coefHue[4]!, ch5 = coefHue[5]!;
  const cs0 = coefSat[0]!, cs1 = coefSat[1]!, cs2 = coefSat[2]!;
  const cs3 = coefSat[3]!, cs4 = coefSat[4]!, cs5 = coefSat[5]!;
  const hasHueHarmonic =
    ch0 !== 0 || ch1 !== 0 || ch2 !== 0 || ch3 !== 0 || ch4 !== 0 || ch5 !== 0 ||
    cs0 !== 0 || cs1 !== 0 || cs2 !== 0 || cs3 !== 0 || cs4 !== 0 || cs5 !== 0;
  const oneMinusLuma = 1.0 - lumaAlpha;
  const gateSq = 0.03 * 0.03;

  for (let b = 0; b < size; b++) {
    const lb = tickLin[b]!;
    for (let g = 0; g < size; g++) {
      const lg = tickLin[g]!;
      for (let r = 0; r < size; r++) {
        const lr = tickLin[r]!;
        const idx = lutIndex(r, g, b, size);
        const l_ = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
        const m_ = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
        const s_ = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);

        const L0 = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
        const a0 = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
        const b0 = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;
        let L1 = oneMinusLuma * L0 + lumaAlpha * clamp01(evalPchip(lSpline, L0));
        if (shadowLift !== 0.0) {
          const om = 1.0 - L1;
          L1 = clamp01(L1 + shadowLift * 0.14 * om * om * om);
        }
        if (highlightRolloff !== 0.0) {
          L1 = clamp01(L1 - highlightRolloff * 0.14 * L1 * L1 * L1);
        }

        const d2 = L0 - c2;
        const w2 = Math.exp(-d2 * d2 * inv2Sigma2);
        const rExp = Math.exp(d2 * BAND_EXP_STEP2);
        const invR = 1.0 / rExp;
        const w1 = w2 * invR * BAND_Q1;
        const w3 = w2 * rExp * BAND_Q1;
        const w0 = w2 * invR * invR * BAND_Q2;
        const w4 = w2 * rExp * rExp * BAND_Q2;
        const invW = 1.0 / Math.max(w0 + w1 + w2 + w3 + w4, 1e-12);

        const da0 = a0 - t0_sa, db0 = b0 - t0_sb;
        const da1 = a0 - t1_sa, db1 = b0 - t1_sb;
        const da2 = a0 - t2_sa, db2 = b0 - t2_sb;
        const da3 = a0 - t3_sa, db3 = b0 - t3_sb;
        const da4 = a0 - t4_sa, db4 = b0 - t4_sb;

        let aOut = invW * (
          w0 * (t0_ra + t0_00 * da0 + t0_01 * db0) +
          w1 * (t1_ra + t1_00 * da1 + t1_01 * db1) +
          w2 * (t2_ra + t2_00 * da2 + t2_01 * db2) +
          w3 * (t3_ra + t3_00 * da3 + t3_01 * db3) +
          w4 * (t4_ra + t4_00 * da4 + t4_01 * db4)
        );
        let bOut = invW * (
          w0 * (t0_rb + t0_10 * da0 + t0_11 * db0) +
          w1 * (t1_rb + t1_10 * da1 + t1_11 * db1) +
          w2 * (t2_rb + t2_10 * da2 + t2_11 * db2) +
          w3 * (t3_rb + t3_10 * da3 + t3_11 * db3) +
          w4 * (t4_rb + t4_10 * da4 + t4_11 * db4)
        );

        if (hasHueHarmonic) {
          const cSq = aOut * aOut + bOut * bOut;
          if (cSq > 1e-14) {
            const c = Math.sqrt(cSq);
            const invC = 1.0 / c;
            const ct = aOut * invC;
            const st = bOut * invC;
            const ct2 = ct * ct;
            const st2 = st * st;
            const hc2 = 2.0 * ct2 - 1.0;
            const hs2 = 2.0 * st * ct;
            const hc3 = 4.0 * ct2 * ct - 3.0 * ct;
            const hs3 = 3.0 * st - 4.0 * st2 * st;
            const gate = cSq / (cSq + gateSq);

            const rawDh = ch0 * ct + ch1 * st + ch2 * hc2 + ch3 * hs2 + ch4 * hc3 + ch5 * hs3;
            const rawDs = cs0 * ct + cs1 * st + cs2 * hc2 + cs3 * hs2 + cs4 * hc3 + cs5 * hs3;
            const dTheta = clamp(rawDh, -maxHueRad, maxHueRad) * gate;
            const dLogC = clamp(rawDs, -maxLogSat, maxLogSat) * gate;
            const scaleC = Math.exp(dLogC);
            const cd = Math.cos(dTheta);
            const sd = Math.sin(dTheta);
            const na = scaleC * (cd * aOut - sd * bOut);
            const nb = scaleC * (sd * aOut + cd * bOut);
            aOut = na;
            bOut = nb;
          }
        }

        let as = aOut;
        let bs = bOut;
        if (skinProtect > 0.0 && L0 > 0.15 && L0 < 0.95 && (a0 * a0 + b0 * b0) < 0.046225) {
          applySkinGuard(L0, a0, b0, aOut, bOut, skinProtect, abSafe);
          as = abSafe[0];
          bs = abSafe[1];
        }

        const ol = 0.9999999984505196 * L1 + 0.3963377921737678 * as + 0.21580375806075877 * bs;
        const om = 1.0000000088817607 * L1 - 0.10556134232365633 * as - 0.0638541747717059 * bs;
        const os = 1.0000000546724108 * L1 - 0.08948418209496574 * as - 1.2914855378640917 * bs;
        const ol3 = ol * ol * ol;
        const om3 = om * om * om;
        const os3 = os * os * os;
        const lrOut = 4.076741661347994 * ol3 - 3.3077115904081933 * om3 + 0.23096992872942793 * os3;
        const lgOut = -1.2684380040921763 * ol3 + 2.6097574006633715 * om3 - 0.3413193963102196 * os3;
        const lbOut = -0.004196086541837074 * ol3 - 0.7034186144594495 * om3 + 1.7076147009309446 * os3;

        workData[idx] = clamp01(linearToSrgb(lrOut));
        workData[idx + 1] = clamp01(linearToSrgb(lgOut));
        workData[idx + 2] = clamp01(linearToSrgb(lbOut));
      }
    }
  }
}

function readWorkingSrgbPixel(
  img: ImageDataLike,
  x: number,
  y: number,
  idt: IdtName | 'srgb-ref',
  outRgb: [number, number, number]
): void {
  const data = img.data;
  const isFloat = data instanceof Float32Array;
  const p = (y * img.width + x) * 4;
  const r = isFloat ? data[p]! : data[p]! / 255;
  const g = isFloat ? data[p + 1]! : data[p + 1]! / 255;
  const b = isFloat ? data[p + 2]! : data[p + 2]! / 255;

  if (idt === 'srgb' || idt === 'srgb-ref') {
    outRgb[0] = clamp01(r);
    outRgb[1] = clamp01(g);
    outRgb[2] = clamp01(b);
    return;
  }
  if (idt === 'rec709') {
    outRgb[0] = rec709ToWorkingSrgb(r);
    outRgb[1] = rec709ToWorkingSrgb(g);
    outRgb[2] = rec709ToWorkingSrgb(b);
    return;
  }
  const rLog = rawIdtChannel(r, idt);
  const gLog = rawIdtChannel(g, idt);
  const bLog = rawIdtChannel(b, idt);
  const mat =
    idt === 'apple-log'
      ? REC2020_TO_709
      : idt === 'slog3'
        ? SGAMUT3_TO_709
        : idt === 'vlog'
          ? VGAMUT_TO_709
          : CGAMUT_TO_709;
  const r7 = mat[0] * rLog + mat[1] * gLog + mat[2] * bLog;
  const g7 = mat[3] * rLog + mat[4] * gLog + mat[5] * bLog;
  const b7 = mat[6] * rLog + mat[7] * gLog + mat[8] * bLog;
  const rx = r7 > 0 ? r7 : 0;
  const gx = g7 > 0 ? g7 : 0;
  const bx = b7 > 0 ? b7 : 0;
  outRgb[0] = clamp01(linearToSrgb(clamp01((rx * (1.0 + 0.12 * rx)) / (rx * (0.85 + 0.12 * rx) + 0.95))));
  outRgb[1] = clamp01(linearToSrgb(clamp01((gx * (1.0 + 0.12 * gx)) / (gx * (0.85 + 0.12 * gx) + 0.95))));
  outRgb[2] = clamp01(linearToSrgb(clamp01((bx * (1.0 + 0.12 * bx)) / (bx * (0.85 + 0.12 * bx) + 0.95))));
}

function pearsonStats(
  x: Float64Array,
  y: Float64Array,
  n: number
): { stdX: number; stdY: number; corr: number } {
  let sumX = 0;
  let sumY = 0;
  for (let i = 0; i < n; i++) {
    sumX += x[i]!;
    sumY += y[i]!;
  }
  const meanX = sumX / n;
  const meanY = sumY / n;
  let num = 0;
  let varX = 0;
  let varY = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i]! - meanX;
    const dy = y[i]! - meanY;
    num += dx * dy;
    varX += dx * dx;
    varY += dy * dy;
  }
  const vx = varX / n;
  const vy = varY / n;
  const stdX = Math.sqrt(vx);
  const stdY = Math.sqrt(vy);
  const denom = Math.sqrt(vx * vy);
  const corr = denom > 1e-12 ? (num / n) / denom : 0;
  return { stdX, stdY, corr };
}

function isPairedShot(source: ImageDataLike, ref: ImageDataLike, idt: IdtName): boolean {
  if (source.width <= 0 || source.height <= 0 || ref.width <= 0 || ref.height <= 0) {
    return false;
  }
  const aspectSrc = source.width / source.height;
  const aspectRef = ref.width / ref.height;
  if (Math.abs(aspectSrc - aspectRef) / Math.max(aspectSrc, aspectRef) > 0.02) {
    return false;
  }

  const n = 32;
  const count = n * n;
  const ls = new Float64Array(count);
  const lr = new Float64Array(count);
  const rgb: [number, number, number] = [0, 0, 0];
  let k = 0;
  for (let iy = 0; iy < n; iy++) {
    const sy = Math.floor((iy * Math.max(source.height - 1, 0)) / (n - 1));
    const ry = Math.floor((iy * Math.max(ref.height - 1, 0)) / (n - 1));
    for (let ix = 0; ix < n; ix++) {
      const sx = Math.floor((ix * Math.max(source.width - 1, 0)) / (n - 1));
      const rx = Math.floor((ix * Math.max(ref.width - 1, 0)) / (n - 1));
      readWorkingSrgbPixel(source, sx, sy, idt, rgb);
      ls[k] = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
      readWorkingSrgbPixel(ref, rx, ry, 'srgb-ref', rgb);
      lr[k] = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
      k++;
    }
  }

  const gradCount = n * (n - 1);
  const dxS = new Float64Array(gradCount);
  const dxR = new Float64Array(gradCount);
  const dyS = new Float64Array(gradCount);
  const dyR = new Float64Array(gradCount);

  let kx = 0;
  for (let iy = 0; iy < n; iy++) {
    const row = iy * n;
    for (let ix = 0; ix < n - 1; ix++) {
      dxS[kx] = ls[row + ix + 1]! - ls[row + ix]!;
      dxR[kx] = lr[row + ix + 1]! - lr[row + ix]!;
      kx++;
    }
  }

  let ky = 0;
  for (let iy = 0; iy < n - 1; iy++) {
    const row = iy * n;
    const nextRow = (iy + 1) * n;
    for (let ix = 0; ix < n; ix++) {
      dyS[ky] = ls[nextRow + ix]! - ls[row + ix]!;
      dyR[ky] = lr[nextRow + ix]! - lr[row + ix]!;
      ky++;
    }
  }

  const xStats = pearsonStats(dxS, dxR, gradCount);
  if (xStats.stdX <= 0.015 || xStats.stdY <= 0.015 || xStats.corr <= 0.92) {
    return false;
  }
  const yStats = pearsonStats(dyS, dyR, gradCount);
  if (yStats.stdX <= 0.015 || yStats.stdY <= 0.015 || yStats.corr <= 0.92) {
    return false;
  }
  const lStats = pearsonStats(ls, lr, count);
  if (lStats.stdX * lStats.stdX <= 1e-6 || lStats.stdY * lStats.stdY <= 1e-6 || lStats.corr <= 0.98) {
    return false;
  }
  return true;
}

function fillPairedFeatures(L: number, a: number, b: number, out: Float64Array): void {
  const cSq = a * a + b * b;
  let ct = 0.0;
  let st = 0.0;
  let c2 = 0.0;
  let s2 = 0.0;
  let c3 = 0.0;
  let s3 = 0.0;
  let wc = 0.0;
  if (cSq > 1e-16) {
    const c = Math.sqrt(cSq);
    const invC = 1.0 / c;
    ct = a * invC;
    st = b * invC;
    const ct2 = ct * ct;
    const st2 = st * st;
    c2 = 2.0 * ct2 - 1.0;
    s2 = 2.0 * st * ct;
    c3 = 4.0 * ct2 * ct - 3.0 * ct;
    s3 = 3.0 * st - 4.0 * st2 * st;
    wc = cSq / (c + 0.04);
  }
  const L2 = L * L;
  out[0] = 1.0;
  out[1] = L;
  out[2] = a;
  out[3] = b;
  out[4] = L2;
  out[5] = L * a;
  out[6] = L * b;
  out[7] = L2 * L;
  out[8] = wc * ct;
  out[9] = wc * st;
  out[10] = wc * c2;
  out[11] = wc * s2;
  out[12] = wc * c3;
  out[13] = wc * s3;
}

function solvePairedWorkingLattice(
  source: ImageDataLike,
  ref: ImageDataLike,
  idt: IdtName,
  workData: Float64Array,
  size: number,
  skinProtect: number,
  shadowLift: number,
  highlightRolloff: number
): void {
  const n = 64;
  const nf = 14;
  const xtx = new Float64Array(nf * nf);
  const xty = new Float64Array(nf * 3);
  const rgb: [number, number, number] = [0, 0, 0];
  const feat = new Float64Array(nf);

  for (let iy = 0; iy < n; iy++) {
    const sy = Math.floor((iy * Math.max(source.height - 1, 0)) / (n - 1));
    const ry = Math.floor((iy * Math.max(ref.height - 1, 0)) / (n - 1));
    for (let ix = 0; ix < n; ix++) {
      const sx = Math.floor((ix * Math.max(source.width - 1, 0)) / (n - 1));
      const rx = Math.floor((ix * Math.max(ref.width - 1, 0)) / (n - 1));

      readWorkingSrgbPixel(source, sx, sy, idt, rgb);
      const slr = srgbToLinear(rgb[0]);
      const slg = srgbToLinear(rgb[1]);
      const slb = srgbToLinear(rgb[2]);
      const sl_ = Math.cbrt(0.4122214708 * slr + 0.5363325363 * slg + 0.0514459929 * slb);
      const sm_ = Math.cbrt(0.2119034982 * slr + 0.6806995451 * slg + 0.1073969566 * slb);
      const ss_ = Math.cbrt(0.0883024619 * slr + 0.2817188376 * slg + 0.6299787005 * slb);
      const sL = 0.2104542553 * sl_ + 0.7936177850 * sm_ - 0.0040720468 * ss_;
      const sa = 1.9779984951 * sl_ - 2.4285922050 * sm_ + 0.4505937099 * ss_;
      const sb = 0.0259040371 * sl_ + 0.7827717662 * sm_ - 0.8086757660 * ss_;

      readWorkingSrgbPixel(ref, rx, ry, 'srgb-ref', rgb);
      const rlr = srgbToLinear(rgb[0]);
      const rlg = srgbToLinear(rgb[1]);
      const rlb = srgbToLinear(rgb[2]);
      const rl_ = Math.cbrt(0.4122214708 * rlr + 0.5363325363 * rlg + 0.0514459929 * rlb);
      const rm_ = Math.cbrt(0.2119034982 * rlr + 0.6806995451 * rlg + 0.1073969566 * rlb);
      const rs_ = Math.cbrt(0.0883024619 * rlr + 0.2817188376 * rlg + 0.6299787005 * rlb);
      const dL = 0.2104542553 * rl_ + 0.7936177850 * rm_ - 0.0040720468 * rs_ - sL;
      const da = 1.9779984951 * rl_ - 2.4285922050 * rm_ + 0.4505937099 * rs_ - sa;
      const db = 0.0259040371 * rl_ + 0.7827717662 * rm_ - 0.8086757660 * rs_ - sb;

      fillPairedFeatures(sL, sa, sb, feat);

      for (let i = 0; i < nf; i++) {
        const fi = feat[i]!;
        const rowOff = i * nf;
        for (let j = 0; j < nf; j++) {
          xtx[rowOff + j]! += fi * feat[j]!;
        }
        xty[i * 3]! += fi * dL;
        xty[i * 3 + 1]! += fi * da;
        xty[i * 3 + 2]! += fi * db;
      }
    }
  }

  for (let i = 0; i < nf; i++) xtx[i * nf + i]! += PAIRED_REG_DIAG[i]!;
  const w = solveLinearSystem(xtx, xty, nf, 3);

  const scale = 1.0 / (size - 1);
  const tickLin = new Float64Array(size);
  for (let i = 0; i < size; i++) tickLin[i] = srgbToLinear(i * scale);
  const abSafe: [number, number] = [0, 0];

  for (let b = 0; b < size; b++) {
    const lb = tickLin[b]!;
    for (let g = 0; g < size; g++) {
      const lg = tickLin[g]!;
      for (let r = 0; r < size; r++) {
        const lr = tickLin[r]!;
        const idx = lutIndex(r, g, b, size);
        const l_ = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
        const m_ = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
        const s_ = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);

        const L0 = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
        const a0 = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
        const b0 = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;

        fillPairedFeatures(L0, a0, b0, feat);
        let dLOut = 0.0;
        let daOut = 0.0;
        let dbOut = 0.0;
        for (let i = 0; i < nf; i++) {
          const fi = feat[i]!;
          dLOut += fi * w[i * 3]!;
          daOut += fi * w[i * 3 + 1]!;
          dbOut += fi * w[i * 3 + 2]!;
        }

        let L1 = clamp01(L0 + dLOut);
        if (shadowLift !== 0.0) {
          const om = 1.0 - L1;
          L1 = clamp01(L1 + shadowLift * 0.14 * om * om * om);
        }
        if (highlightRolloff !== 0.0) {
          L1 = clamp01(L1 - highlightRolloff * 0.14 * L1 * L1 * L1);
        }
        const aOut = a0 + daOut;
        const bOut = b0 + dbOut;

        let as = aOut;
        let bs = bOut;
        if (skinProtect > 0.0 && L0 > 0.15 && L0 < 0.95 && a0 * a0 + b0 * b0 < 0.046225) {
          applySkinGuard(L0, a0, b0, aOut, bOut, skinProtect, abSafe);
          as = abSafe[0];
          bs = abSafe[1];
        }

        const ol = 0.9999999984505196 * L1 + 0.3963377921737678 * as + 0.21580375806075877 * bs;
        const om = 1.0000000088817607 * L1 - 0.10556134232365633 * as - 0.0638541747717059 * bs;
        const os = 1.0000000546724108 * L1 - 0.08948418209496574 * as - 1.2914855378640917 * bs;
        const ol3 = ol * ol * ol;
        const om3 = om * om * om;
        const os3 = os * os * os;
        const lrOut = 4.076741661347994 * ol3 - 3.3077115904081933 * om3 + 0.23096992872942793 * os3;
        const lgOut = -1.2684380040921763 * ol3 + 2.6097574006633715 * om3 - 0.3413193963102196 * os3;
        const lbOut = -0.004196086541837074 * ol3 - 0.7034186144594495 * om3 + 1.7076147009309446 * os3;

        workData[idx] = clamp01(linearToSrgb(lrOut));
        workData[idx + 1] = clamp01(linearToSrgb(lgOut));
        workData[idx + 2] = clamp01(linearToSrgb(lbOut));
      }
    }
  }
}

function composeIdtIntoLut(lutWorking: Lut3D, idt: IdtName): Float32Array {
  const size = lutWorking.size;
  const scale = 1.0 / (size - 1);
  const out = new Float32Array(size * size * size * 3);
  const tmpOut: [number, number, number] = [0, 0, 0];

  if (idt === 'rec709') {
    const tick = new Float64Array(size);
    for (let i = 0; i < size; i++) tick[i] = rec709ToWorkingSrgb(i * scale);
    for (let b = 0; b < size; b++) {
      const wb = tick[b]!;
      for (let g = 0; g < size; g++) {
        const wg = tick[g]!;
        for (let r = 0; r < size; r++) {
          evaluateLutPoint(lutWorking, tick[r]!, wg, wb, tmpOut);
          const idx = lutIndex(r, g, b, size);
          out[idx] = clamp01(tmpOut[0]);
          out[idx + 1] = clamp01(tmpOut[1]);
          out[idx + 2] = clamp01(tmpOut[2]);
        }
      }
    }
    return out;
  }

  const tickLog = new Float64Array(size);
  for (let i = 0; i < size; i++) tickLog[i] = rawIdtChannel(i * scale, idt);
  const mat =
    idt === 'apple-log'
      ? REC2020_TO_709
      : idt === 'slog3'
        ? SGAMUT3_TO_709
        : idt === 'vlog'
          ? VGAMUT_TO_709
          : CGAMUT_TO_709;
  const m0 = mat[0], m1 = mat[1], m2 = mat[2];
  const m3 = mat[3], m4 = mat[4], m5 = mat[5];
  const m6 = mat[6], m7 = mat[7], m8 = mat[8];

  for (let b = 0; b < size; b++) {
    const bLog = tickLog[b]!;
    for (let g = 0; g < size; g++) {
      const gLog = tickLog[g]!;
      for (let r = 0; r < size; r++) {
        const rLog = tickLog[r]!;
        const r7 = m0 * rLog + m1 * gLog + m2 * bLog;
        const g7 = m3 * rLog + m4 * gLog + m5 * bLog;
        const b7 = m6 * rLog + m7 * gLog + m8 * bLog;
        const rx = r7 > 0 ? r7 : 0;
        const gx = g7 > 0 ? g7 : 0;
        const bx = b7 > 0 ? b7 : 0;
        const sr = clamp01(linearToSrgb(clamp01((rx * (1.0 + 0.12 * rx)) / (rx * (0.85 + 0.12 * rx) + 0.95))));
        const sg = clamp01(linearToSrgb(clamp01((gx * (1.0 + 0.12 * gx)) / (gx * (0.85 + 0.12 * gx) + 0.95))));
        const sb = clamp01(linearToSrgb(clamp01((bx * (1.0 + 0.12 * bx)) / (bx * (0.85 + 0.12 * bx) + 0.95))));
        evaluateLutPoint(lutWorking, sr, sg, sb, tmpOut);
        const idx = lutIndex(r, g, b, size);
        out[idx] = clamp01(tmpOut[0]);
        out[idx + 1] = clamp01(tmpOut[1]);
        out[idx + 2] = clamp01(tmpOut[2]);
      }
    }
  }
  return out;
}

function computeInversions(
  data: Float32Array,
  size: number
): { inversionCount: number; inversionRate: number } {
  if (size < 2) return { inversionCount: 0, inversionRate: 0 };
  const eps = 1e-6;
  const strideG = size * 3;
  const strideB = size * size * 3;
  const totalChecks = 6 * (size - 1) * size * size;
  let inversions = 0;

  for (let b = 0; b < size; b++) {
    for (let g = 0; g < size; g++) {
      const rowBase = b * strideB + g * strideG;
      for (let r = 0; r < size; r++) {
        const idx = rowBase + r * 3;
        const r0 = data[idx]!;
        const g0 = data[idx + 1]!;
        const b0 = data[idx + 2]!;
        const l0 = 0.2126 * r0 + 0.7152 * g0 + 0.0722 * b0;

        if (r + 1 < size) {
          const next = idx + 3;
          const r1 = data[next]!;
          const l1 = 0.2126 * r1 + 0.7152 * data[next + 1]! + 0.0722 * data[next + 2]!;
          if (r1 < r0 - eps) inversions++;
          if (l1 < l0 - eps) inversions++;
        }
        if (g + 1 < size) {
          const next = idx + strideG;
          const g1 = data[next + 1]!;
          const l1 = 0.2126 * data[next]! + 0.7152 * g1 + 0.0722 * data[next + 2]!;
          if (g1 < g0 - eps) inversions++;
          if (l1 < l0 - eps) inversions++;
        }
        if (b + 1 < size) {
          const next = idx + strideB;
          const b1 = data[next + 2]!;
          const l1 = 0.2126 * data[next]! + 0.7152 * data[next + 1]! + 0.0722 * b1;
          if (b1 < b0 - eps) inversions++;
          if (l1 < l0 - eps) inversions++;
        }
      }
    }
  }

  return { inversionCount: inversions, inversionRate: inversions / totalChecks };
}

function applyStrengthToWorkOut(
  workData: Float64Array,
  size: number,
  strength: number,
  workOut: Float32Array
): void {
  const workLen = size * size * size * 3;
  if (strength < 1.0) {
    const scale = 1.0 / (size - 1);
    for (let b = 0; b < size; b++) {
      const fb = b * scale;
      for (let g = 0; g < size; g++) {
        const fg = g * scale;
        for (let r = 0; r < size; r++) {
          const fr = r * scale;
          const idx = lutIndex(r, g, b, size);
          workOut[idx] = clamp01((1.0 - strength) * fr + strength * workData[idx]!);
          workOut[idx + 1] = clamp01((1.0 - strength) * fg + strength * workData[idx + 1]!);
          workOut[idx + 2] = clamp01((1.0 - strength) * fb + strength * workData[idx + 2]!);
        }
      }
    }
  } else {
    for (let i = 0; i < workLen; i++) {
      workOut[i] = clamp01(workData[i]!);
    }
  }
}

export function fitLut(options: FitLutOptions): Lut3D {
  const startMs = performance.now();

  if (!options.references || options.references.length === 0) {
    throw new Error('fitLut requires at least one reference image');
  }
  const sources = Array.isArray(options.source) ? options.source : [options.source];
  if (sources.length === 0) {
    throw new Error('fitLut requires at least one source image');
  }
  sources.forEach((src, i) => validateImage(src, `source[${i}]`));
  options.references.forEach((ref, i) => validateImage(ref, `reference[${i}]`));

  const size: LutSize = options.size ?? 33;
  if (size !== 17 && size !== 33 && size !== 65) {
    throw new Error(`LUT size must be 17, 33, or 65; got ${size}`);
  }
  const strength = options.strength ?? 1.0;
  if (strength < 0 || strength > 1) {
    throw new Error(`strength must be in [0, 1], got ${strength}`);
  }
  const skinProtect = options.skinProtect ?? 0.5;
  if (skinProtect < 0 || skinProtect > 1) {
    throw new Error(`skinProtect must be in [0, 1], got ${skinProtect}`);
  }
  const highlightRolloff = options.highlightRolloff ?? 0.0;
  const shadowLift = options.shadowLift ?? 0.0;
  const smoothness = options.smoothness ?? 0.35;
  const idt: IdtName = options.idt ?? 'srgb';
  const mode: TransferMode = options.mode ?? 'balanced';
  if (mode !== 'balanced' && mode !== 'creative') {
    throw new Error(`mode must be 'balanced' or 'creative'; got ${String(mode)}`);
  }
  const cfg = MODE_CONFIG[mode];

  const workLen = size * size * size * 3;
  ensureCubeScratch(size);
  const workData = scratchCubeWork;
  const workOut = new Float32Array(workLen);
  let srcCount = 0;
  let refCount = 0;
  let solvedPaired = false;

  if (
    sources.length === 1 &&
    options.references.length === 1 &&
    isPairedShot(sources[0]!, options.references[0]!, idt)
  ) {
    srcCount = 4096;
    refCount = 4096;
    solvePairedWorkingLattice(
      sources[0]!,
      options.references[0]!,
      idt,
      workData,
      size,
      skinProtect,
      shadowLift,
      highlightRolloff
    );
    const sm = smoothness * 0.15;
    laplacianSmooth3d(workData, size, sm);
    projectMonotoneLattice(workData, size, 64);
    monotoneSeparableSmooth3d(workData, size, sm);
    applyStrengthToWorkOut(workData, size, strength, workOut);
    if (computeInversions(workOut, size).inversionCount === 0) {
      solvedPaired = true;
    }
  }

  if (!solvedPaired) {
    const srcSamples = sampleSources(sources, idt, MAX_SAMPLES_PER_IMAGE);
    srcCount = srcSamples.count;
    const refSamples = sampleReferences(options.references, MAX_SAMPLES_PER_IMAGE);
    refCount = refSamples.count;

    const lSpline = fitLuminanceCurve(
      srcSamples.L,
      srcSamples.weights,
      srcCount,
      refSamples.L,
      refSamples.weights,
      refSamples.count,
      0.0,
      0.0
    );

    const inv2Sigma2 = 1.0 / (2.0 * BAND_SIGMA * BAND_SIGMA);
    const inv2Neu2 = 1.0 / (2.0 * 0.14 * 0.14);
    const srcNeuDecay = scratchSrcNeu;
    for (let i = 0; i < srcCount; i++) {
      const av = srcSamples.a[i]!;
      const bv = srcSamples.b[i]!;
      srcNeuDecay[i] = Math.exp(-(av * av + bv * bv) * inv2Neu2);
    }
    const refNeuDecay = scratchRefNeu;
    for (let i = 0; i < refSamples.count; i++) {
      const av = refSamples.a[i]!;
      const bv = refSamples.b[i]!;
      refNeuDecay[i] = Math.exp(-(av * av + bv * bv) * inv2Neu2);
    }

    const srcBandW = scratchSrcBandW;
    const refBandW = scratchRefBandW;
    const trFlat = new Float64Array(BAND_CENTERS.length * 8);
    const oneMinusMkl = 1.0 - cfg.mklAlpha;

    for (let k = 0; k < BAND_CENTERS.length; k++) {
      const ck = BAND_CENTERS[k]!;
      const statsS = huberMeanCovBand(
        srcSamples.L,
        srcSamples.a,
        srcSamples.b,
        srcSamples.weights,
        srcNeuDecay,
        srcBandW,
        ck,
        inv2Sigma2,
        srcCount
      );
      const statsR = huberMeanCovBand(
        refSamples.L,
        refSamples.a,
        refSamples.b,
        refSamples.weights,
        refNeuDecay,
        refBandW,
        ck,
        inv2Sigma2,
        refSamples.count
      );
      const [mkl00, mkl01, mkl10, mkl11] = regularizedMkl2x2(
        statsS.c00, statsS.c01, statsS.c11,
        statsR.c00, statsR.c01, statsR.c11
      );
      const t00 = oneMinusMkl + cfg.mklAlpha * mkl00;
      const t01 = cfg.mklAlpha * mkl01;
      const t10 = cfg.mklAlpha * mkl10;
      const t11 = oneMinusMkl + cfg.mklAlpha * mkl11;

      const neuConfidence = clamp(
        Math.min(statsS.neutralCoverage, statsR.neutralCoverage) / 0.18,
        0.15,
        1.0
      );
      let shiftA = cfg.shiftAlpha * (statsR.muA - statsS.muA);
      let shiftB = cfg.shiftAlpha * (statsR.muB - statsS.muB);
      const shiftNorm = Math.sqrt(shiftA * shiftA + shiftB * shiftB);
      const maxShift = cfg.shiftMaxScale * neuConfidence + cfg.shiftMaxBase;
      if (shiftNorm > maxShift) {
        const scaleShift = maxShift / shiftNorm;
        shiftA *= scaleShift;
        shiftB *= scaleShift;
      }

      const off = k * 8;
      trFlat[off] = statsS.muA;
      trFlat[off + 1] = statsS.muB;
      trFlat[off + 2] = statsS.muA + shiftA;
      trFlat[off + 3] = statsS.muB + shiftB;
      trFlat[off + 4] = t00;
      trFlat[off + 5] = t01;
      trFlat[off + 6] = t10;
      trFlat[off + 7] = t11;
    }

    // Evaluate 5-band MKL on sampled source pixels, then fit k=1,2,3 hue-harmonic residual
    const srcMklA = scratchSrcMklA;
    const srcMklB = scratchSrcMklB;
    const c2 = BAND_CENTERS[2];
    const t0_sa = trFlat[0]!, t0_sb = trFlat[1]!, t0_ra = trFlat[2]!, t0_rb = trFlat[3]!;
    const t0_00 = trFlat[4]!, t0_01 = trFlat[5]!, t0_10 = trFlat[6]!, t0_11 = trFlat[7]!;
    const t1_sa = trFlat[8]!, t1_sb = trFlat[9]!, t1_ra = trFlat[10]!, t1_rb = trFlat[11]!;
    const t1_00 = trFlat[12]!, t1_01 = trFlat[13]!, t1_10 = trFlat[14]!, t1_11 = trFlat[15]!;
    const t2_sa = trFlat[16]!, t2_sb = trFlat[17]!, t2_ra = trFlat[18]!, t2_rb = trFlat[19]!;
    const t2_00 = trFlat[20]!, t2_01 = trFlat[21]!, t2_10 = trFlat[22]!, t2_11 = trFlat[23]!;
    const t3_sa = trFlat[24]!, t3_sb = trFlat[25]!, t3_ra = trFlat[26]!, t3_rb = trFlat[27]!;
    const t3_00 = trFlat[28]!, t3_01 = trFlat[29]!, t3_10 = trFlat[30]!, t3_11 = trFlat[31]!;
    const t4_sa = trFlat[32]!, t4_sb = trFlat[33]!, t4_ra = trFlat[34]!, t4_rb = trFlat[35]!;
    const t4_00 = trFlat[36]!, t4_01 = trFlat[37]!, t4_10 = trFlat[38]!, t4_11 = trFlat[39]!;

    for (let i = 0; i < srcCount; i++) {
      const L0 = srcSamples.L[i]!;
      const a0 = srcSamples.a[i]!;
      const b0 = srcSamples.b[i]!;
      const d2 = L0 - c2;
      const w2 = Math.exp(-d2 * d2 * inv2Sigma2);
      const rExp = Math.exp(d2 * BAND_EXP_STEP2);
      const invR = 1.0 / rExp;
      const w1 = w2 * invR * BAND_Q1;
      const w3 = w2 * rExp * BAND_Q1;
      const w0 = w2 * invR * invR * BAND_Q2;
      const w4 = w2 * rExp * rExp * BAND_Q2;
      const invW = 1.0 / Math.max(w0 + w1 + w2 + w3 + w4, 1e-12);
      const da0 = a0 - t0_sa, db0 = b0 - t0_sb;
      const da1 = a0 - t1_sa, db1 = b0 - t1_sb;
      const da2 = a0 - t2_sa, db2 = b0 - t2_sb;
      const da3 = a0 - t3_sa, db3 = b0 - t3_sb;
      const da4 = a0 - t4_sa, db4 = b0 - t4_sb;
      srcMklA[i] = invW * (
        w0 * (t0_ra + t0_00 * da0 + t0_01 * db0) +
        w1 * (t1_ra + t1_00 * da1 + t1_01 * db1) +
        w2 * (t2_ra + t2_00 * da2 + t2_01 * db2) +
        w3 * (t3_ra + t3_00 * da3 + t3_01 * db3) +
        w4 * (t4_ra + t4_00 * da4 + t4_01 * db4)
      );
      srcMklB[i] = invW * (
        w0 * (t0_rb + t0_10 * da0 + t0_11 * db0) +
        w1 * (t1_rb + t1_10 * da1 + t1_11 * db1) +
        w2 * (t2_rb + t2_10 * da2 + t2_11 * db2) +
        w3 * (t3_rb + t3_10 * da3 + t3_11 * db3) +
        w4 * (t4_rb + t4_10 * da4 + t4_11 * db4)
      );
    }

    const coefHue = new Float64Array(6);
    const coefSat = new Float64Array(6);
    fitHueHarmonicResidual(
      srcMklA,
      srcMklB,
      srcSamples.weights,
      srcCount,
      refSamples.a,
      refSamples.b,
      refSamples.weights,
      refSamples.count,
      cfg,
      coefHue,
      coefSat
    );

    evaluateLattice(
      workData,
      size,
      lSpline,
      trFlat,
      skinProtect,
      shadowLift,
      highlightRolloff,
      inv2Sigma2,
      cfg.lumaAlpha,
      coefHue,
      coefSat,
      cfg.maxHueRad,
      cfg.maxLogSat
    );
    laplacianSmooth3d(workData, size, smoothness);
    projectMonotoneLattice(workData, size);
    monotoneSeparableSmooth3d(workData, size, smoothness);
    applyStrengthToWorkOut(workData, size, strength, workOut);
  }

  const lut: Lut3D = {
    size,
    domainMin: [0, 0, 0],
    domainMax: [1, 1, 1],
    data: workOut,
    title: options.title ?? `lut-match (${idt})`,
  };

  if (idt !== 'srgb') {
    lut.data = composeIdtIntoLut(lut, idt);
  }

  const { inversionCount, inversionRate } = computeInversions(lut.data, size);
  lut.diagnostics = {
    solveMs: performance.now() - startMs,
    sourceSamples: srcCount,
    referenceSamples: refCount,
    inversionCount,
    inversionRate,
  };

  return lut;
}
