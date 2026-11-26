import { clamp01 } from './color.js';
import type {
  HaldClutImage,
  ImageDataLike,
  Lut3D,
  LutInspection,
  LutSize,
} from './types.js';

export function lutIndex(r: number, g: number, b: number, size: number): number {
  return ((b * size + g) * size + r) * 3;
}

export function createIdentityLut(size: LutSize = 33, title = 'Identity'): Lut3D {
  const data = new Float32Array(size * size * size * 3);
  const scale = 1 / (size - 1);
  for (let b = 0; b < size; b++) {
    const fb = b * scale;
    for (let g = 0; g < size; g++) {
      const fg = g * scale;
      for (let r = 0; r < size; r++) {
        const idx = lutIndex(r, g, b, size);
        data[idx] = r * scale;
        data[idx + 1] = fg;
        data[idx + 2] = fb;
      }
    }
  }
  return {
    size,
    domainMin: [0, 0, 0],
    domainMax: [1, 1, 1],
    data,
    title,
  };
}

export function evaluateLutPoint(
  lut: Lut3D,
  r: number,
  g: number,
  b: number,
  out: [number, number, number]
): void {
  const size = lut.size;
  const data = lut.data;
  const maxIdx = size - 1;

  const [minR, minG, minB] = lut.domainMin;
  const [maxR, maxG, maxB] = lut.domainMax;
  const spanR = maxR - minR || 1;
  const spanG = maxG - minG || 1;
  const spanB = maxB - minB || 1;

  const nr = clamp01((r - minR) / spanR) * maxIdx;
  const ng = clamp01((g - minG) / spanG) * maxIdx;
  const nb = clamp01((b - minB) / spanB) * maxIdx;

  const r0 = Math.min(size - 2, Math.floor(nr));
  const g0 = Math.min(size - 2, Math.floor(ng));
  const b0 = Math.min(size - 2, Math.floor(nb));
  const r1 = r0 + 1;
  const g1 = g0 + 1;
  const b1 = b0 + 1;

  const fx = nr - r0;
  const fy = ng - g0;
  const fz = nb - b0;

  const i000 = ((b0 * size + g0) * size + r0) * 3;
  const i100 = ((b0 * size + g0) * size + r1) * 3;
  const i010 = ((b0 * size + g1) * size + r0) * 3;
  const i110 = ((b0 * size + g1) * size + r1) * 3;
  const i001 = ((b1 * size + g0) * size + r0) * 3;
  const i101 = ((b1 * size + g0) * size + r1) * 3;
  const i011 = ((b1 * size + g1) * size + r0) * 3;
  const i111 = ((b1 * size + g1) * size + r1) * 3;

  let w000 = 0;
  let w100 = 0;
  let w010 = 0;
  let w110 = 0;
  let w001 = 0;
  let w101 = 0;
  let w011 = 0;
  let w111 = 0;

  if (fx >= fy) {
    if (fy >= fz) {
      w000 = 1 - fx;
      w100 = fx - fy;
      w110 = fy - fz;
      w111 = fz;
    } else if (fx >= fz) {
      w000 = 1 - fx;
      w100 = fx - fz;
      w101 = fz - fy;
      w111 = fy;
    } else {
      w000 = 1 - fz;
      w001 = fz - fx;
      w101 = fx - fy;
      w111 = fy;
    }
  } else {
    if (fz >= fy) {
      w000 = 1 - fz;
      w001 = fz - fy;
      w011 = fy - fx;
      w111 = fx;
    } else if (fz >= fx) {
      w000 = 1 - fy;
      w010 = fy - fz;
      w011 = fz - fx;
      w111 = fx;
    } else {
      w000 = 1 - fy;
      w010 = fy - fx;
      w110 = fx - fz;
      w111 = fz;
    }
  }

  for (let c = 0; c < 3; c++) {
    out[c] =
      w000 * (data[i000 + c] ?? 0) +
      w100 * (data[i100 + c] ?? 0) +
      w010 * (data[i010 + c] ?? 0) +
      w110 * (data[i110 + c] ?? 0) +
      w001 * (data[i001 + c] ?? 0) +
      w101 * (data[i101 + c] ?? 0) +
      w011 * (data[i011 + c] ?? 0) +
      w111 * (data[i111 + c] ?? 0);
  }
}

export function applyLutCpu(image: ImageDataLike, lut: Lut3D): ImageDataLike {
  const { width, height, data } = image;
  const total = width * height;
  const isFloat = data instanceof Float32Array;
  const outData = isFloat
    ? new Float32Array(total * 4)
    : new Uint8ClampedArray(total * 4);
  const rgb: [number, number, number] = [0, 0, 0];

  for (let i = 0; i < total; i++) {
    const p = i * 4;
    const r = isFloat ? (data[p] ?? 0) : (data[p] ?? 0) / 255;
    const g = isFloat ? (data[p + 1] ?? 0) : (data[p + 1] ?? 0) / 255;
    const b = isFloat ? (data[p + 2] ?? 0) : (data[p + 2] ?? 0) / 255;
    evaluateLutPoint(lut, r, g, b, rgb);
    if (isFloat) {
      outData[p] = clamp01(rgb[0]);
      outData[p + 1] = clamp01(rgb[1]);
      outData[p + 2] = clamp01(rgb[2]);
      outData[p + 3] = data[p + 3] ?? 1;
    } else {
      outData[p] = Math.round(clamp01(rgb[0]) * 255);
      outData[p + 1] = Math.round(clamp01(rgb[1]) * 255);
      outData[p + 2] = Math.round(clamp01(rgb[2]) * 255);
      outData[p + 3] = data[p + 3] ?? 255;
    }
  }

  return { width, height, data: outData };
}

export function serializeCube(lut: Lut3D): string {
  const safeTitle = (lut.title ?? 'lut-match').replace(/[\r\n"]/g, ' ').trim();
  const lines: string[] = [
    `TITLE "${safeTitle}"`,
    `LUT_3D_SIZE ${lut.size}`,
    `DOMAIN_MIN ${lut.domainMin[0]} ${lut.domainMin[1]} ${lut.domainMin[2]}`,
    `DOMAIN_MAX ${lut.domainMax[0]} ${lut.domainMax[1]} ${lut.domainMax[2]}`,
  ];

  const size = lut.size;
  const data = lut.data;
  for (let b = 0; b < size; b++) {
    for (let g = 0; g < size; g++) {
      for (let r = 0; r < size; r++) {
        const idx = ((b * size + g) * size + r) * 3;
        const vr = (data[idx] ?? 0).toFixed(6);
        const vg = (data[idx + 1] ?? 0).toFixed(6);
        const vb = (data[idx + 2] ?? 0).toFixed(6);
        lines.push(`${vr} ${vg} ${vb}`);
      }
    }
  }
  return `${lines.join('\n')}\n`;
}

export function parseCube(text: string): Lut3D {
  let title = 'Imported LUT';
  let size: number | null = null;
  let domainMin: [number, number, number] = [0, 0, 0];
  let domainMax: [number, number, number] = [1, 1, 1];
  const values: number[] = [];

  const lines = text.split(/\r?\n/);
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;

    if (line.startsWith('TITLE')) {
      const m = line.match(/^TITLE\s+"?([^"]*)"?$/);
      if (m && m[1] !== undefined) title = m[1].trim();
      continue;
    }
    if (line.startsWith('LUT_3D_SIZE')) {
      const parts = line.split(/\s+/);
      const parsed = Number(parts[1]);
      if (!Number.isInteger(parsed) || parsed < 2) {
        throw new Error(`Invalid LUT_3D_SIZE: ${line}`);
      }
      size = parsed;
      continue;
    }
    if (line.startsWith('DOMAIN_MIN')) {
      const parts = line.split(/\s+/).slice(1).map(Number);
      if (parts.length !== 3 || parts.some(Number.isNaN)) {
        throw new Error(`Invalid DOMAIN_MIN: ${line}`);
      }
      domainMin = [parts[0]!, parts[1]!, parts[2]!];
      continue;
    }
    if (line.startsWith('DOMAIN_MAX')) {
      const parts = line.split(/\s+/).slice(1).map(Number);
      if (parts.length !== 3 || parts.some(Number.isNaN)) {
        throw new Error(`Invalid DOMAIN_MAX: ${line}`);
      }
      domainMax = [parts[0]!, parts[1]!, parts[2]!];
      continue;
    }
    if (/^[A-Z_]+/.test(line) && !/^[+-]?\d/.test(line)) {
      continue;
    }

    const parts = line.split(/\s+/);
    if (parts.length >= 3) {
      const r = Number(parts[0]);
      const g = Number(parts[1]);
      const b = Number(parts[2]);
      if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) {
        throw new Error(`Invalid LUT entry: ${line}`);
      }
      values.push(r, g, b);
    }
  }

  if (size === null) {
    throw new Error('Missing LUT_3D_SIZE in .cube file');
  }
  const expected = size * size * size * 3;
  if (values.length !== expected) {
    throw new Error(
      `Expected ${expected / 3} LUT entries for size ${size}, got ${values.length / 3}`
    );
  }
  if (size !== 17 && size !== 33 && size !== 65) {
    throw new Error(`Unsupported LUT_3D_SIZE ${size}; expected 17, 33, or 65`);
  }

  return {
    size,
    domainMin,
    domainMax,
    data: new Float32Array(values),
    title,
  };
}

export function generateHaldClut(lut: Lut3D): HaldClutImage {
  const side = 64;
  const width = 512;
  const height = 512;
  const totalPixels = width * height;
  const data = new Uint8ClampedArray(totalPixels * 4);
  const scale = 1 / (side - 1);
  const rgb: [number, number, number] = [0, 0, 0];

  for (let p = 0; p < totalPixels; p++) {
    const rIdx = p % side;
    const gIdx = Math.floor(p / side) % side;
    const bIdx = Math.floor(p / (side * side));

    evaluateLutPoint(lut, rIdx * scale, gIdx * scale, bIdx * scale, rgb);
    const outIdx = p * 4;
    data[outIdx] = Math.round(clamp01(rgb[0]) * 255);
    data[outIdx + 1] = Math.round(clamp01(rgb[1]) * 255);
    data[outIdx + 2] = Math.round(clamp01(rgb[2]) * 255);
    data[outIdx + 3] = 255;
  }

  return { width: 512, height: 512, data };
}

export function inspectLut(lut: Lut3D): LutInspection {
  const size = lut.size;
  const data = lut.data;
  const eps = 1e-6;

  let inversions = 0;
  let maxStep = 0;
  const totalSteps = size >= 2 ? 6 * (size - 1) * size * size : 0;

  for (let b = 0; b < size; b++) {
    for (let g = 0; g < size; g++) {
      for (let r = 0; r < size; r++) {
        const idx = ((b * size + g) * size + r) * 3;
        const r0 = data[idx]!;
        const g0 = data[idx + 1]!;
        const b0 = data[idx + 2]!;
        const l0 = 0.2126 * r0 + 0.7152 * g0 + 0.0722 * b0;

        if (r + 1 < size) {
          const next = idx + 3;
          const r1 = data[next]!;
          const g1 = data[next + 1]!;
          const b1 = data[next + 2]!;
          const l1 = 0.2126 * r1 + 0.7152 * g1 + 0.0722 * b1;
          if (r1 < r0 - eps) inversions++;
          if (l1 < l0 - eps) inversions++;
          maxStep = Math.max(maxStep, Math.hypot(r1 - r0, g1 - g0, b1 - b0));
        }
        if (g + 1 < size) {
          const next = idx + size * 3;
          const r1 = data[next]!;
          const g1 = data[next + 1]!;
          const b1 = data[next + 2]!;
          const l1 = 0.2126 * r1 + 0.7152 * g1 + 0.0722 * b1;
          if (g1 < g0 - eps) inversions++;
          if (l1 < l0 - eps) inversions++;
          maxStep = Math.max(maxStep, Math.hypot(r1 - r0, g1 - g0, b1 - b0));
        }
        if (b + 1 < size) {
          const next = idx + size * size * 3;
          const r1 = data[next]!;
          const g1 = data[next + 1]!;
          const b1 = data[next + 2]!;
          const l1 = 0.2126 * r1 + 0.7152 * g1 + 0.0722 * b1;
          if (b1 < b0 - eps) inversions++;
          if (l1 < l0 - eps) inversions++;
          maxStep = Math.max(maxStep, Math.hypot(r1 - r0, g1 - g0, b1 - b0));
        }
      }
    }
  }

  let lapSumSq = 0;
  let lapCount = 0;
  const strideG = size * 3;
  const strideB = size * size * 3;
  if (size >= 3) {
    for (let b = 1; b < size - 1; b++) {
      for (let g = 1; g < size - 1; g++) {
        for (let r = 1; r < size - 1; r++) {
          const c0 = b * strideB + g * strideG + r * 3;
          for (let c = 0; c < 3; c++) {
            const p = c0 + c;
            const lap =
              data[p - 3]! +
              data[p + 3]! +
              data[p - strideG]! +
              data[p + strideG]! +
              data[p - strideB]! +
              data[p + strideB]! -
              6 * data[p]!;
            lapSumSq += lap * lap;
          }
          lapCount++;
        }
      }
    }
  }

  let clippedNodes = 0;
  const totalNodes = size * size * size;
  for (let i = 0; i < totalNodes; i++) {
    const p = i * 3;
    const r = data[p]!;
    const g = data[p + 1]!;
    const b = data[p + 2]!;
    if (
      r < eps || r > 1 - eps ||
      g < eps || g > 1 - eps ||
      b < eps || b > 1 - eps
    ) {
      clippedNodes++;
    }
  }

  return {
    inversionCount: inversions,
    inversionRate: totalSteps > 0 ? inversions / totalSteps : 0,
    laplacianRoughness: lapCount > 0 ? Math.sqrt(lapSumSq / lapCount) : 0,
    maxStep,
    clippingFraction: totalNodes > 0 ? clippedNodes / totalNodes : 0,
  };
}
