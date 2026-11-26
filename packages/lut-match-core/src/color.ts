import type { IdtName } from './types.js';

export const IDT_LIST: readonly IdtName[] = [
  'srgb',
  'rec709',
  'apple-log',
  'slog3',
  'vlog',
  'clog3',
];

const M_REC2020_TO_REC709 = [
  1.6604910021084343, -0.5876411387885495, -0.07284986331988486,
  -0.1245504745215906, 1.13289989712596, -0.00834942260436947,
  -0.01815076335490522, -0.10057889800800743, 1.118729661362913,
] as const;

const M_SGAMUT3CINE_TO_REC709 = [
  1.6269474097290801, -0.5401385388696353, -0.0868088708594452,
  -0.17851552711487662, 1.4179409274640795, -0.23942540034920257,
  -0.04443611500929753, -0.1959199661720156, 1.240356081181313,
] as const;

const M_VGAMUT_TO_REC709 = [
  1.806576, -0.695697, -0.110879,
  -0.170090, 1.305955, -0.135865,
  -0.025206, -0.154468, 1.179674,
] as const;

const M_CGAMUT_TO_REC709 = [
  1.9238612958985506, -0.7987606632484301, -0.12510063265012045,
  -0.2043108482009208, 1.4958985097509383, -0.29158766155001764,
  -0.02368502097034144, -0.4201270110302702, 1.4438120320006116,
] as const;

export function clamp01(v: number): number {
  return v <= 0 ? 0 : v >= 1 ? 1 : v;
}

export function clamp(v: number, lo: number, hi: number): number {
  return v <= lo ? lo : v >= hi ? hi : v;
}

export function srgbToLinear(c: number): number {
  if (c <= 0.04045) return c <= 0 ? 0 : c / 12.92;
  return Math.pow((c + 0.055) / 1.055, 2.4);
}

export function linearToSrgb(c: number): number {
  if (c <= 0) return 0;
  if (c >= 1) return 1;
  if (c <= 0.0031308) return 12.92 * c;
  return 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

function rec709ToLinearRaw(c: number): number {
  const alpha = 1.09929682680944;
  const cut = 0.0812428582986315;
  if (c < cut) return c / 4.5;
  return Math.pow((Math.max(c, cut) + (alpha - 1.0)) / alpha, 1.0 / 0.45);
}

export function rec709ToLinear(c: number): number {
  return c <= 0 ? 0 : rec709ToLinearRaw(c);
}

function appleLogToLinearRaw(p: number): number {
  const r0 = -0.05641088;
  const rt = 0.01;
  const coeff = 47.28711236;
  const beta = 0.00964052;
  const gamma = 0.08550479;
  const delta = 0.69336945;
  const pt = coeff * (rt - r0) * (rt - r0);
  if (p >= pt) {
    return Math.pow(2.0, (p - delta) / gamma) - beta;
  }
  if (p > 0.0) {
    return Math.sqrt(Math.max(p, 0.0) / coeff) + r0;
  }
  return r0;
}

export function appleLogToLinear(c: number): number {
  return Math.max(0, appleLogToLinearRaw(c));
}

function slog3ToLinearRaw(c: number): number {
  const t = c * 1023.0;
  const cut = 171.2102946929;
  if (t >= cut) {
    return Math.pow(10.0, (t - 420.0) / 261.5) * 0.19 - 0.01;
  }
  return ((t - 95.0) * 0.01125) / (cut - 95.0);
}

export function slog3ToLinear(c: number): number {
  return Math.max(0, slog3ToLinearRaw(c));
}

function vlogToLinearRaw(c: number): number {
  if (c >= 0.181) {
    return Math.pow(10.0, (c - 0.598206) / 0.241514) - 0.00873;
  }
  return (c - 0.125) / 5.6;
}

export function vlogToLinear(c: number): number {
  return Math.max(0, vlogToLinearRaw(c));
}

function clog3ToLinearRaw(y: number): number {
  if (y >= 0.15277891) {
    return (Math.pow(10.0, (y - 0.12240537) / 0.36726845) - 1.0) / 14.98325;
  }
  if (y <= 0.09746547) {
    return -(Math.pow(10.0, (0.12783901 - y) / 0.36726845) - 1.0) / 14.98325;
  }
  return (y - 0.12512219) / 1.9754798;
}

export function clog3ToLinear(c: number): number {
  return Math.max(0, clog3ToLinearRaw(c));
}

export function idtChannelToLinear(c: number, idt: IdtName): number {
  switch (idt) {
    case 'srgb':
      return srgbToLinear(c);
    case 'rec709':
      return rec709ToLinear(c);
    case 'apple-log':
      return appleLogToLinear(c);
    case 'slog3':
      return slog3ToLinear(c);
    case 'vlog':
      return vlogToLinear(c);
    case 'clog3':
      return clog3ToLinear(c);
  }
}

export function idtRgbToLinear(
  r: number,
  g: number,
  b: number,
  idt: IdtName,
  out: [number, number, number]
): void {
  out[0] = idtChannelToLinear(r, idt);
  out[1] = idtChannelToLinear(g, idt);
  out[2] = idtChannelToLinear(b, idt);
}

function toneMapSceneToDisplay(v: number): number {
  const x = v > 0 ? v : 0;
  const mapped = (x * (1.0 + 0.12 * x)) / (x * (0.85 + 0.12 * x) + 0.95);
  return clamp01(mapped);
}

/** Convert encoded image RGB from the specified IDT into working sRGB [0, 1]. */
export function applyIdtSrgb(
  r: number,
  g: number,
  b: number,
  idt: IdtName,
  out: [number, number, number]
): void {
  if (idt === 'srgb') {
    out[0] = clamp01(r);
    out[1] = clamp01(g);
    out[2] = clamp01(b);
    return;
  }
  if (idt === 'rec709') {
    out[0] = clamp01(linearToSrgb(clamp01(rec709ToLinearRaw(r))));
    out[1] = clamp01(linearToSrgb(clamp01(rec709ToLinearRaw(g))));
    out[2] = clamp01(linearToSrgb(clamp01(rec709ToLinearRaw(b))));
    return;
  }

  let lr: number;
  let lg: number;
  let lb: number;
  let m: readonly [
    number, number, number,
    number, number, number,
    number, number, number,
  ];
  if (idt === 'apple-log') {
    lr = appleLogToLinearRaw(r);
    lg = appleLogToLinearRaw(g);
    lb = appleLogToLinearRaw(b);
    m = M_REC2020_TO_REC709;
  } else if (idt === 'slog3') {
    lr = slog3ToLinearRaw(r);
    lg = slog3ToLinearRaw(g);
    lb = slog3ToLinearRaw(b);
    m = M_SGAMUT3CINE_TO_REC709;
  } else if (idt === 'vlog') {
    lr = vlogToLinearRaw(r);
    lg = vlogToLinearRaw(g);
    lb = vlogToLinearRaw(b);
    m = M_VGAMUT_TO_REC709;
  } else {
    lr = clog3ToLinearRaw(r);
    lg = clog3ToLinearRaw(g);
    lb = clog3ToLinearRaw(b);
    m = M_CGAMUT_TO_REC709;
  }

  const r709 = m[0] * lr + m[1] * lg + m[2] * lb;
  const g709 = m[3] * lr + m[4] * lg + m[5] * lb;
  const b709 = m[6] * lr + m[7] * lg + m[8] * lb;

  out[0] = clamp01(linearToSrgb(toneMapSceneToDisplay(r709)));
  out[1] = clamp01(linearToSrgb(toneMapSceneToDisplay(g709)));
  out[2] = clamp01(linearToSrgb(toneMapSceneToDisplay(b709)));
}

export function linearRgbToOklab(
  r: number,
  g: number,
  b: number,
  out: [number, number, number]
): void {
  const l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
  const m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
  const s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

  const l_ = Math.cbrt(l);
  const m_ = Math.cbrt(m);
  const s_ = Math.cbrt(s);

  out[0] = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
  out[1] = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
  out[2] = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;
}

export function oklabToLinearRgb(
  L: number,
  a: number,
  b: number,
  out: [number, number, number]
): void {
  const l_ = 0.9999999984505196 * L + 0.3963377921737678 * a + 0.21580375806075877 * b;
  const m_ = 1.0000000088817607 * L - 0.10556134232365633 * a - 0.0638541747717059 * b;
  const s_ = 1.0000000546724108 * L - 0.08948418209496574 * a - 1.2914855378640917 * b;

  const l = l_ * l_ * l_;
  const m = m_ * m_ * m_;
  const s = s_ * s_ * s_;

  out[0] = 4.076741661347994 * l - 3.3077115904081933 * m + 0.23096992872942793 * s;
  out[1] = -1.2684380040921763 * l + 2.6097574006633715 * m - 0.3413193963102196 * s;
  out[2] = -0.004196086541837074 * l - 0.7034186144594495 * m + 1.7076147009309446 * s;
}
