export type IdtName = 'srgb' | 'rec709' | 'apple-log' | 'slog3' | 'vlog' | 'clog3';

export type TransferMode = 'balanced' | 'creative';

export type LutSize = 17 | 33 | 65;

export type PixelBuffer = Uint8ClampedArray | Uint8Array | Float32Array;

export interface ImageDataLike {
  width: number;
  height: number;
  data: PixelBuffer;
}

export interface LutSolveDiagnostics {
  solveMs: number;
  sourceSamples: number;
  referenceSamples: number;
  inversionCount: number;
  inversionRate: number;
}

export interface Lut3D {
  size: LutSize;
  domainMin: [number, number, number];
  domainMax: [number, number, number];
  data: Float32Array;
  title?: string;
  diagnostics?: LutSolveDiagnostics;
}

export interface FitLutOptions {
  references: ImageDataLike[];
  source: ImageDataLike | ImageDataLike[];
  size?: LutSize;
  strength?: number;
  skinProtect?: number;
  highlightRolloff?: number;
  shadowLift?: number;
  smoothness?: number;
  mode?: TransferMode;
  idt?: IdtName;
  title?: string;
}

export interface LutInspection {
  inversionCount: number;
  inversionRate: number;
  laplacianRoughness: number;
  maxStep: number;
  clippingFraction: number;
}

export interface HaldClutImage {
  width: 512;
  height: 512;
  data: Uint8ClampedArray;
}
