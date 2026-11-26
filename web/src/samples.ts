import type { ImageDataLike } from 'lut-match-core';

export interface SampleAsset {
  id: string;
  label: string;
  image: ImageDataLike;
  previewUrl: string;
}

export interface SampleSpec {
  id: string;
  label: string;
  path: string;
}

const BASE = import.meta.env.BASE_URL ?? '/';

function assetUrl(relPath: string): string {
  return `${BASE.replace(/\/$/, '')}/${relPath.replace(/^\//, '')}`;
}

export const SOURCE_SAMPLE_SPECS: readonly SampleSpec[] = [
  { id: 'src-portrait', label: 'Portrait', path: 'samples/src-portrait.jpg' },
  { id: 'src-foliage', label: 'Motocross', path: 'samples/src-foliage.jpg' },
  { id: 'src-coastal', label: 'Coastal', path: 'samples/src-coastal.jpg' },
  { id: 'src-parrots', label: 'Parrots', path: 'samples/src-parrots.jpg' },
];

export const REFERENCE_PRESET_SPECS: readonly SampleSpec[] = [
  { id: 'ref-kodak-2383', label: 'Kodak 2383 Warm', path: 'samples/ref-kodak-2383.jpg' },
  { id: 'ref-teal-orange', label: 'Teal & Orange', path: 'samples/ref-teal-orange.jpg' },
  { id: 'ref-fuji-eterna', label: 'Fuji Eterna Cool', path: 'samples/ref-fuji-eterna.jpg' },
  { id: 'ref-golden-hour', label: 'Golden Hour', path: 'samples/ref-golden-hour.jpg' },
  { id: 'ref-cross-process', label: 'Cross-Process', path: 'samples/ref-cross-process.jpg' },
];

async function decodeUrlAsset(spec: SampleSpec): Promise<SampleAsset> {
  const url = assetUrl(spec.path);
  const img = new Image();
  img.decoding = 'async';
  img.src = url;
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error(`Failed to load sample ${spec.label}`));
  });

  const width = img.naturalWidth;
  const height = img.naturalHeight;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas context unavailable');
  ctx.drawImage(img, 0, 0, width, height);
  const imgData = ctx.getImageData(0, 0, width, height);

  return {
    id: spec.id,
    label: spec.label,
    image: { width, height, data: new Uint8ClampedArray(imgData.data) },
    previewUrl: url,
  };
}

export async function loadSourceSamples(): Promise<SampleAsset[]> {
  return Promise.all(SOURCE_SAMPLE_SPECS.map(decodeUrlAsset));
}

export async function loadReferencePresets(): Promise<SampleAsset[]> {
  return Promise.all(REFERENCE_PRESET_SPECS.map(decodeUrlAsset));
}
