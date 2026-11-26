import { useCallback, useEffect, useRef } from 'react';
import { evaluateLutPoint, type ImageDataLike, type Lut3D } from 'lut-match-core';

export type ScopeMode = 'vectorscope' | 'waveform' | 'parade';

interface ScopesProps {
  source: ImageDataLike;
  lut: Lut3D;
  comparingOriginal: boolean;
  mode: ScopeMode;
  onModeChange: (mode: ScopeMode) => void;
}

const SAMPLE_W = 160;
const SAMPLE_H = 100;

function sampleGradedGrid(
  source: ImageDataLike,
  lut: Lut3D,
  skipLut: boolean
): Float32Array {
  const out = new Float32Array(SAMPLE_W * SAMPLE_H * 3);
  const srcW = source.width;
  const srcH = source.height;
  const data = source.data;
  const isFloat = data instanceof Float32Array;
  const rgb: [number, number, number] = [0, 0, 0];

  for (let y = 0; y < SAMPLE_H; y++) {
    const sy = Math.min(srcH - 1, Math.floor((y / SAMPLE_H) * srcH));
    for (let x = 0; x < SAMPLE_W; x++) {
      const sx = Math.min(srcW - 1, Math.floor((x / SAMPLE_W) * srcW));
      const sp = (sy * srcW + sx) * 4;
      const r = isFloat ? data[sp]! : data[sp]! / 255;
      const g = isFloat ? data[sp + 1]! : data[sp + 1]! / 255;
      const b = isFloat ? data[sp + 2]! : data[sp + 2]! / 255;

      const dp = (y * SAMPLE_W + x) * 3;
      if (skipLut) {
        out[dp] = r;
        out[dp + 1] = g;
        out[dp + 2] = b;
      } else {
        evaluateLutPoint(lut, r, g, b, rgb);
        out[dp] = rgb[0];
        out[dp + 1] = rgb[1];
        out[dp + 2] = rgb[2];
      }
    }
  }
  return out;
}

function drawVectorscope(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  pixels: Float32Array
): void {
  ctx.fillStyle = '#0d0e10';
  ctx.fillRect(0, 0, w, h);

  const cx = w * 0.5;
  const cy = h * 0.5;
  const radius = Math.min(w, h) * 0.43;

  // Graticule rings
  ctx.strokeStyle = 'rgba(156, 163, 175, 0.16)';
  ctx.lineWidth = 1;
  for (const frac of [0.25, 0.5, 0.75, 1.0]) {
    ctx.beginPath();
    ctx.arc(cx, cy, radius * frac, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(cx - radius, cy);
  ctx.lineTo(cx + radius, cy);
  ctx.moveTo(cx, cy - radius);
  ctx.lineTo(cx, cy + radius);
  ctx.stroke();

  // Skin-tone I-line reticle (~123 deg in Cb/Cr plane)
  const iRad = (123 * Math.PI) / 180;
  ctx.strokeStyle = 'rgba(217, 176, 134, 0.55)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx + Math.cos(iRad) * radius, cy - Math.sin(iRad) * radius);
  ctx.stroke();
  ctx.setLineDash([]);

  // Primary target dots (R, Yl, G, Cy, B, Mg)
  const targets: [number, number, number, string][] = [
    [1, 0, 0, 'R'],
    [1, 1, 0, 'Yl'],
    [0, 1, 0, 'G'],
    [0, 1, 1, 'Cy'],
    [0, 0, 1, 'B'],
    [1, 0, 1, 'Mg'],
  ];
  ctx.font = '9px ui-monospace, monospace';
  ctx.fillStyle = 'rgba(156, 163, 175, 0.55)';
  for (const [tr, tg, tb, label] of targets) {
    const cb = -0.168736 * tr - 0.331264 * tg + 0.5 * tb;
    const cr = 0.5 * tr - 0.418688 * tg - 0.081312 * tb;
    const px = cx + cb * 2 * radius * 0.88;
    const py = cy - cr * 2 * radius * 0.88;
    ctx.strokeRect(px - 3, py - 3, 6, 6);
    ctx.fillText(label, px + 5, py + 3);
  }

  // Plot pixel chroma cloud
  const img = ctx.getImageData(0, 0, w, h);
  const buf = img.data;
  const total = SAMPLE_W * SAMPLE_H;
  for (let i = 0; i < total; i++) {
    const p = i * 3;
    const r = pixels[p]!;
    const g = pixels[p + 1]!;
    const b = pixels[p + 2]!;
    const cb = -0.168736 * r - 0.331264 * g + 0.5 * b;
    const cr = 0.5 * r - 0.418688 * g - 0.081312 * b;
    const px = Math.round(cx + cb * 2 * radius * 0.88);
    const py = Math.round(cy - cr * 2 * radius * 0.88);
    if (px >= 1 && px < w - 1 && py >= 1 && py < h - 1) {
      const addR = Math.round(48 + r * 110);
      const addG = Math.round(48 + g * 110);
      const addB = Math.round(48 + b * 110);
      for (const [ox, oy] of [
        [0, 0],
        [1, 0],
        [0, 1],
      ] as const) {
        const idx = ((py + oy) * w + (px + ox)) * 4;
        const scale = ox === 0 && oy === 0 ? 1 : 0.45;
        buf[idx] = Math.min(255, buf[idx]! + Math.round(addR * scale));
        buf[idx + 1] = Math.min(255, buf[idx + 1]! + Math.round(addG * scale));
        buf[idx + 2] = Math.min(255, buf[idx + 2]! + Math.round(addB * scale));
        buf[idx + 3] = 255;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}

function drawWaveform(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  pixels: Float32Array
): void {
  ctx.fillStyle = '#0d0e10';
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = 'rgba(156, 163, 175, 0.14)';
  ctx.lineWidth = 1;
  for (const ire of [0.25, 0.5, 0.75]) {
    const y = Math.round(h - 8 - ire * (h - 16)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  const img = ctx.getImageData(0, 0, w, h);
  const buf = img.data;
  const plotH = h - 16;

  for (let sy = 0; sy < SAMPLE_H; sy++) {
    for (let sx = 0; sx < SAMPLE_W; sx++) {
      const p = (sy * SAMPLE_W + sx) * 3;
      const r = pixels[p]!;
      const g = pixels[p + 1]!;
      const b = pixels[p + 2]!;
      const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      const px = Math.min(w - 1, Math.floor((sx / SAMPLE_W) * w));
      const py = Math.max(0, Math.min(h - 1, Math.round(h - 8 - luma * plotH)));
      const idx = (py * w + px) * 4;
      buf[idx] = Math.min(255, buf[idx]! + 22);
      buf[idx + 1] = Math.min(255, buf[idx + 1]! + 26);
      buf[idx + 2] = Math.min(255, buf[idx + 2]! + 24);
      buf[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function drawParade(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  pixels: Float32Array
): void {
  ctx.fillStyle = '#0d0e10';
  ctx.fillRect(0, 0, w, h);

  const third = Math.floor(w / 3);
  ctx.strokeStyle = 'rgba(156, 163, 175, 0.14)';
  ctx.lineWidth = 1;
  for (const ire of [0.25, 0.5, 0.75]) {
    const y = Math.round(h - 8 - ire * (h - 16)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(third + 0.5, 0);
  ctx.lineTo(third + 0.5, h);
  ctx.moveTo(third * 2 + 0.5, 0);
  ctx.lineTo(third * 2 + 0.5, h);
  ctx.stroke();

  const img = ctx.getImageData(0, 0, w, h);
  const buf = img.data;
  const plotH = h - 16;

  for (let sy = 0; sy < SAMPLE_H; sy++) {
    for (let sx = 0; sx < SAMPLE_W; sx++) {
      const p = (sy * SAMPLE_W + sx) * 3;
      const localX = Math.min(third - 2, Math.floor((sx / SAMPLE_W) * (third - 2)));

      const pr = Math.max(0, Math.min(h - 1, Math.round(h - 8 - pixels[p]! * plotH)));
      const ir = (pr * w + localX) * 4;
      buf[ir] = Math.min(255, buf[ir]! + 46);
      buf[ir + 1] = Math.min(255, buf[ir + 1]! + 12);
      buf[ir + 2] = Math.min(255, buf[ir + 2]! + 12);
      buf[ir + 3] = 255;

      const pg = Math.max(0, Math.min(h - 1, Math.round(h - 8 - pixels[p + 1]! * plotH)));
      const ig = (pg * w + third + localX) * 4;
      buf[ig] = Math.min(255, buf[ig]! + 14);
      buf[ig + 1] = Math.min(255, buf[ig + 1]! + 46);
      buf[ig + 2] = Math.min(255, buf[ig + 2]! + 16);
      buf[ig + 3] = 255;

      const pb = Math.max(0, Math.min(h - 1, Math.round(h - 8 - pixels[p + 2]! * plotH)));
      const ib = (pb * w + third * 2 + localX) * 4;
      buf[ib] = Math.min(255, buf[ib]! + 14);
      buf[ib + 1] = Math.min(255, buf[ib + 1]! + 24);
      buf[ib + 2] = Math.min(255, buf[ib + 2]! + 48);
      buf[ib + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

export function Scopes({
  source,
  lut,
  comparingOriginal,
  mode,
  onModeChange,
}: ScopesProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const renderScope = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(240, Math.round(rect.width || 480));
    const h = Math.max(120, Math.round(rect.height || 164));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const graded = sampleGradedGrid(source, lut, comparingOriginal);
    if (mode === 'vectorscope') {
      drawVectorscope(ctx, w, h, graded);
    } else if (mode === 'waveform') {
      drawWaveform(ctx, w, h, graded);
    } else {
      drawParade(ctx, w, h, graded);
    }
  }, [source, lut, comparingOriginal, mode]);

  useEffect(() => {
    renderScope();
  }, [renderScope]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(() => renderScope());
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [renderScope]);

  return (
    <section className="scopes-panel" aria-label="Color scopes">
      <div className="scopes-header">
        <div className="segmented" role="tablist" aria-label="Scope mode">
          {(
            [
              ['vectorscope', 'Vectorscope'],
              ['waveform', 'Luma Waveform'],
              ['parade', 'RGB Parade'],
            ] as const
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={mode === id}
              className={`segmented-btn ${mode === id ? 'is-active' : ''}`}
              onClick={() => onModeChange(id)}
            >
              {label}
            </button>
          ))}
        </div>
        <span className="scope-meta">
          {mode === 'vectorscope' ? 'Skin I-line 123°' : 'Rec.709 0–100 IRE'}
        </span>
      </div>
      <div className="scope-canvas-wrap">
        <canvas ref={canvasRef} className="scope-canvas" />
      </div>
    </section>
  );
}
