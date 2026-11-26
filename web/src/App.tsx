import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  fitLut,
  generateHaldClut,
  serializeCube,
  type IdtName,
  type ImageDataLike,
  type Lut3D,
  type LutSize,
  type TransferMode,
} from 'lut-match-core';
import { Viewport } from './components/Viewport';
import { Scopes, type ScopeMode } from './components/Scopes';
import {
  loadReferencePresets,
  loadSourceSamples,
  type SampleAsset,
} from './samples';

interface RefEntry {
  id: string;
  label: string;
  image: ImageDataLike;
  previewUrl: string;
}

const IDT_OPTIONS: { value: IdtName; label: string }[] = [
  { value: 'srgb', label: 'sRGB / Rec.709 (Working / Resolve CST)' },
  { value: 'rec709', label: 'Rec.709 (Broadcast Gamma → Working)' },
  { value: 'apple-log', label: 'Apple Log → Rec.709' },
  { value: 'slog3', label: 'Sony S-Log3 → Rec.709' },
  { value: 'vlog', label: 'Panasonic V-Log → Rec.709' },
  { value: 'clog3', label: 'Canon C-Log3 → Rec.709' },
];

const MAX_REFS = 5;

async function decodeImageFile(file: File, maxDim = 1280): Promise<SampleAsset> {
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.src = url;
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error(`Failed to decode ${file.name}`));
    });

    const scale = Math.min(1, maxDim / Math.max(img.naturalWidth, img.naturalHeight));
    const width = Math.max(1, Math.round(img.naturalWidth * scale));
    const height = Math.max(1, Math.round(img.naturalHeight * scale));

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas context unavailable');
    ctx.drawImage(img, 0, 0, width, height);
    const imgData = ctx.getImageData(0, 0, width, height);

    return {
      id: `upload-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      label: file.name.replace(/\.[^.]+$/, ''),
      image: { width, height, data: imgData.data },
      previewUrl: canvas.toDataURL('image/jpeg', 0.86),
    };
  } finally {
    URL.revokeObjectURL(url);
  }
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const EMPTY_IMAGE: ImageDataLike = {
  width: 2,
  height: 2,
  data: new Uint8ClampedArray([
    128, 128, 128, 255, 128, 128, 128, 255,
    128, 128, 128, 255, 128, 128, 128, 255,
  ]),
};

const LOADING_ASSET: SampleAsset = {
  id: 'loading',
  label: 'Loading',
  image: EMPTY_IMAGE,
  previewUrl: '',
};

export function App() {
  const [sourceSamples, setSourceSamples] = useState<SampleAsset[]>([]);
  const [refPresets, setRefPresets] = useState<SampleAsset[]>([]);
  const [sourceAsset, setSourceAsset] = useState<SampleAsset>(LOADING_ASSET);
  const [references, setReferences] = useState<RefEntry[]>([]);

  const [idt, setIdt] = useState<IdtName>('srgb');
  const [mode, setMode] = useState<TransferMode>('balanced');
  const [gridSize, setGridSize] = useState<LutSize>(33);
  const [strength, setStrength] = useState(1.0);
  const [skinProtect, setSkinProtect] = useState(0.5);
  const [highlightRolloff, setHighlightRolloff] = useState(0.10);
  const [shadowLift, setShadowLift] = useState(0.0);
  const [smoothness, setSmoothness] = useState(0.35);

  const [splitPos, setSplitPos] = useState(0.5);
  const [comparingOriginal, setComparingOriginal] = useState(false);
  const [scopeMode, setScopeMode] = useState<ScopeMode>('vectorscope');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const sourceInputRef = useRef<HTMLInputElement | null>(null);
  const refInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    let active = true;
    Promise.all([loadSourceSamples(), loadReferencePresets()])
      .then(([srcs, refs]) => {
        if (!active) return;
        setSourceSamples(srcs);
        setRefPresets(refs);
        if (srcs[0]) setSourceAsset((prev) => (prev.id === 'loading' ? srcs[0]! : prev));
        if (refs[0]) setReferences((prev) => (prev.length === 0 ? [refs[0]!] : prev));
      })
      .catch((err) => {
        if (!active) return;
        setErrorMsg(err instanceof Error ? err.message : 'Failed to load Kodak samples');
      });
    return () => {
      active = false;
    };
  }, []);

  const { lut, inversionCount, inversionRate } = useMemo(() => {
    const activeRefs =
      references.length > 0 ? references : refPresets[0] ? [refPresets[0]] : [LOADING_ASSET];
    const solved: Lut3D = fitLut({
      source: sourceAsset.image,
      references: activeRefs.map((r) => r.image),
      size: gridSize,
      strength,
      skinProtect,
      highlightRolloff,
      shadowLift,
      smoothness,
      mode,
      idt,
      title: `lut-match ${sourceAsset.label}`,
    });
    return {
      lut: solved,
      inversionCount: solved.diagnostics?.inversionCount ?? 0,
      inversionRate: solved.diagnostics?.inversionRate ?? 0,
    };
  }, [
    sourceAsset,
    references,
    refPresets,
    gridSize,
    strength,
    skinProtect,
    highlightRolloff,
    shadowLift,
    smoothness,
    mode,
    idt,
  ]);

  const handleLoadSourceFile = useCallback(async (file: File) => {
    try {
      setErrorMsg(null);
      const asset = await decodeImageFile(file);
      setSourceAsset(asset);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Failed to load source image');
    }
  }, []);

  const handleAddReferenceFiles = useCallback(async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    try {
      setErrorMsg(null);
      const decoded: RefEntry[] = [];
      for (const file of Array.from(files).slice(0, MAX_REFS)) {
        decoded.push(await decodeImageFile(file, 640));
      }
      setReferences((prev) => [...prev, ...decoded].slice(0, MAX_REFS));
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Failed to load reference image');
    }
  }, []);

  const handleTogglePresetRef = useCallback((preset: SampleAsset) => {
    setReferences((prev) => {
      const exists = prev.some((r) => r.id === preset.id);
      if (exists) {
        if (prev.length === 1) return prev;
        return prev.filter((r) => r.id !== preset.id);
      }
      if (prev.length >= MAX_REFS) {
        return [...prev.slice(1), preset];
      }
      return [...prev, preset];
    });
  }, []);

  const handleDownloadCube = useCallback(() => {
    const text = serializeCube(lut);
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    triggerDownload(blob, `lut-match-${gridSize}-${idt}.cube`);
  }, [lut, gridSize, idt]);

  const handleDownloadHald = useCallback(() => {
    const hald = generateHaldClut(lut);
    const canvas = document.createElement('canvas');
    canvas.width = hald.width;
    canvas.height = hald.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.putImageData(
      new ImageData(new Uint8ClampedArray(hald.data), hald.width, hald.height),
      0,
      0
    );
    canvas.toBlob((blob) => {
      if (blob) triggerDownload(blob, `lut-match-hald8-${idt}.png`);
    }, 'image/png');
  }, [lut, idt]);

  // Keyboard hold-to-compare (\)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === '\\' && !e.repeat) {
        setComparingOriginal(true);
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key === '\\') {
        setComparingOriginal(false);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, []);

  const solveMs = Math.round(lut.diagnostics?.solveMs ?? 0);

  return (
    <div className="studio-shell">
      <header className="topbar">
        <div className="topbar-left">
          <span className="brand">lut-match</span>
          <span className="health-readout" aria-live="polite">
            {inversionCount} {inversionCount === 1 ? 'inversion' : 'inversions'} ({(inversionRate * 100).toFixed(2)}%) · {gridSize}³ · {solveMs} ms
          </span>
        </div>
        <div className="topbar-actions">
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleDownloadCube}
          >
            Download .cube
          </button>
          <button type="button" className="btn" onClick={handleDownloadHald}>
            Download HaldCLUT .png
          </button>
        </div>
      </header>

      <main className="workspace">
        <section className="stage-column" aria-label="Viewport and scopes">
          <div className="viewport-bar">
            <div className="viewport-bar-group">
              <button
                type="button"
                className={`btn ${comparingOriginal ? 'is-active' : ''}`}
                onPointerDown={() => setComparingOriginal(true)}
                onPointerUp={() => setComparingOriginal(false)}
                onPointerLeave={() => setComparingOriginal(false)}
                onPointerCancel={() => setComparingOriginal(false)}
                onKeyDown={(e) => {
                  if (e.key === ' ' || e.key === 'Enter') {
                    e.preventDefault();
                    setComparingOriginal(true);
                  }
                }}
                onKeyUp={(e) => {
                  if (e.key === ' ' || e.key === 'Enter') {
                    setComparingOriginal(false);
                  }
                }}
              >
                Hold Original
              </button>
              <button
                type="button"
                className="btn"
                onClick={() => setSplitPos(0.5)}
              >
                50 / 50
              </button>
            </div>
            <span className="viewport-hint">
              Drag wipe or press \ to compare · Drop image to load
            </span>
          </div>

          <Viewport
            source={sourceAsset.image}
            lut={lut}
            splitPos={splitPos}
            onSplitChange={setSplitPos}
            comparingOriginal={comparingOriginal}
            onDropImageFile={handleLoadSourceFile}
          />

          <Scopes
            source={sourceAsset.image}
            lut={lut}
            comparingOriginal={comparingOriginal}
            mode={scopeMode}
            onModeChange={setScopeMode}
          />
        </section>

        <aside className="control-rail" aria-label="Grading controls">
          {errorMsg && (
            <div className="error-banner" role="alert">
              {errorMsg}
            </div>
          )}

          <div className="rail-section">
            <div className="section-head">
              <span className="section-title">Source Frame</span>
              <button
                type="button"
                className="btn"
                onClick={() => sourceInputRef.current?.click()}
              >
                Load Image
              </button>
              <input
                ref={sourceInputRef}
                type="file"
                accept="image/png,image/jpeg,image/webp"
                className="sr-only-input"
                aria-label="Upload source image"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) void handleLoadSourceFile(file);
                  e.target.value = '';
                }}
              />
            </div>
            <div className="chip-row" role="group" aria-label="Source samples">
              {sourceSamples.map((sample) => (
                <button
                  key={sample.id}
                  type="button"
                  className={`chip-btn ${sourceAsset.id === sample.id ? 'is-active' : ''}`}
                  onClick={() => setSourceAsset(sample)}
                >
                  {sample.label}
                </button>
              ))}
            </div>
          </div>

          <div className="rail-section">
            <div className="section-head">
              <span className="section-title">
                Reference Stills ({references.length}/{MAX_REFS})
              </span>
              <button
                type="button"
                className="btn"
                disabled={references.length >= MAX_REFS}
                onClick={() => refInputRef.current?.click()}
              >
                Add Still
              </button>
              <input
                ref={refInputRef}
                type="file"
                accept="image/png,image/jpeg,image/webp"
                multiple
                className="sr-only-input"
                aria-label="Upload reference stills"
                onChange={(e) => {
                  void handleAddReferenceFiles(e.target.files);
                  e.target.value = '';
                }}
              />
            </div>

            <div className="ref-tray">
              {references.map((ref) => (
                <div key={ref.id} className="ref-thumb" title={ref.label}>
                  <img src={ref.previewUrl} alt={ref.label} />
                  {references.length > 1 && (
                    <button
                      type="button"
                      className="ref-remove"
                      aria-label={`Remove ${ref.label}`}
                      onClick={() =>
                        setReferences((prev) => prev.filter((item) => item.id !== ref.id))
                      }
                    >
                      ×
                    </button>
                  )}
                </div>
              ))}
            </div>

            <div className="chip-row" role="group" aria-label="Built-in film looks">
              {refPresets.map((preset) => {
                const active = references.some((r) => r.id === preset.id);
                return (
                  <button
                    key={preset.id}
                    type="button"
                    className={`chip-btn ${active ? 'is-active' : ''}`}
                    onClick={() => handleTogglePresetRef(preset)}
                  >
                    {preset.label}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="rail-section">
            <div className="field-col">
              <label
                className="field-label"
                htmlFor="idt-select"
                title="Use sRGB / Rec.709 for display-referred stills or DaVinci Resolve CST working nodes; pick a Camera Log IDT to bake Log-to-Rec.709 into the .cube lattice."
              >
                Input Color Space (IDT)
              </label>
              <select
                id="idt-select"
                className="select-input"
                value={idt}
                title="Display/Resolve CST working space or Camera Log IDT baked directly into the exported 3D LUT"
                onChange={(e) => setIdt(e.target.value as IdtName)}
              >
                {IDT_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="field-grid field-grid-even">
              <div className="field-col">
                <span
                  className="field-label"
                  title="Balanced: Empirical-Bayes shrinkage for multi-still & safe cross-scene accuracy. Creative: expressive single-still film look transfer with skin protection."
                >
                  Transfer Mode
                </span>
                <div className="segmented segmented-fill" role="group" aria-label="Transfer mode">
                  {(['balanced', 'creative'] as const).map((m) => (
                    <button
                      key={m}
                      type="button"
                      aria-pressed={mode === m}
                      className={`segmented-btn ${mode === m ? 'is-active' : ''}`}
                      onClick={() => setMode(m)}
                    >
                      {m === 'balanced' ? 'Balanced' : 'Creative'}
                    </button>
                  ))}
                </div>
              </div>
              <div className="field-col">
                <span className="field-label">Grid Size</span>
                <div className="segmented segmented-fill" role="group" aria-label="LUT grid size">
                  {([17, 33, 65] as const).map((sz) => (
                    <button
                      key={sz}
                      type="button"
                      aria-pressed={gridSize === sz}
                      className={`segmented-btn ${gridSize === sz ? 'is-active' : ''}`}
                      onClick={() => setGridSize(sz)}
                    >
                      {sz}³
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="rail-section">
            <SliderControl
              id="slider-strength"
              label="Strength"
              value={strength}
              min={0}
              max={1}
              step={0.01}
              defaultValue={1.0}
              format={(v) => `${Math.round(v * 100)}%`}
              onChange={setStrength}
            />
            <SliderControl
              id="slider-skin"
              label="Skin Protection"
              value={skinProtect}
              min={0}
              max={1}
              step={0.01}
              defaultValue={0.5}
              format={(v) => `${Math.round(v * 100)}%`}
              onChange={setSkinProtect}
            />
            <SliderControl
              id="slider-rolloff"
              label="Highlight Rolloff"
              value={highlightRolloff}
              min={0}
              max={1}
              step={0.01}
              defaultValue={0.10}
              format={(v) => `${Math.round(v * 100)}%`}
              onChange={setHighlightRolloff}
            />
            <SliderControl
              id="slider-shadow"
              label="Shadow Lift"
              value={shadowLift}
              min={-0.25}
              max={0.25}
              step={0.01}
              defaultValue={0.0}
              format={(v) => `${v >= 0 ? '+' : ''}${Math.round(v * 100)}%`}
              onChange={setShadowLift}
            />
            <SliderControl
              id="slider-smooth"
              label="Smoothness"
              value={smoothness}
              min={0}
              max={1}
              step={0.01}
              defaultValue={0.35}
              format={(v) => `${Math.round(v * 100)}%`}
              onChange={setSmoothness}
            />
          </div>
        </aside>
      </main>
    </div>
  );
}

interface SliderControlProps {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
  format: (v: number) => string;
  onChange: (v: number) => void;
}

function SliderControl({
  id,
  label,
  value,
  min,
  max,
  step,
  defaultValue,
  format,
  onChange,
}: SliderControlProps) {
  return (
    <div className="slider-row">
      <div className="slider-head">
        <label className="slider-label" htmlFor={id}>
          {label}
        </label>
        <span className="slider-value">{format(value)}</span>
      </div>
      <input
        id={id}
        type="range"
        className="range-input"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        onDoubleClick={() => onChange(defaultValue)}
      />
    </div>
  );
}
