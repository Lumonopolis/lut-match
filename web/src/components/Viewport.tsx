import { useEffect, useRef, useState } from 'react';
import { applyLutCpu, type ImageDataLike, type Lut3D } from 'lut-match-core';

interface ViewportProps {
  source: ImageDataLike;
  lut: Lut3D;
  splitPos: number;
  onSplitChange: (pos: number) => void;
  comparingOriginal: boolean;
  onDropImageFile: (file: File) => void;
}

const f32Buf = new Float32Array(1);
const u32Buf = new Uint32Array(f32Buf.buffer);

function floatToHalf(val: number): number {
  f32Buf[0] = val;
  const x = u32Buf[0]!;
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  let mant = (x >> 13) & 0x3ff;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant = ((x & 0x7fffff) | 0x800000) >> (1 - exp + 13);
    return sign | mant;
  }
  if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | mant;
}

const VERT_SRC = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FRAG_3D_SRC = `#version 300 es
precision highp float;
precision highp sampler3D;

uniform sampler2D uSource;
uniform sampler3D uLut;
uniform float uLutSize;
uniform float uSplit;
uniform int uShowOriginal;

in vec2 vUv;
out vec4 fragColor;

void main() {
  vec2 texUv = vec2(vUv.x, 1.0 - vUv.y);
  vec4 src = texture(uSource, texUv);
  if (uShowOriginal == 1 || vUv.x < uSplit) {
    fragColor = vec4(src.rgb, 1.0);
    return;
  }
  vec3 uvw = (clamp(src.rgb, 0.0, 1.0) * (uLutSize - 1.0) + 0.5) / uLutSize;
  vec3 graded = texture(uLut, uvw).rgb;
  fragColor = vec4(graded, 1.0);
}`;

const VERT_FALLBACK_SRC = `
attribute vec2 aPos;
varying vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FRAG_ATLAS_SRC = `
precision highp float;
uniform sampler2D uSource;
uniform sampler2D uAtlas;
uniform float uLutSize;
uniform float uSplit;
uniform int uShowOriginal;
varying vec2 vUv;

vec3 sampleAtlas(vec3 rgb) {
  float n = uLutSize;
  float slice = clamp(rgb.b, 0.0, 1.0) * (n - 1.0);
  float b0 = floor(slice);
  float b1 = min(n - 1.0, b0 + 1.0);
  float f = slice - b0;

  float uR = (clamp(rgb.r, 0.0, 1.0) * (n - 1.0) + 0.5) / (n * n);
  float vG = (clamp(rgb.g, 0.0, 1.0) * (n - 1.0) + 0.5) / n;

  vec3 c0 = texture2D(uAtlas, vec2(b0 / n + uR, vG)).rgb;
  vec3 c1 = texture2D(uAtlas, vec2(b1 / n + uR, vG)).rgb;
  return mix(c0, c1, f);
}

void main() {
  vec2 texUv = vec2(vUv.x, 1.0 - vUv.y);
  vec4 src = texture2D(uSource, texUv);
  if (uShowOriginal == 1 || vUv.x < uSplit) {
    gl_FragColor = vec4(src.rgb, 1.0);
    return;
  }
  gl_FragColor = vec4(sampleAtlas(src.rgb), 1.0);
}`;

interface GlState {
  mode: 'webgl2' | 'webgl1-atlas';
  gl: WebGL2RenderingContext | WebGLRenderingContext;
  program: WebGLProgram;
  quadBuffer: WebGLBuffer | null;
  srcTex: WebGLTexture;
  lutTex: WebGLTexture;
  maxTexSize: number;
  uSource: WebGLUniformLocation | null;
  uLut: WebGLUniformLocation | null;
  uLutSize: WebGLUniformLocation | null;
  uSplit: WebGLUniformLocation | null;
  uShowOriginal: WebGLUniformLocation | null;
}

function compileShader(
  gl: WebGLRenderingContext | WebGL2RenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function createProgram(
  gl: WebGLRenderingContext | WebGL2RenderingContext,
  vsSrc: string,
  fsSrc: string
): WebGLProgram | null {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  if (!vs || !fs) return null;
  const prog = gl.createProgram();
  if (!prog) return null;
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    gl.deleteProgram(prog);
    return null;
  }
  return prog;
}

function cleanupGlState(state: GlState | null): void {
  if (!state) return;
  const { gl, srcTex, lutTex, quadBuffer, program } = state;
  gl.deleteTexture(srcTex);
  gl.deleteTexture(lutTex);
  if (quadBuffer) gl.deleteBuffer(quadBuffer);
  gl.deleteProgram(program);
}

function initGl(canvas: HTMLCanvasElement): GlState | null {
  const gl2 = canvas.getContext('webgl2', {
    antialias: false,
    alpha: false,
    preserveDrawingBuffer: true,
  });
  if (gl2) {
    const prog = createProgram(gl2, VERT_SRC, FRAG_3D_SRC);
    if (prog) {
      const quad = gl2.createBuffer();
      gl2.bindBuffer(gl2.ARRAY_BUFFER, quad);
      gl2.bufferData(
        gl2.ARRAY_BUFFER,
        new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
        gl2.STATIC_DRAW
      );
      const loc = gl2.getAttribLocation(prog, 'aPos');
      gl2.enableVertexAttribArray(loc);
      gl2.vertexAttribPointer(loc, 2, gl2.FLOAT, false, 0, 0);

      const srcTex = gl2.createTexture();
      const lutTex = gl2.createTexture();
      if (srcTex && lutTex) {
        return {
          mode: 'webgl2',
          gl: gl2,
          program: prog,
          quadBuffer: quad,
          srcTex,
          lutTex,
          maxTexSize: Number(gl2.getParameter(gl2.MAX_TEXTURE_SIZE)) || 4096,
          uSource: gl2.getUniformLocation(prog, 'uSource'),
          uLut: gl2.getUniformLocation(prog, 'uLut'),
          uLutSize: gl2.getUniformLocation(prog, 'uLutSize'),
          uSplit: gl2.getUniformLocation(prog, 'uSplit'),
          uShowOriginal: gl2.getUniformLocation(prog, 'uShowOriginal'),
        };
      }
      if (srcTex) gl2.deleteTexture(srcTex);
      if (lutTex) gl2.deleteTexture(lutTex);
      if (quad) gl2.deleteBuffer(quad);
      gl2.deleteProgram(prog);
    }
    // Do not attempt 'webgl' on a canvas that already acquired 'webgl2'
    return null;
  }

  const gl1 = canvas.getContext('webgl', {
    antialias: false,
    alpha: false,
    preserveDrawingBuffer: true,
  });
  if (gl1) {
    const prog = createProgram(gl1, VERT_FALLBACK_SRC, FRAG_ATLAS_SRC);
    if (prog) {
      const quad = gl1.createBuffer();
      gl1.bindBuffer(gl1.ARRAY_BUFFER, quad);
      gl1.bufferData(
        gl1.ARRAY_BUFFER,
        new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
        gl1.STATIC_DRAW
      );
      const loc = gl1.getAttribLocation(prog, 'aPos');
      gl1.enableVertexAttribArray(loc);
      gl1.vertexAttribPointer(loc, 2, gl1.FLOAT, false, 0, 0);

      const srcTex = gl1.createTexture();
      const lutTex = gl1.createTexture();
      if (srcTex && lutTex) {
        return {
          mode: 'webgl1-atlas',
          gl: gl1,
          program: prog,
          quadBuffer: quad,
          srcTex,
          lutTex,
          maxTexSize: Number(gl1.getParameter(gl1.MAX_TEXTURE_SIZE)) || 4096,
          uSource: gl1.getUniformLocation(prog, 'uSource'),
          uLut: gl1.getUniformLocation(prog, 'uAtlas'),
          uLutSize: gl1.getUniformLocation(prog, 'uLutSize'),
          uSplit: gl1.getUniformLocation(prog, 'uSplit'),
          uShowOriginal: gl1.getUniformLocation(prog, 'uShowOriginal'),
        };
      }
      if (srcTex) gl1.deleteTexture(srcTex);
      if (lutTex) gl1.deleteTexture(lutTex);
      if (quad) gl1.deleteBuffer(quad);
      gl1.deleteProgram(prog);
    }
  }

  return null;
}

function toRgbaBytes(img: ImageDataLike): Uint8Array {
  if (img.data instanceof Uint8Array || img.data instanceof Uint8ClampedArray) {
    return new Uint8Array(img.data.buffer, img.data.byteOffset, img.width * img.height * 4);
  }
  const out = new Uint8Array(img.width * img.height * 4);
  for (let i = 0; i < out.length; i++) {
    const v = img.data[i] ?? 0;
    out[i] = v <= 0 ? 0 : v >= 1 ? 255 : Math.round(v * 255);
  }
  return out;
}

export function Viewport({
  source,
  lut,
  splitPos,
  onSplitChange,
  comparingOriginal,
  onDropImageFile,
}: ViewportProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const glStateRef = useRef<GlState | null>(null);
  const [draggingOver, setDraggingOver] = useState(false);
  const [rendererMode, setRendererMode] = useState<'WebGL2 3D' | 'WebGL Atlas' | 'CPU 2D'>(
    'WebGL2 3D'
  );
  const [useCpuCanvas, setUseCpuCanvas] = useState(false);
  const draggingSplitRef = useRef(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || useCpuCanvas) return;
    if (!glStateRef.current) {
      const state = initGl(canvas);
      glStateRef.current = state;
      if (!state) {
        setRendererMode('CPU 2D');
        setUseCpuCanvas(true);
      } else if (state.mode === 'webgl1-atlas') {
        if (lut.size * lut.size > state.maxTexSize) {
          cleanupGlState(state);
          glStateRef.current = null;
          setRendererMode('CPU 2D');
          setUseCpuCanvas(true);
        } else {
          setRendererMode('WebGL Atlas');
        }
      } else {
        setRendererMode('WebGL2 3D');
      }
    }
    return () => {
      cleanupGlState(glStateRef.current);
      glStateRef.current = null;
    };
  }, [useCpuCanvas]);

  // Guard WebGL1 atlas against MAX_TEXTURE_SIZE overflow when LUT size changes (e.g., 65^3 -> 4225 > 4096)
  useEffect(() => {
    const state = glStateRef.current;
    if (state && state.mode === 'webgl1-atlas' && lut.size * lut.size > state.maxTexSize) {
      cleanupGlState(state);
      glStateRef.current = null;
      setRendererMode('CPU 2D');
      setUseCpuCanvas(true);
    }
  }, [lut.size]);

  // Upload source image whenever source changes
  useEffect(() => {
    const canvas = canvasRef.current;
    const state = glStateRef.current;
    if (!canvas) return;

    canvas.width = source.width;
    canvas.height = source.height;

    if (!state) return;
    const { gl, srcTex } = state;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, srcTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      source.width,
      source.height,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      toRgbaBytes(source)
    );
  }, [source, useCpuCanvas]);

  // Upload 3D LUT texture whenever solved LUT changes
  useEffect(() => {
    const state = glStateRef.current;
    if (!state) return;

    const size = lut.size;
    if (state.mode === 'webgl1-atlas' && size * size > state.maxTexSize) {
      return;
    }
    const count = size * size * size;
    if (state.mode === 'webgl2') {
      const gl2 = state.gl as WebGL2RenderingContext;
      const rgba16 = new Uint16Array(count * 4);
      const data = lut.data;
      const oneHalf = floatToHalf(1);
      for (let i = 0; i < count; i++) {
        const s = i * 3;
        const d = i * 4;
        rgba16[d] = floatToHalf(data[s]!);
        rgba16[d + 1] = floatToHalf(data[s + 1]!);
        rgba16[d + 2] = floatToHalf(data[s + 2]!);
        rgba16[d + 3] = oneHalf;
      }

      gl2.activeTexture(gl2.TEXTURE1);
      gl2.bindTexture(gl2.TEXTURE_3D, state.lutTex);
      gl2.pixelStorei(gl2.UNPACK_ALIGNMENT, 1);
      gl2.texParameteri(gl2.TEXTURE_3D, gl2.TEXTURE_MIN_FILTER, gl2.LINEAR);
      gl2.texParameteri(gl2.TEXTURE_3D, gl2.TEXTURE_MAG_FILTER, gl2.LINEAR);
      gl2.texParameteri(gl2.TEXTURE_3D, gl2.TEXTURE_WRAP_S, gl2.CLAMP_TO_EDGE);
      gl2.texParameteri(gl2.TEXTURE_3D, gl2.TEXTURE_WRAP_T, gl2.CLAMP_TO_EDGE);
      gl2.texParameteri(gl2.TEXTURE_3D, gl2.TEXTURE_WRAP_R, gl2.CLAMP_TO_EDGE);
      gl2.texImage3D(
        gl2.TEXTURE_3D,
        0,
        gl2.RGBA16F,
        size,
        size,
        size,
        0,
        gl2.RGBA,
        gl2.HALF_FLOAT,
        rgba16
      );
    } else {
      // 2D atlas layout: width = size*size (B slices across X, R within slice), height = size (G)
      const gl1 = state.gl as WebGLRenderingContext;
      const atlasW = size * size;
      const atlasH = size;
      const rgba8 = new Uint8Array(atlasW * atlasH * 4);
      const data = lut.data;
      for (let b = 0; b < size; b++) {
        for (let g = 0; g < size; g++) {
          for (let r = 0; r < size; r++) {
            const srcIdx = ((b * size + g) * size + r) * 3;
            const x = b * size + r;
            const dstIdx = (g * atlasW + x) * 4;
            rgba8[dstIdx] = Math.round((data[srcIdx] ?? 0) * 255);
            rgba8[dstIdx + 1] = Math.round((data[srcIdx + 1] ?? 0) * 255);
            rgba8[dstIdx + 2] = Math.round((data[srcIdx + 2] ?? 0) * 255);
            rgba8[dstIdx + 3] = 255;
          }
        }
      }
      gl1.activeTexture(gl1.TEXTURE1);
      gl1.bindTexture(gl1.TEXTURE_2D, state.lutTex);
      gl1.pixelStorei(gl1.UNPACK_ALIGNMENT, 1);
      gl1.texParameteri(gl1.TEXTURE_2D, gl1.TEXTURE_MIN_FILTER, gl1.LINEAR);
      gl1.texParameteri(gl1.TEXTURE_2D, gl1.TEXTURE_MAG_FILTER, gl1.LINEAR);
      gl1.texParameteri(gl1.TEXTURE_2D, gl1.TEXTURE_WRAP_S, gl1.CLAMP_TO_EDGE);
      gl1.texParameteri(gl1.TEXTURE_2D, gl1.TEXTURE_WRAP_T, gl1.CLAMP_TO_EDGE);
      gl1.texImage2D(
        gl1.TEXTURE_2D,
        0,
        gl1.RGBA,
        atlasW,
        atlasH,
        0,
        gl1.RGBA,
        gl1.UNSIGNED_BYTE,
        rgba8
      );
    }
  }, [lut]);

  // Draw viewport
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const state = glStateRef.current;

    if (state) {
      const { gl, program, srcTex, lutTex } = state;
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.useProgram(program);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, srcTex);
      gl.uniform1i(state.uSource, 0);

      gl.activeTexture(gl.TEXTURE1);
      if (state.mode === 'webgl2') {
        gl.bindTexture((gl as WebGL2RenderingContext).TEXTURE_3D, lutTex);
      } else {
        gl.bindTexture(gl.TEXTURE_2D, lutTex);
      }
      gl.uniform1i(state.uLut, 1);

      gl.uniform1f(state.uLutSize, lut.size);
      gl.uniform1f(state.uSplit, splitPos);
      gl.uniform1i(state.uShowOriginal, comparingOriginal ? 1 : 0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      return;
    }

    // 2D CPU fallback
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const srcBytes = toRgbaBytes(source);
    if (comparingOriginal) {
      ctx.putImageData(
        new ImageData(new Uint8ClampedArray(srcBytes), source.width, source.height),
        0,
        0
      );
      return;
    }
    const graded = applyLutCpu(source, lut);
    const out = new Uint8ClampedArray(srcBytes);
    const splitCol = Math.round(splitPos * source.width);
    const gradedBytes = graded.data as Uint8ClampedArray;
    for (let y = 0; y < source.height; y++) {
      for (let x = splitCol; x < source.width; x++) {
        const p = (y * source.width + x) * 4;
        out[p] = gradedBytes[p]!;
        out[p + 1] = gradedBytes[p + 1]!;
        out[p + 2] = gradedBytes[p + 2]!;
      }
    }
    ctx.putImageData(new ImageData(out, source.width, source.height), 0, 0);
  }, [source, lut, splitPos, comparingOriginal, useCpuCanvas]);

  function updateSplitFromClientX(clientX: number) {
    const el = wrapRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0) return;
    const next = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    onSplitChange(next);
  }

  return (
    <div
      className="viewport-frame"
      onDragOver={(e) => {
        e.preventDefault();
        setDraggingOver(true);
      }}
      onDragLeave={() => setDraggingOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDraggingOver(false);
        const file = e.dataTransfer.files?.[0];
        if (file && file.type.startsWith('image/')) {
          onDropImageFile(file);
        }
      }}
    >
      <div
        ref={wrapRef}
        className="viewport-canvas-wrap"
        onPointerDown={(e) => {
          draggingSplitRef.current = true;
          e.currentTarget.setPointerCapture(e.pointerId);
          updateSplitFromClientX(e.clientX);
        }}
        onPointerMove={(e) => {
          if (!draggingSplitRef.current) return;
          updateSplitFromClientX(e.clientX);
        }}
        onPointerUp={(e) => {
          draggingSplitRef.current = false;
          if (e.currentTarget.hasPointerCapture(e.pointerId)) {
            e.currentTarget.releasePointerCapture(e.pointerId);
          }
        }}
        onPointerCancel={() => {
          draggingSplitRef.current = false;
        }}
      >
        <canvas
          key={useCpuCanvas ? 'cpu-2d' : 'webgl'}
          ref={canvasRef}
          className="viewport-canvas"
          aria-label="Color grading preview viewport"
        />
        {!comparingOriginal && (
          <>
            <span className="split-tag split-tag-left">Original</span>
            <span className="split-tag split-tag-right">
              Graded · {rendererMode}
            </span>
            <button
              type="button"
              className="split-handle"
              style={{ left: `${splitPos * 100}%` }}
              role="slider"
              aria-label="Original vs Graded split comparison"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(splitPos * 100)}
              onKeyDown={(e) => {
                const step = e.shiftKey ? 0.1 : 0.02;
                if (e.key === 'ArrowLeft') {
                  e.preventDefault();
                  onSplitChange(Math.max(0, splitPos - step));
                } else if (e.key === 'ArrowRight') {
                  e.preventDefault();
                  onSplitChange(Math.min(1, splitPos + step));
                } else if (e.key === 'Home') {
                  e.preventDefault();
                  onSplitChange(0);
                } else if (e.key === 'End') {
                  e.preventDefault();
                  onSplitChange(1);
                }
              }}
            >
              <span className="split-line" aria-hidden="true" />
              <span className="split-knob" aria-hidden="true" />
            </button>
          </>
        )}
        {comparingOriginal && (
          <span className="split-tag split-tag-left">Original (Hold)</span>
        )}
      </div>
      {draggingOver && (
        <div className="drop-overlay">Drop PNG, JPG, or WebP as source frame</div>
      )}
    </div>
  );
}
