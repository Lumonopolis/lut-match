# lut-match-core

Zero-dependency TypeScript/JavaScript solver for fitting smooth, zero-inversion Adobe `.cube` 3D LUTs (`17³` / `33³` / `65³`) and Level-8 HaldCLUT PNGs from one or more reference stills.

- **Browser Studio:** https://lumonopolis.github.io/lut-match/
- **Repository & Benchmarks:** https://github.com/Lumonopolis/lut-match

## Usage

```ts
import { fitLut, generateHaldClut, serializeCube } from 'lut-match-core';

const lut = fitLut({
  source,            // ImageDataLike | ImageDataLike[]
  references: [refStill],
  size: 33,
  mode: 'balanced',  // or 'creative'
  skinProtect: 0.5,
  idt: 'srgb',       // 'srgb' | 'rec709' | 'apple-log' | 'slog3' | 'vlog' | 'clog3'
});

const cubeText = serializeCube(lut);
const haldRgba = generateHaldClut(lut); // 512x512 Level-8 HaldCLUT RGBA
```

Fits a `33³` LUT in ~25 ms with `0.00%` monotonicity inversions in sRGB/Rec.709 working space and `< 6e-8` parity with the Python `lut-match` package.
