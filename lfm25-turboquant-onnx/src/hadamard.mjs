import { makeRademacherSigns } from './random.mjs';

export class HadamardRotation {
  constructor(dim, seed = 0x243f6a88) {
    if (!Number.isInteger(dim) || dim <= 0 || (dim & (dim - 1)) !== 0) {
      throw new RangeError(`HadamardRotation requires a power-of-two dim, got ${dim}`);
    }
    this.dim = dim;
    this.scale = 1 / Math.sqrt(dim);
    this.signs = makeRademacherSigns(dim, seed);
  }

  rotate(input) {
    if (input.length !== this.dim) throw new RangeError('input length mismatch');
    const out = new Float32Array(this.dim);
    for (let i = 0; i < this.dim; i++) out[i] = input[i] * this.signs[i];
    fwhtInPlace(out);
    for (let i = 0; i < this.dim; i++) out[i] *= this.scale;
    return out;
  }

  inverse(input) {
    if (input.length !== this.dim) throw new RangeError('input length mismatch');
    const out = Float32Array.from(input);
    fwhtInPlace(out);
    for (let i = 0; i < this.dim; i++) out[i] = out[i] * this.scale * this.signs[i];
    return out;
  }
}

export function fwhtInPlace(data) {
  const n = data.length;
  for (let width = 1; width < n; width <<= 1) {
    for (let start = 0; start < n; start += width << 1) {
      for (let i = 0; i < width; i++) {
        const a = data[start + i];
        const b = data[start + i + width];
        data[start + i] = a + b;
        data[start + i + width] = a - b;
      }
    }
  }
  return data;
}
