import { packCodes, packSigns, packedByteLength, unpackCodes, unpackSigns } from './bitpack.mjs';
import { makeCodebook, nearestCode } from './codebooks.mjs';
import { HadamardRotation } from './hadamard.mjs';
import { makeGaussianMatrix } from './random.mjs';

const SQRT_PI_OVER_2 = Math.sqrt(Math.PI / 2);

export class TurboQuantKVCodec {
  constructor({
    dim = 64,
    bits = 3,
    mode = 'prod',
    rotationSeed = 0x243f6a88,
    qjlSeed = 0x6a09e667,
  } = {}) {
    if (!Number.isInteger(bits) || bits < 1 || bits > 4) {
      throw new RangeError(`bits must be an integer in [1, 4], got ${bits}`);
    }
    if (mode !== 'mse' && mode !== 'prod') {
      throw new RangeError(`mode must be "mse" or "prod", got ${mode}`);
    }

    this.dim = dim;
    this.bits = bits;
    this.mode = mode;
    this.rotation = new HadamardRotation(dim, rotationSeed);
    this.codeBits = mode === 'prod' ? Math.max(0, bits - 1) : bits;
    this.codebook = this.codeBits > 0 ? makeCodebook(this.codeBits, dim) : null;
    this.qjlMatrix = mode === 'prod' ? makeGaussianMatrix(dim, dim, qjlSeed) : null;
    this.codeBytesPerVector = this.codeBits > 0 ? packedByteLength(dim, this.codeBits) : 0;
    this.signBytesPerVector = mode === 'prod' ? packedByteLength(dim, 1) : 0;
  }

  compressTensor(data, dims) {
    const flat = toFloatView(data);
    assertTensorShape(flat, dims, this.dim);
    const vectorCount = flat.length / this.dim;
    const norms = new Float32Array(vectorCount);
    const codes = new Uint8Array(vectorCount * this.codeBytesPerVector);
    const residualNorms = this.mode === 'prod' ? new Float32Array(vectorCount) : null;
    const qjlSigns = this.mode === 'prod' ? new Uint8Array(vectorCount * this.signBytesPerVector) : null;

    const unit = new Float32Array(this.dim);
    const deqRot = new Float32Array(this.dim);
    const approx = new Float32Array(this.dim);
    const residual = new Float32Array(this.dim);

    for (let v = 0; v < vectorCount; v++) {
      const base = v * this.dim;
      const norm = l2(flat, base, this.dim);
      norms[v] = norm;
      if (norm === 0) continue;

      for (let i = 0; i < this.dim; i++) unit[i] = flat[base + i] / norm;

      if (this.codeBits > 0) {
        const rotated = this.rotation.rotate(unit);
        const unpacked = new Uint8Array(this.dim);
        for (let i = 0; i < this.dim; i++) {
          const code = nearestCode(rotated[i], this.codebook.thresholds);
          unpacked[i] = code;
          deqRot[i] = this.codebook.centroids[code];
        }
        codes.set(packCodes(unpacked, this.codeBits), v * this.codeBytesPerVector);
        approx.set(this.rotation.inverse(deqRot));
      } else {
        approx.fill(0);
      }

      if (this.mode === 'prod') {
        for (let i = 0; i < this.dim; i++) residual[i] = unit[i] - approx[i];
        const gamma = l2(residual, 0, this.dim);
        residualNorms[v] = gamma;
        const signs = qjlQuantizeSigns(residual, this.qjlMatrix, this.dim);
        qjlSigns.set(packSigns(signs), v * this.signBytesPerVector);
      }
    }

    return {
      __turboquant: 'lfm25-kv-cache',
      version: 1,
      mode: this.mode,
      bits: this.bits,
      codeBits: this.codeBits,
      dim: this.dim,
      dims: Array.from(dims),
      vectorCount,
      norms,
      codes,
      residualNorms,
      qjlSigns,
    };
  }

  decompressTensor(compressed) {
    assertCompressed(compressed, this);
    const out = new Float32Array(compressed.vectorCount * this.dim);
    const deqRot = new Float32Array(this.dim);
    const approx = new Float32Array(this.dim);

    for (let v = 0; v < compressed.vectorCount; v++) {
      const norm = compressed.norms[v];
      if (norm === 0) continue;

      if (this.codeBits > 0) {
        const codeStart = v * this.codeBytesPerVector;
        const codeBytes = compressed.codes.subarray(codeStart, codeStart + this.codeBytesPerVector);
        const unpacked = unpackCodes(codeBytes, this.codeBits, this.dim);
        for (let i = 0; i < this.dim; i++) deqRot[i] = this.codebook.centroids[unpacked[i]];
        approx.set(this.rotation.inverse(deqRot));
      } else {
        approx.fill(0);
      }

      if (this.mode === 'prod') {
        const signStart = v * this.signBytesPerVector;
        const signBytes = compressed.qjlSigns.subarray(signStart, signStart + this.signBytesPerVector);
        addQjlResidual(approx, signBytes, compressed.residualNorms[v], this.qjlMatrix, this.dim);
      }

      const base = v * this.dim;
      for (let i = 0; i < this.dim; i++) out[base + i] = approx[i] * norm;
    }

    return out;
  }

  estimatedBytes(compressed) {
    const residualBytes = compressed.residualNorms ? compressed.residualNorms.byteLength : 0;
    const signBytes = compressed.qjlSigns ? compressed.qjlSigns.byteLength : 0;
    return compressed.norms.byteLength + compressed.codes.byteLength + residualBytes + signBytes;
  }
}

export function compressLfmKvCache(cache, codec) {
  const out = {};
  for (const [name, value] of Object.entries(cache)) {
    out[name] = shouldCompressName(name) ? codec.compressTensor(value.data ?? value, value.dims) : value;
  }
  return out;
}

export function decompressLfmKvCache(cache, codec, ort) {
  const out = {};
  for (const [name, value] of Object.entries(cache)) {
    if (value?.__turboquant) {
      out[name] = ort ? new ort.Tensor('float32', codec.decompressTensor(value), value.dims) : codec.decompressTensor(value);
    } else {
      out[name] = value;
    }
  }
  return out;
}

export function updateCompressedLfmCacheFromOutputs(outputs, codec) {
  const cache = {};
  for (const [name, tensor] of Object.entries(outputs)) {
    if (name.startsWith('present.')) {
      const pastName = name.replace('present.', 'past_key_values.');
      cache[pastName] = codec.compressTensor(tensor.data, tensor.dims);
    } else if (name.startsWith('present_conv')) {
      cache[name.replace('present_conv', 'past_conv')] = tensor;
    }
  }
  return cache;
}

export function shouldCompressName(name) {
  return name.startsWith('past_key_values.') || name.startsWith('present.');
}

function qjlQuantizeSigns(vector, matrix, dim) {
  const signs = new Int8Array(dim);
  for (let row = 0; row < dim; row++) {
    let dot = 0;
    const rowBase = row * dim;
    for (let col = 0; col < dim; col++) dot += matrix[rowBase + col] * vector[col];
    signs[row] = dot >= 0 ? 1 : -1;
  }
  return signs;
}

function addQjlResidual(out, packedSigns, residualNorm, matrix, dim) {
  if (residualNorm === 0) return;
  const signs = unpackSigns(packedSigns, dim);
  const scale = (residualNorm * SQRT_PI_OVER_2) / dim;
  for (let col = 0; col < dim; col++) {
    let sum = 0;
    for (let row = 0; row < dim; row++) sum += matrix[row * dim + col] * signs[row];
    out[col] += scale * sum;
  }
}

function toFloatView(data) {
  if (data instanceof Float32Array) return data;
  return Float32Array.from(data);
}

function l2(data, offset, length) {
  let sum = 0;
  for (let i = 0; i < length; i++) {
    const value = data[offset + i];
    sum += value * value;
  }
  return Math.sqrt(sum);
}

function assertTensorShape(data, dims, dim) {
  if (!Array.isArray(dims) || dims.length === 0 || dims[dims.length - 1] !== dim) {
    throw new RangeError(`Expected tensor dims ending in ${dim}, got ${JSON.stringify(dims)}`);
  }
  if (data.length % dim !== 0) {
    throw new RangeError(`Tensor length ${data.length} is not divisible by dim ${dim}`);
  }
}

function assertCompressed(value, codec) {
  if (value?.__turboquant !== 'lfm25-kv-cache') throw new TypeError('not a TurboQuant KV tensor');
  if (value.dim !== codec.dim || value.mode !== codec.mode || value.bits !== codec.bits) {
    throw new RangeError('compressed tensor was produced by a different codec configuration');
  }
}
