export function packedByteLength(valueCount, bits) {
  assertBits(bits);
  return Math.ceil((valueCount * bits) / 8);
}

export function packCodes(codes, bits) {
  assertBits(bits);
  const out = new Uint8Array(packedByteLength(codes.length, bits));
  const mask = (1 << bits) - 1;
  let bitOffset = 0;

  for (let i = 0; i < codes.length; i++) {
    const code = codes[i];
    if (code < 0 || code > mask) {
      throw new RangeError(`Code ${code} does not fit in ${bits} bits`);
    }

    let value = code;
    let remaining = bits;
    while (remaining > 0) {
      const byteIndex = bitOffset >> 3;
      const intra = bitOffset & 7;
      const writable = Math.min(8 - intra, remaining);
      const chunkMask = (1 << writable) - 1;
      out[byteIndex] |= (value & chunkMask) << intra;
      value >>= writable;
      remaining -= writable;
      bitOffset += writable;
    }
  }

  return out;
}

export function unpackCodes(packed, bits, valueCount) {
  assertBits(bits);
  const out = new Uint8Array(valueCount);
  const mask = (1 << bits) - 1;
  let bitOffset = 0;

  for (let i = 0; i < valueCount; i++) {
    let value = 0;
    let shift = 0;
    let remaining = bits;
    while (remaining > 0) {
      const byteIndex = bitOffset >> 3;
      const intra = bitOffset & 7;
      const readable = Math.min(8 - intra, remaining);
      const chunkMask = (1 << readable) - 1;
      value |= ((packed[byteIndex] >> intra) & chunkMask) << shift;
      shift += readable;
      remaining -= readable;
      bitOffset += readable;
    }
    out[i] = value & mask;
  }

  return out;
}

export function packSigns(signs) {
  const out = new Uint8Array(Math.ceil(signs.length / 8));
  for (let i = 0; i < signs.length; i++) {
    if (signs[i] >= 0) out[i >> 3] |= 1 << (i & 7);
  }
  return out;
}

export function unpackSigns(packed, valueCount) {
  const out = new Int8Array(valueCount);
  for (let i = 0; i < valueCount; i++) {
    out[i] = (packed[i >> 3] & (1 << (i & 7))) !== 0 ? 1 : -1;
  }
  return out;
}

function assertBits(bits) {
  if (!Number.isInteger(bits) || bits < 1 || bits > 8) {
    throw new RangeError(`bits must be an integer in [1, 8], got ${bits}`);
  }
}
