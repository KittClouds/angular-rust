export function makeRademacherSigns(count, seed = 0x9e3779b9) {
  const signs = new Int8Array(count);
  let state = seed >>> 0;
  for (let i = 0; i < count; i++) {
    state = splitMix32(state + i + 1);
    signs[i] = (state & 1) === 0 ? -1 : 1;
  }
  return signs;
}

export function makeGaussianMatrix(rows, cols, seed = 0x6a09e667) {
  const matrix = new Float32Array(rows * cols);
  let state = seed >>> 0;

  for (let i = 0; i < matrix.length; i += 2) {
    state = splitMix32(state + i + 1);
    const u1 = uintToUnitOpen(state);
    state = splitMix32(state + i + 2);
    const u2 = uintToUnitOpen(state);
    const radius = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    matrix[i] = radius * Math.cos(theta);
    if (i + 1 < matrix.length) matrix[i + 1] = radius * Math.sin(theta);
  }

  return matrix;
}

export function splitMix32(x) {
  let z = (x + 0x9e3779b9) >>> 0;
  z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
  z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
  return (z ^ (z >>> 16)) >>> 0;
}

function uintToUnitOpen(x) {
  return ((x >>> 0) + 0.5) / 4294967296;
}
