const NORMAL_LLOYD_MAX = new Map([
  [1, [-0.7978845608, 0.7978845608]],
  [2, [-1.5104176087, -0.452780034, 0.452780034, 1.5104176087]],
  [
    3,
    [
      -2.1519456699, -1.3439092552, -0.756005248, -0.2450941631, 0.2450941631,
      0.756005248, 1.3439092552, 2.1519456699,
    ],
  ],
  [
    4,
    [
      -2.7325888025, -2.0690178387, -1.6180460517, -1.2562309487, -0.9423401345,
      -0.6567589958, -0.3880482345, -0.1283950167, 0.1283950167, 0.3880482345,
      0.6567589958, 0.9423401345, 1.2562309487, 1.6180460517, 2.0690178387,
      2.7325888025,
    ],
  ],
]);

export function makeCodebook(bits, dim) {
  if (!NORMAL_LLOYD_MAX.has(bits)) {
    throw new RangeError(`Only 1-4 bit Lloyd-Max codebooks are bundled, got ${bits}`);
  }
  if (!Number.isInteger(dim) || dim <= 0) {
    throw new RangeError(`dim must be a positive integer, got ${dim}`);
  }

  const scale = 1 / Math.sqrt(dim);
  const centroids = Float32Array.from(NORMAL_LLOYD_MAX.get(bits), (v) => v * scale);
  const thresholds = new Float32Array(centroids.length - 1);
  for (let i = 0; i < thresholds.length; i++) {
    thresholds[i] = 0.5 * (centroids[i] + centroids[i + 1]);
  }
  return { bits, dim, centroids, thresholds };
}

export function nearestCode(value, thresholds) {
  let lo = 0;
  let hi = thresholds.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (value <= thresholds[mid]) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}
