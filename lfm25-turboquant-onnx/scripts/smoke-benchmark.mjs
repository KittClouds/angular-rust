import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import ort from 'onnxruntime-node';
import { Tokenizer } from '@huggingface/tokenizers';
import {
  TurboQuantKVCodec,
  decompressLfmKvCache,
  updateCompressedLfmCacheFromOutputs,
} from '../src/index.mjs';

const MODEL_ID = 'LiquidAI/LFM2.5-350M-ONNX';
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const CACHE_ROOT = path.join(ROOT, '.cache', 'hf-models', 'LiquidAI', 'LFM2.5-350M-ONNX');
const RESULT_ROOT = path.join(ROOT, 'results');
const DEFAULT_PROMPT = 'Summarize TurboQuant in one short sentence.';
const DEFAULT_STEPS = 8;
const DEFAULT_VARIANTS = ['q4', 'q8'];
const VARIANT_FILE = {
  fp16: 'model_fp16.onnx',
  q4: 'model_q4.onnx',
  q4f32: 'model_q4f32.onnx',
  q8: 'model_q8.onnx',
};

const args = parseArgs(process.argv.slice(2));
const variants = args.variants ?? DEFAULT_VARIANTS;
const prompt = args.prompt ?? DEFAULT_PROMPT;
const steps = Number(args.steps ?? DEFAULT_STEPS);
const codec = new TurboQuantKVCodec({
  dim: Number(args.dim ?? 64),
  bits: Number(args.bits ?? 3),
  mode: args.mode ?? 'prod',
});

await fs.promises.mkdir(CACHE_ROOT, { recursive: true });
await fs.promises.mkdir(RESULT_ROOT, { recursive: true });

console.log(`Loading tokenizer for ${MODEL_ID}...`);
const tokenizer = await loadTokenizer();

const results = {
  modelId: MODEL_ID,
  prompt,
  steps,
  timestamp: new Date().toISOString(),
  turboquant: {
    mode: codec.mode,
    bits: codec.bits,
    dim: codec.dim,
    codeBits: codec.codeBits,
  },
  variants: {},
  crossPrecision: {},
};

for (const variant of variants) {
  console.log(`\n=== Benchmarking ${variant} ===`);
  const benchmark = await benchmarkVariant({ tokenizer, variant, prompt, steps, codec });
  results.variants[variant] = benchmark;
  printVariantSummary(benchmark);
}

if (variants.length >= 2) {
  const [left, right] = variants;
  results.crossPrecision[`${left}_vs_${right}`] = compareVariantBaselines(
    results.variants[left],
    results.variants[right],
  );
  printCrossSummary(left, right, results.crossPrecision[`${left}_vs_${right}`]);
}

const stamp = compactStamp();
const jsonPath = path.join(RESULT_ROOT, `smoke-benchmark-${stamp}.json`);
await fs.promises.writeFile(jsonPath, `${JSON.stringify(results, null, 2)}\n`, 'utf8');
console.log(`\nSaved metrics to ${jsonPath}`);

async function benchmarkVariant({ tokenizer, variant, prompt, steps, codec }) {
  const modelPath = await ensureVariantFiles(variant);
  const initStart = performance.now();
  const session = await ort.InferenceSession.create(modelPath);
  const initMs = performance.now() - initStart;

  const promptIds = tokenizer.encode(prompt).ids;
  const baseCache = initEmptyCache(session);

  const firstFeed = makeFeed(session, promptIds, promptIds.length, baseCache);
  const firstStart = performance.now();
  const firstOutputs = await session.run(firstFeed);
  const firstMs = performance.now() - firstStart;

  const firstLogits = getLastLogits(firstOutputs.logits);
  const firstToken = argmax(firstLogits);

  let rawCache = updateRawCacheFromOutputs(firstOutputs);
  let compressedCache = updateCompressedLfmCacheFromOutputs(firstOutputs, codec);

  const rawTokens = [firstToken];
  const compressedTokens = [firstToken];
  const decodeBaselineMs = [];
  const decodeTurboMs = [];
  const encodeMs = [];
  const decodeCacheMs = [];
  const cacheMse = [];
  const cacheCosine = [];
  const logitCosine = [];
  const logitRmse = [];
  const logitMaxAbs = [];
  const perStep = [];
  let prevRawToken = firstToken;
  let prevCompressedToken = firstToken;
  let curLen = promptIds.length + 1;

  for (let step = 1; step < steps; step++) {
    const rawFeed = makeFeed(session, [prevRawToken], curLen, rawCache);
    const rawStart = performance.now();
    const rawOutputs = await session.run(rawFeed);
    decodeBaselineMs.push(performance.now() - rawStart);
    const rawLogits = getLastLogits(rawOutputs.logits);
    const nextRaw = argmax(rawLogits);
    rawTokens.push(nextRaw);

    const decompressStart = performance.now();
    const turboFeedCache = decompressLfmKvCache(compressedCache, codec, ort);
    decodeCacheMs.push(performance.now() - decompressStart);
    const turboFeed = makeFeed(session, [prevCompressedToken], curLen, turboFeedCache);
    const turboStart = performance.now();
    const turboOutputs = await session.run(turboFeed);
    decodeTurboMs.push(performance.now() - turboStart);
    const turboLogits = getLastLogits(turboOutputs.logits);
    const nextTurbo = argmax(turboLogits);
    compressedTokens.push(nextTurbo);

    const rawKv = extractKvTensors(rawOutputs);
    const encodeStart = performance.now();
    const nextCompressedCache = updateCompressedLfmCacheFromOutputs(turboOutputs, codec);
    encodeMs.push(performance.now() - encodeStart);
    const decompressedKv = extractKvArrays(decompressLfmKvCache(nextCompressedCache, codec, ort));

    const stepLogitStats = compareArrays(rawLogits, turboLogits);
    const stepCacheStats = compareCacheMaps(rawKv, decompressedKv);
    logitCosine.push(stepLogitStats.cosine);
    logitRmse.push(stepLogitStats.rmse);
    logitMaxAbs.push(stepLogitStats.maxAbs);
    cacheMse.push(stepCacheStats.relativeMse);
    cacheCosine.push(stepCacheStats.cosine);

    rawCache = updateRawCacheFromOutputs(rawOutputs);
    compressedCache = nextCompressedCache;
    prevRawToken = nextRaw;
    prevCompressedToken = nextTurbo;
    perStep.push({
      step,
      rawToken: nextRaw,
      turboToken: nextTurbo,
      tokenMatch: nextRaw === nextTurbo,
      rawCacheKvBytes: cacheBytes(rawCache).kvBytes,
      turboCacheKvBytes: compressedCacheBytes(compressedCache),
      logitCosine: stepLogitStats.cosine,
      logitRmse: stepLogitStats.rmse,
      logitMaxAbs: stepLogitStats.maxAbs,
      cacheRelativeMse: stepCacheStats.relativeMse,
      cacheCosine: stepCacheStats.cosine,
    });
    curLen++;
  }

  const rawDecoded = tokenizer.decode(rawTokens);
  const turboDecoded = tokenizer.decode(compressedTokens);
  const rawCacheSize = cacheBytes(rawCache);
  const turboCacheSize = compressedCacheBytes(compressedCache);

  return {
    variant,
    modelPath,
    modelBytes: fileBytes(modelPath) + fileBytes(`${modelPath}_data`),
    promptTokenCount: promptIds.length,
    firstToken,
    initMs,
    firstMs,
    rawTokens,
    turboTokens: compressedTokens,
    rawText: rawDecoded,
    turboText: turboDecoded,
    exactTokenMatch: arrayEqual(rawTokens, compressedTokens),
    matchingPrefixLength: matchingPrefix(rawTokens, compressedTokens),
    rawCache: rawCacheSize,
    turboCache: {
      kvBytes: turboCacheSize,
      compressionRatio: rawCacheSize.kvBytes / Math.max(turboCacheSize, 1),
    },
    averages: {
      decodeBaselineMs: average(decodeBaselineMs),
      decodeTurboMs: average(decodeTurboMs),
      turboEncodeMs: average(encodeMs),
      turboDecodeCacheMs: average(decodeCacheMs),
      logitCosine: average(logitCosine),
      logitRmse: average(logitRmse),
      logitMaxAbs: average(logitMaxAbs),
      cacheRelativeMse: average(cacheMse),
      cacheCosine: average(cacheCosine),
    },
    perStep,
  };
}

async function ensureVariantFiles(variant) {
  const filename = VARIANT_FILE[variant];
  if (!filename) throw new Error(`Unknown variant "${variant}"`);

  const localOnnx = path.join(CACHE_ROOT, filename);
  const localData = `${localOnnx}_data`;
  await ensureDownload(`https://huggingface.co/${MODEL_ID}/resolve/main/onnx/${filename}`, localOnnx);
  await ensureDownload(`https://huggingface.co/${MODEL_ID}/resolve/main/onnx/${filename}_data`, localData);
  return localOnnx;
}

async function loadTokenizer() {
  const tokenizerJsonPath = path.join(CACHE_ROOT, 'tokenizer.json');
  const tokenizerConfigPath = path.join(CACHE_ROOT, 'tokenizer_config.json');
  await ensureDownload(`https://huggingface.co/${MODEL_ID}/resolve/main/tokenizer.json`, tokenizerJsonPath);
  await ensureDownload(
    `https://huggingface.co/${MODEL_ID}/resolve/main/tokenizer_config.json`,
    tokenizerConfigPath,
  );
  const tokenizerJson = JSON.parse(await fs.promises.readFile(tokenizerJsonPath, 'utf8'));
  const tokenizerConfig = JSON.parse(await fs.promises.readFile(tokenizerConfigPath, 'utf8'));
  return new Tokenizer(tokenizerJson, tokenizerConfig);
}

async function ensureDownload(url, filePath) {
  if (fs.existsSync(filePath) && fs.statSync(filePath).size > 0) return;

  await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
  console.log(`Downloading ${path.basename(filePath)}...`);
  const response = await fetch(url);
  if (!response.ok || !response.body) {
    throw new Error(`Download failed for ${url}: ${response.status} ${response.statusText}`);
  }

  const sink = fs.createWriteStream(filePath);
  await new Promise((resolve, reject) => {
    response.body.pipeTo(
      new WritableStream({
        write(chunk) {
          return new Promise((res, rej) => sink.write(Buffer.from(chunk), (err) => (err ? rej(err) : res())));
        },
        close() {
          sink.end(resolve);
        },
        abort(reason) {
          sink.destroy(reason);
          reject(reason);
        },
      }),
    ).catch((err) => {
      sink.destroy(err);
      reject(err);
    });
  });
}

function initEmptyCache(session) {
  const cache = {};
  for (const name of session.inputNames) {
    if (name.startsWith('past_conv')) {
      cache[name] = new ort.Tensor('float32', new Float32Array(1024 * 3), [1, 1024, 3]);
    } else if (name.startsWith('past_key_values')) {
      cache[name] = new ort.Tensor('float32', new Float32Array(0), [1, 8, 0, 64]);
    }
  }
  return cache;
}

function makeFeed(session, ids, totalLength, cache) {
  const feed = {
    input_ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]),
    attention_mask: new ort.Tensor('int64', new BigInt64Array(totalLength).fill(1n), [1, totalLength]),
    ...cache,
  };

  if (session.inputNames.includes('position_ids')) {
    const start = totalLength - ids.length;
    feed.position_ids = new ort.Tensor(
      'int64',
      BigInt64Array.from({ length: ids.length }, (_, i) => BigInt(start + i)),
      [1, ids.length],
    );
  }

  return feed;
}

function updateRawCacheFromOutputs(outputs) {
  const cache = {};
  for (const [name, tensor] of Object.entries(outputs)) {
    if (name.startsWith('present_conv')) {
      cache[name.replace('present_conv', 'past_conv')] = tensor;
    } else if (name.startsWith('present.')) {
      cache[name.replace('present.', 'past_key_values.')] = tensor;
    }
  }
  return cache;
}

function extractKvTensors(outputs) {
  const kv = {};
  for (const [name, tensor] of Object.entries(outputs)) {
    if (name.startsWith('present.')) kv[name] = tensor.data;
  }
  return kv;
}

function extractKvArrays(cache) {
  const kv = {};
  for (const [name, tensor] of Object.entries(cache)) {
    if (name.startsWith('past_key_values.')) kv[name.replace('past_key_values.', 'present.')] = tensor.data;
  }
  return kv;
}

function compressedCacheBytes(cache) {
  let bytes = 0;
  for (const value of Object.values(cache)) {
    if (value?.__turboquant === 'lfm25-kv-cache') {
      bytes += value.norms.byteLength;
      bytes += value.codes.byteLength;
      if (value.residualNorms) bytes += value.residualNorms.byteLength;
      if (value.qjlSigns) bytes += value.qjlSigns.byteLength;
    }
  }
  return bytes;
}

function cacheBytes(cache) {
  let kvBytes = 0;
  let totalBytes = 0;
  for (const [name, tensor] of Object.entries(cache)) {
    const bytes = tensor.data.byteLength;
    totalBytes += bytes;
    if (name.startsWith('past_key_values.')) kvBytes += bytes;
  }
  return { kvBytes, totalBytes };
}

function getLastLogits(logitsTensor) {
  const [batch, seq, vocab] = logitsTensor.dims;
  if (batch !== 1) throw new Error(`Expected batch=1 logits, got ${batch}`);
  return logitsTensor.data.slice((seq - 1) * vocab, seq * vocab);
}

function argmax(values) {
  let bestIndex = 0;
  let bestValue = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > bestValue) {
      bestValue = values[i];
      bestIndex = i;
    }
  }
  return bestIndex;
}

function compareArrays(left, right) {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  let squared = 0;
  let maxAbs = 0;
  for (let i = 0; i < left.length; i++) {
    const l = Number(left[i]);
    const r = Number(right[i]);
    const d = l - r;
    dot += l * r;
    leftNorm += l * l;
    rightNorm += r * r;
    squared += d * d;
    const abs = Math.abs(d);
    if (abs > maxAbs) maxAbs = abs;
  }
  return {
    cosine: dot / Math.max(Math.sqrt(leftNorm) * Math.sqrt(rightNorm), Number.EPSILON),
    rmse: Math.sqrt(squared / Math.max(left.length, 1)),
    maxAbs,
  };
}

function compareCacheMaps(left, right) {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  let squared = 0;
  for (const name of Object.keys(left)) {
    const a = left[name];
    const b = right[name];
    for (let i = 0; i < a.length; i++) {
      const av = Number(a[i]);
      const bv = Number(b[i]);
      const d = av - bv;
      dot += av * bv;
      leftNorm += av * av;
      rightNorm += bv * bv;
      squared += d * d;
    }
  }
  return {
    cosine: dot / Math.max(Math.sqrt(leftNorm) * Math.sqrt(rightNorm), Number.EPSILON),
    relativeMse: squared / Math.max(leftNorm, Number.EPSILON),
  };
}

function compareVariantBaselines(left, right) {
  return {
    rawTokenMatch: arrayEqual(left.rawTokens, right.rawTokens),
    matchingPrefixLength: matchingPrefix(left.rawTokens, right.rawTokens),
    rawTextLeft: left.rawText,
    rawTextRight: right.rawText,
    firstTokenMatch: left.firstToken === right.firstToken,
  };
}

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const item = argv[i];
    if (!item.startsWith('--')) continue;
    const key = item.slice(2);
    const value = argv[i + 1];
    if (value && !value.startsWith('--')) {
      out[key] = key === 'variants' ? value.split(',').map((v) => v.trim()).filter(Boolean) : value;
      i++;
    } else {
      out[key] = true;
    }
  }
  return out;
}

function average(values) {
  return values.length === 0 ? 0 : values.reduce((sum, v) => sum + v, 0) / values.length;
}

function arrayEqual(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function matchingPrefix(left, right) {
  let i = 0;
  while (i < left.length && i < right.length && left[i] === right[i]) i++;
  return i;
}

function fileBytes(filePath) {
  return fs.existsSync(filePath) ? fs.statSync(filePath).size : 0;
}

function compactStamp() {
  return new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d+Z$/, 'Z');
}

function printVariantSummary(result) {
  console.log(`prompt tokens: ${result.promptTokenCount}`);
  console.log(`raw text:   ${JSON.stringify(result.rawText)}`);
  console.log(`turbo text: ${JSON.stringify(result.turboText)}`);
  console.log(
    `token match: ${result.exactTokenMatch} (prefix ${result.matchingPrefixLength}/${result.rawTokens.length})`,
  );
  console.log(
    `kv bytes raw=${result.rawCache.kvBytes} turbo=${result.turboCache.kvBytes} ratio=${result.turboCache.compressionRatio.toFixed(2)}x`,
  );
  console.log(
    `avg decode ms raw=${result.averages.decodeBaselineMs.toFixed(1)} turbo=${result.averages.decodeTurboMs.toFixed(1)} encode=${result.averages.turboEncodeMs.toFixed(1)} decodeCache=${result.averages.turboDecodeCacheMs.toFixed(1)}`,
  );
  console.log(
    `avg drift logits cosine=${result.averages.logitCosine.toFixed(6)} rmse=${result.averages.logitRmse.toFixed(6)} cache relMSE=${result.averages.cacheRelativeMse.toExponential(3)}`,
  );
}

function printCrossSummary(left, right, summary) {
  console.log(`\n=== Cross precision: ${left} vs ${right} ===`);
  console.log(
    `raw token match: ${summary.rawTokenMatch} (prefix ${summary.matchingPrefixLength}) first token match=${summary.firstTokenMatch}`,
  );
  console.log(`${left}: ${JSON.stringify(summary.rawTextLeft)}`);
  console.log(`${right}: ${JSON.stringify(summary.rawTextRight)}`);
}
