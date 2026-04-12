import ortModule from 'onnxruntime-web/webgpu';
import { AutoTokenizer } from '@huggingface/transformers';
import {
  TurboQuantKVCodec,
  decompressLfmKvCache,
  updateCompressedLfmCacheFromOutputs,
} from '../src/index.mjs';

const ort = ortModule.default ?? ortModule;
const modelId = 'LiquidAI/LFM2.5-350M-ONNX';
const modelBase = `https://huggingface.co/${modelId}/resolve/main`;
const maxNewTokens = 64;

ort.env.wasm.numThreads = 1;

const tokenizer = await AutoTokenizer.from_pretrained(modelId);
const session = await ort.InferenceSession.create(`${modelBase}/onnx/model_q4.onnx`, {
  executionProviders: ['webgpu'],
  externalData: [{ path: 'model_q4.onnx_data', data: `${modelBase}/onnx/model_q4.onnx_data` }],
});

const codec = new TurboQuantKVCodec({ dim: 64, bits: 3, mode: 'prod' });
let compressedCache = initEmptyLfmCache(session, codec);

const messages = [{ role: 'user', content: 'Give me a one sentence summary of TurboQuant.' }];
const prompt = tokenizer.apply_chat_template(messages, {
  add_generation_prompt: true,
  tokenize: false,
});
const inputIds = tokenizer.encode(prompt);
const generated = [];
let ids = inputIds;
let curLen = inputIds.length;

for (let step = 0; step < maxNewTokens; step++) {
  const feedCache = decompressLfmKvCache(compressedCache, codec, ort);
  const feed = {
    input_ids: makeInt64Tensor(ids, [1, ids.length]),
    attention_mask: makeOnesInt64Tensor(curLen, [1, curLen]),
    ...feedCache,
  };

  if (session.inputNames.includes('position_ids')) {
    const start = curLen - ids.length;
    feed.position_ids = makeInt64Tensor(
      Array.from({ length: ids.length }, (_, i) => start + i),
      [1, ids.length],
    );
  }

  const outputs = await session.run(feed);
  const logits = outputs.logits;
  const nextToken = greedyLastToken(logits);
  generated.push(nextToken);
  if (nextToken === tokenizer.eos_token_id) break;

  compressedCache = {
    ...compressedCache,
    ...updateCompressedLfmCacheFromOutputs(outputs, codec),
  };
  ids = [nextToken];
  curLen++;
}

console.log(tokenizer.decode(generated, { skip_special_tokens: true }));

function initEmptyLfmCache(activeSession, activeCodec) {
  const cache = {};
  for (const name of activeSession.inputNames) {
    if (name.startsWith('past_conv')) {
      cache[name] = new ort.Tensor('float32', new Float32Array(1024 * 3), [1, 1024, 3]);
    } else if (name.startsWith('past_key_values')) {
      cache[name] = activeCodec.compressTensor(new Float32Array(0), [1, 8, 0, 64]);
    }
  }
  return cache;
}

function greedyLastToken(logits) {
  const [batch, steps, vocab] = logits.dims;
  if (batch !== 1) throw new Error(`Only batch=1 is supported by this demo, got ${batch}`);
  const offset = (steps - 1) * vocab;
  let best = 0;
  let bestValue = -Infinity;
  for (let i = 0; i < vocab; i++) {
    const value = logits.data[offset + i];
    if (value > bestValue) {
      best = i;
      bestValue = value;
    }
  }
  return best;
}

function makeInt64Tensor(values, dims) {
  return new ort.Tensor('int64', BigInt64Array.from(values, BigInt), dims);
}

function makeOnesInt64Tensor(count, dims) {
  return new ort.Tensor('int64', new BigInt64Array(count).fill(1n), dims);
}
