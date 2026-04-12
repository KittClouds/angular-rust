from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_ID = "LiquidAI/LFM2.5-350M-ONNX"
ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = ROOT / ".cache" / "hf-models" / "LiquidAI" / "LFM2.5-350M-ONNX"
RESULT_ROOT = ROOT / "results"
MODEL_FILES = {
    "fp16": "model_fp16.onnx",
    "q4": "model_q4.onnx",
    "q4f32": "model_q4f32.onnx",
    "q8": "model_q8.onnx",
}
DEFAULT_PROMPT = "Summarize TurboQuant in one short sentence."

LLOYD_MAX = {
    1: np.array([-0.7978845608, 0.7978845608], dtype=np.float32),
    2: np.array([-1.5104176087, -0.4527800340, 0.4527800340, 1.5104176087], dtype=np.float32),
    3: np.array(
        [-2.1519456699, -1.3439092552, -0.7560052480, -0.2450941631, 0.2450941631, 0.7560052480, 1.3439092552, 2.1519456699],
        dtype=np.float32,
    ),
    4: np.array(
        [
            -2.7325888025,
            -2.0690178387,
            -1.6180460517,
            -1.2562309487,
            -0.9423401345,
            -0.6567589958,
            -0.3880482345,
            -0.1283950167,
            0.1283950167,
            0.3880482345,
            0.6567589958,
            0.9423401345,
            1.2562309487,
            1.6180460517,
            2.0690178387,
            2.7325888025,
        ],
        dtype=np.float32,
    ),
}


@dataclass
class CompressedTensor:
    dims: tuple[int, ...]
    norms: np.ndarray
    codes: np.ndarray
    residual_norms: np.ndarray | None
    signs: np.ndarray | None
    shift: np.ndarray | None
    scale: np.ndarray | None
    packed_bytes: int


class TurboQuantKVCodec:
    def __init__(
        self,
        dim: int = 64,
        bits: int = 3,
        mode: str = "prod",
        rotation_seed: int = 0x243F6A88,
        qjl_seed: int = 0x6A09E667,
        precondition: str = "none",
    ):
        if mode not in {"mse", "prod"}:
            raise ValueError(f"mode must be 'mse' or 'prod', got {mode}")
        if bits < 1 or bits > 4:
            raise ValueError(f"bits must be in [1, 4], got {bits}")
        if precondition not in {"none", "mean_rms"}:
            raise ValueError(f"precondition must be 'none' or 'mean_rms', got {precondition}")

        self.dim = dim
        self.bits = bits
        self.mode = mode
        self.precondition = precondition
        self.code_bits = bits if mode == "mse" else max(0, bits - 1)
        centroids = LLOYD_MAX[self.code_bits] / math.sqrt(dim) if self.code_bits > 0 else np.zeros((1,), dtype=np.float32)
        self.centroids = centroids.astype(np.float32)
        self.thresholds = ((self.centroids[:-1] + self.centroids[1:]) * 0.5).astype(np.float32) if self.code_bits > 0 else np.zeros((0,), dtype=np.float32)
        self.scale = 1.0 / math.sqrt(dim)

        rotation_rng = np.random.default_rng(rotation_seed)
        self.signs = rotation_rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(dim,))

        if mode == "prod":
            qjl_rng = np.random.default_rng(qjl_seed)
            self.qjl = qjl_rng.standard_normal((dim, dim), dtype=np.float32)
        else:
            self.qjl = None

    def compress_tensor(self, data: np.ndarray, dims: tuple[int, ...]) -> CompressedTensor:
        flat = np.asarray(data, dtype=np.float32).reshape(-1, self.dim)
        working = flat
        shift = None
        scale = None

        if self.precondition == "mean_rms":
            shift = flat.mean(axis=1).astype(np.float32)
            centered = flat - shift[:, None]
            scale = np.sqrt(np.mean(centered * centered, axis=1) + 1e-6).astype(np.float32)
            working = np.divide(centered, scale[:, None], out=np.zeros_like(centered), where=scale[:, None] > 0)

        norms = np.linalg.norm(working, axis=1).astype(np.float32)
        unit = np.divide(working, norms[:, None], out=np.zeros_like(working), where=norms[:, None] > 0)

        if self.code_bits > 0:
            rotated = fwht_rows(unit * self.signs) * self.scale
            codes = np.searchsorted(self.thresholds, rotated, side="left").astype(np.uint8)
            approx_rot = self.centroids[codes]
            approx = fwht_rows(approx_rot) * self.scale * self.signs
        else:
            codes = np.zeros_like(unit, dtype=np.uint8)
            approx = np.zeros_like(unit, dtype=np.float32)

        residual_norms = None
        signs = None
        if self.mode == "prod":
            residual = unit - approx
            residual_norms = np.linalg.norm(residual, axis=1).astype(np.float32)
            dots = residual @ self.qjl.T
            signs = np.where(dots >= 0, 1.0, -1.0).astype(np.float32)

        packed_bytes = norms.nbytes
        packed_bytes += math.ceil(flat.shape[0] * self.dim * self.code_bits / 8)
        if residual_norms is not None:
            packed_bytes += residual_norms.nbytes
            packed_bytes += math.ceil(flat.shape[0] * self.dim / 8)
        if shift is not None and scale is not None:
            packed_bytes += shift.nbytes + scale.nbytes

        return CompressedTensor(tuple(dims), norms, codes, residual_norms, signs, shift, scale, packed_bytes)

    def decompress_tensor(self, compressed: CompressedTensor) -> np.ndarray:
        if self.code_bits > 0:
            approx_rot = self.centroids[compressed.codes]
            approx = fwht_rows(approx_rot) * self.scale * self.signs
        else:
            approx = np.zeros((compressed.norms.shape[0], self.dim), dtype=np.float32)

        if self.mode == "prod" and compressed.signs is not None and compressed.residual_norms is not None:
            residual_scale = (compressed.residual_norms[:, None] * math.sqrt(math.pi / 2.0)) / self.dim
            approx = approx + residual_scale * (compressed.signs @ self.qjl)

        restored = approx * compressed.norms[:, None]
        if compressed.shift is not None and compressed.scale is not None:
            restored = restored * compressed.scale[:, None] + compressed.shift[:, None]
        return restored.reshape(compressed.dims).astype(np.float32)


@dataclass
class SegmentedCompressedTensor:
    dims: tuple[int, ...]
    prefix: np.ndarray | None
    middle: CompressedTensor | None
    suffix: np.ndarray | None
    packed_bytes: int


class ResearchCacheCompressor:
    def __init__(
        self,
        key_codec: TurboQuantKVCodec,
        value_codec: TurboQuantKVCodec,
        keep_prefix: int = 0,
        keep_recent: int = 0,
    ):
        self.key_codec = key_codec
        self.value_codec = value_codec
        self.keep_prefix = max(0, keep_prefix)
        self.keep_recent = max(0, keep_recent)

    def codec_for_name(self, name: str) -> TurboQuantKVCodec | None:
        if name.endswith(".key"):
            return self.key_codec
        if name.endswith(".value"):
            return self.value_codec
        return None

    def compress_named(self, name: str, value: np.ndarray) -> object:
        codec = self.codec_for_name(name)
        if codec is None:
            return value

        seq_len = int(value.shape[2])
        prefix_len = min(self.keep_prefix, seq_len)
        suffix_len = min(self.keep_recent, max(0, seq_len - prefix_len))
        middle_end = seq_len - suffix_len

        prefix = value[:, :, :prefix_len, :].copy() if prefix_len > 0 else None
        suffix = value[:, :, middle_end:, :].copy() if suffix_len > 0 else None
        middle_raw = value[:, :, prefix_len:middle_end, :]
        middle = codec.compress_tensor(middle_raw, tuple(middle_raw.shape)) if middle_raw.shape[2] > 0 else None

        packed_bytes = 0
        if prefix is not None:
            packed_bytes += int(prefix.nbytes)
        if suffix is not None:
            packed_bytes += int(suffix.nbytes)
        if middle is not None:
            packed_bytes += middle.packed_bytes

        return SegmentedCompressedTensor(tuple(value.shape), prefix, middle, suffix, packed_bytes)

    def decompress_named(self, name: str, value: object) -> np.ndarray:
        if isinstance(value, SegmentedCompressedTensor):
            parts: list[np.ndarray] = []
            if value.prefix is not None:
                parts.append(value.prefix)
            if value.middle is not None:
                codec = self.codec_for_name(name)
                if codec is None:
                    raise ValueError(f"missing codec for {name}")
                parts.append(codec.decompress_tensor(value.middle))
            if value.suffix is not None:
                parts.append(value.suffix)
            if not parts:
                return np.zeros(value.dims, dtype=np.float32)
            return np.concatenate(parts, axis=2).astype(np.float32)
        if isinstance(value, CompressedTensor):
            codec = self.codec_for_name(name)
            if codec is None:
                raise ValueError(f"missing codec for {name}")
            return codec.decompress_tensor(value)
        return value


def fwht_rows(array: np.ndarray) -> np.ndarray:
    out = np.array(array, dtype=np.float32, copy=True)
    n = out.shape[1]
    h = 1
    while h < n:
        reshaped = out.reshape((-1, n // (2 * h), 2, h))
        a = reshaped[:, :, 0, :].copy()
        b = reshaped[:, :, 1, :].copy()
        reshaped[:, :, 0, :] = a + b
        reshaped[:, :, 1, :] = a - b
        out = reshaped.reshape((-1, n))
        h *= 2
    return out


def ensure_file(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    print(f"Downloading {target.name}...")
    with urllib.request.urlopen(url) as response, target.open("wb") as sink:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            sink.write(chunk)
    return target


def ensure_variant_files(variant: str) -> Path:
    filename = MODEL_FILES[variant]
    model_path = CACHE_ROOT / filename
    ensure_file(f"https://huggingface.co/{MODEL_ID}/resolve/main/onnx/{filename}", model_path)
    ensure_file(f"https://huggingface.co/{MODEL_ID}/resolve/main/onnx/{filename}_data", Path(f"{model_path}_data"))
    return model_path


def ensure_tokenizer() -> Tokenizer:
    tokenizer_path = CACHE_ROOT / "tokenizer.json"
    ensure_file(f"https://huggingface.co/{MODEL_ID}/resolve/main/tokenizer.json", tokenizer_path)
    return Tokenizer.from_file(str(tokenizer_path))


def init_empty_cache(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        name = inp.name
        if name.startswith("past_conv"):
            cache[name] = np.zeros((1, 1024, 3), dtype=np.float32)
        elif name.startswith("past_key_values"):
            cache[name] = np.zeros((1, 8, 0, 64), dtype=np.float32)
    return cache


def make_feed(session: ort.InferenceSession, ids: list[int], total_length: int, cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    input_names = {inp.name for inp in session.get_inputs()}
    feed: dict[str, np.ndarray] = {
        "input_ids": np.asarray([ids], dtype=np.int64),
        "attention_mask": np.ones((1, total_length), dtype=np.int64),
        **cache,
    }
    if "num_logits_to_keep" in input_names:
        feed["num_logits_to_keep"] = np.asarray(1, dtype=np.int64)
    if "position_ids" in input_names:
        start = total_length - len(ids)
        feed["position_ids"] = np.arange(start, start + len(ids), dtype=np.int64)[None, :]
    return feed


def update_raw_cache(outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    for name, value in outputs.items():
        if name.startswith("present_conv"):
            cache[name.replace("present_conv", "past_conv")] = value
        elif name.startswith("present."):
            cache[name.replace("present.", "past_key_values.")] = value
    return cache


def update_compressed_cache(outputs: dict[str, np.ndarray], compressor: ResearchCacheCompressor) -> dict[str, object]:
    cache: dict[str, object] = {}
    for name, value in outputs.items():
        if name.startswith("present_conv"):
            cache[name.replace("present_conv", "past_conv")] = value
        elif name.startswith("present."):
            cache_name = name.replace("present.", "past_key_values.")
            cache[cache_name] = compressor.compress_named(cache_name, value)
    return cache


def decompress_cache(cache: dict[str, object], compressor: ResearchCacheCompressor) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {}
    for name, value in cache.items():
        feed[name] = compressor.decompress_named(name, value)
    return feed


def cache_bytes(cache: dict[str, np.ndarray]) -> dict[str, int]:
    kv_bytes = 0
    total_bytes = 0
    for name, value in cache.items():
        size = int(value.nbytes)
        total_bytes += size
        if name.startswith("past_key_values."):
            kv_bytes += size
    return {"kvBytes": kv_bytes, "totalBytes": total_bytes}


def compressed_cache_bytes(cache: dict[str, object]) -> int:
    total = 0
    for value in cache.values():
        if isinstance(value, (CompressedTensor, SegmentedCompressedTensor)):
            total += value.packed_bytes
    return total


def last_logits(logits: np.ndarray) -> np.ndarray:
    return np.asarray(logits[0, -1], dtype=np.float32)


def argmax(values: np.ndarray) -> int:
    return int(np.argmax(values))


def compare_arrays(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    diff = left - right
    left_norm = float(np.dot(left, left))
    right_norm = float(np.dot(right, right))
    dot = float(np.dot(left, right))
    return {
        "cosine": dot / max(math.sqrt(left_norm) * math.sqrt(right_norm), np.finfo(np.float64).eps),
        "rmse": math.sqrt(float(np.mean(diff * diff))),
        "maxAbs": float(np.max(np.abs(diff))),
    }


def compare_cache_maps(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, float]:
    left_flat = np.concatenate([np.asarray(left[name], dtype=np.float64).reshape(-1) for name in sorted(left)])
    right_flat = np.concatenate([np.asarray(right[name], dtype=np.float64).reshape(-1) for name in sorted(right)])
    diff = left_flat - right_flat
    left_norm = float(np.dot(left_flat, left_flat))
    right_norm = float(np.dot(right_flat, right_flat))
    dot = float(np.dot(left_flat, right_flat))
    return {
        "cosine": dot / max(math.sqrt(left_norm) * math.sqrt(right_norm), np.finfo(np.float64).eps),
        "relativeMse": float(np.dot(diff, diff) / max(left_norm, np.finfo(np.float64).eps)),
    }


def extract_present(outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value for name, value in outputs.items() if name.startswith("present.")}


def benchmark_variant(tokenizer: Tokenizer, variant: str, prompt: str, steps: int, compressor: ResearchCacheCompressor) -> dict[str, object]:
    model_path = ensure_variant_files(variant)
    start = time.perf_counter()
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    init_ms = (time.perf_counter() - start) * 1000.0

    prompt_ids = tokenizer.encode(prompt).ids
    base_cache = init_empty_cache(session)
    first_feed = make_feed(session, prompt_ids, len(prompt_ids), base_cache)

    start = time.perf_counter()
    first_outputs = session.run(None, first_feed)
    first_ms = (time.perf_counter() - start) * 1000.0
    output_names = [out.name for out in session.get_outputs()]
    first_outputs = dict(zip(output_names, first_outputs))

    first_logits = last_logits(first_outputs["logits"])
    first_token = argmax(first_logits)
    raw_tokens = [first_token]
    turbo_tokens = [first_token]

    raw_cache = update_raw_cache(first_outputs)
    compressed_cache = update_compressed_cache(first_outputs, compressor)

    decode_raw_ms: list[float] = []
    decode_turbo_ms: list[float] = []
    encode_ms: list[float] = []
    decode_cache_ms: list[float] = []
    logit_cosine: list[float] = []
    logit_rmse: list[float] = []
    logit_max_abs: list[float] = []
    cache_cosine: list[float] = []
    cache_mse: list[float] = []
    per_step: list[dict[str, object]] = []

    prev_raw = first_token
    prev_turbo = first_token
    total_length = len(prompt_ids) + 1

    for step in range(1, steps):
        raw_feed = make_feed(session, [prev_raw], total_length, raw_cache)
        start = time.perf_counter()
        raw_values = session.run(None, raw_feed)
        decode_raw_ms.append((time.perf_counter() - start) * 1000.0)
        raw_outputs = dict(zip(output_names, raw_values))
        raw_logits = last_logits(raw_outputs["logits"])
        next_raw = argmax(raw_logits)
        raw_tokens.append(next_raw)

        start = time.perf_counter()
        turbo_feed_cache = decompress_cache(compressed_cache, compressor)
        decode_cache_ms.append((time.perf_counter() - start) * 1000.0)
        turbo_feed = make_feed(session, [prev_turbo], total_length, turbo_feed_cache)
        start = time.perf_counter()
        turbo_values = session.run(None, turbo_feed)
        decode_turbo_ms.append((time.perf_counter() - start) * 1000.0)
        turbo_outputs = dict(zip(output_names, turbo_values))
        turbo_logits = last_logits(turbo_outputs["logits"])
        next_turbo = argmax(turbo_logits)
        turbo_tokens.append(next_turbo)

        start = time.perf_counter()
        next_compressed_cache = update_compressed_cache(turbo_outputs, compressor)
        encode_ms.append((time.perf_counter() - start) * 1000.0)

        raw_present = extract_present(raw_outputs)
        turbo_present = extract_present(
            {
                k.replace("past_key_values.", "present."): v
                for k, v in decompress_cache(next_compressed_cache, compressor).items()
                if k.startswith("past_key_values.")
            }
        )
        stats_logits = compare_arrays(raw_logits, turbo_logits)
        stats_cache = compare_cache_maps(raw_present, turbo_present)

        logit_cosine.append(stats_logits["cosine"])
        logit_rmse.append(stats_logits["rmse"])
        logit_max_abs.append(stats_logits["maxAbs"])
        cache_cosine.append(stats_cache["cosine"])
        cache_mse.append(stats_cache["relativeMse"])

        raw_cache = update_raw_cache(raw_outputs)
        compressed_cache = next_compressed_cache
        prev_raw = next_raw
        prev_turbo = next_turbo
        per_step.append(
            {
                "step": step,
                "rawToken": next_raw,
                "turboToken": next_turbo,
                "tokenMatch": next_raw == next_turbo,
                "rawCacheKvBytes": cache_bytes(raw_cache)["kvBytes"],
                "turboCacheKvBytes": compressed_cache_bytes(compressed_cache),
                "logitCosine": stats_logits["cosine"],
                "logitRmse": stats_logits["rmse"],
                "logitMaxAbs": stats_logits["maxAbs"],
                "cacheRelativeMse": stats_cache["relativeMse"],
                "cacheCosine": stats_cache["cosine"],
            }
        )
        total_length += 1

    raw_cache_size = cache_bytes(raw_cache)
    turbo_cache_size = compressed_cache_bytes(compressed_cache)
    return {
        "variant": variant,
        "modelPath": str(model_path),
        "modelBytes": model_path.stat().st_size + Path(f"{model_path}_data").stat().st_size,
        "promptTokenCount": len(prompt_ids),
        "firstToken": first_token,
        "initMs": init_ms,
        "firstMs": first_ms,
        "rawTokens": raw_tokens,
        "turboTokens": turbo_tokens,
        "rawText": tokenizer.decode(raw_tokens, skip_special_tokens=True),
        "turboText": tokenizer.decode(turbo_tokens, skip_special_tokens=True),
        "exactTokenMatch": raw_tokens == turbo_tokens,
        "matchingPrefixLength": matching_prefix(raw_tokens, turbo_tokens),
        "rawCache": raw_cache_size,
        "turboCache": {
            "kvBytes": turbo_cache_size,
            "compressionRatio": raw_cache_size["kvBytes"] / max(turbo_cache_size, 1),
        },
        "averages": {
            "decodeBaselineMs": average(decode_raw_ms),
            "decodeTurboMs": average(decode_turbo_ms),
            "turboEncodeMs": average(encode_ms),
            "turboDecodeCacheMs": average(decode_cache_ms),
            "logitCosine": average(logit_cosine),
            "logitRmse": average(logit_rmse),
            "logitMaxAbs": average(logit_max_abs),
            "cacheRelativeMse": average(cache_mse),
            "cacheCosine": average(cache_cosine),
        },
        "perStep": per_step,
    }


def matching_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for lval, rval in zip(left, right):
        if lval != rval:
            break
        count += 1
    return count


def average(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def print_variant(result: dict[str, object]) -> None:
    print(f"prompt tokens: {result['promptTokenCount']}")
    print(f"raw text:   {json.dumps(result['rawText'])}")
    print(f"turbo text: {json.dumps(result['turboText'])}")
    print(f"token match: {result['exactTokenMatch']} (prefix {result['matchingPrefixLength']}/{len(result['rawTokens'])})")
    print(
        "kv bytes raw={raw} turbo={turbo} ratio={ratio:.2f}x".format(
            raw=result["rawCache"]["kvBytes"],
            turbo=result["turboCache"]["kvBytes"],
            ratio=result["turboCache"]["compressionRatio"],
        )
    )
    print(
        "avg decode ms raw={raw:.1f} turbo={turbo:.1f} encode={enc:.1f} decodeCache={dec:.1f}".format(
            raw=result["averages"]["decodeBaselineMs"],
            turbo=result["averages"]["decodeTurboMs"],
            enc=result["averages"]["turboEncodeMs"],
            dec=result["averages"]["turboDecodeCacheMs"],
        )
    )
    print(
        "avg drift logits cosine={cos:.6f} rmse={rmse:.6f} cache relMSE={mse:.3e}".format(
            cos=result["averages"]["logitCosine"],
            rmse=result["averages"]["logitRmse"],
            mse=result["averages"]["cacheRelativeMse"],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="q4,q4f32")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--mode", default="prod")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--key-bits", type=int)
    parser.add_argument("--value-bits", type=int)
    parser.add_argument("--key-mode")
    parser.add_argument("--value-mode")
    parser.add_argument("--keep-prefix", type=int, default=0)
    parser.add_argument("--keep-recent", type=int, default=0)
    parser.add_argument("--precondition", default="none", choices=["none", "mean_rms"])
    args = parser.parse_args()

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    tokenizer = ensure_tokenizer()
    key_codec = TurboQuantKVCodec(
        dim=args.dim,
        bits=args.key_bits or args.bits,
        mode=args.key_mode or args.mode,
        precondition=args.precondition,
    )
    value_codec = TurboQuantKVCodec(
        dim=args.dim,
        bits=args.value_bits or args.bits,
        mode=args.value_mode or args.mode,
        precondition=args.precondition,
    )
    compressor = ResearchCacheCompressor(
        key_codec=key_codec,
        value_codec=value_codec,
        keep_prefix=args.keep_prefix,
        keep_recent=args.keep_recent,
    )
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]

    results: dict[str, object] = {
        "modelId": MODEL_ID,
        "prompt": args.prompt,
        "steps": args.steps,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "turboquant": {
            "dim": args.dim,
            "keepPrefix": args.keep_prefix,
            "keepRecent": args.keep_recent,
            "key": {
                "mode": key_codec.mode,
                "bits": key_codec.bits,
                "codeBits": key_codec.code_bits,
                "precondition": key_codec.precondition,
            },
            "value": {
                "mode": value_codec.mode,
                "bits": value_codec.bits,
                "codeBits": value_codec.code_bits,
                "precondition": value_codec.precondition,
            },
        },
        "variants": {},
    }

    for variant in variants:
        print(f"\n=== Benchmarking {variant} ===")
        result = benchmark_variant(tokenizer, variant, args.prompt, args.steps, compressor)
        results["variants"][variant] = result
        print_variant(result)

    if len(variants) >= 2:
        left, right = variants[0], variants[1]
        left_result = results["variants"][left]
        right_result = results["variants"][right]
        results["crossPrecision"] = {
            f"{left}_vs_{right}": {
                "rawTokenMatch": left_result["rawTokens"] == right_result["rawTokens"],
                "matchingPrefixLength": matching_prefix(left_result["rawTokens"], right_result["rawTokens"]),
                "firstTokenMatch": left_result["firstToken"] == right_result["firstToken"],
                "rawTextLeft": left_result["rawText"],
                "rawTextRight": right_result["rawText"],
            }
        }

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = RESULT_ROOT / f"smoke-benchmark-{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
