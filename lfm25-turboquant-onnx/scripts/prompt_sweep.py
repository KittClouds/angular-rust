from __future__ import annotations

import json
import time
from pathlib import Path

import onnxruntime as ort

import smoke_benchmark as sb

ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = ROOT / "config" / "prompt_sweep_prompts.json"


def make_profile(name: str) -> dict[str, object]:
    if name == "uniform_prod4":
        return {
            "name": name,
            "compressor": sb.ResearchCacheCompressor(
                key_codec=sb.TurboQuantKVCodec(dim=64, bits=4, mode="prod"),
                value_codec=sb.TurboQuantKVCodec(dim=64, bits=4, mode="prod"),
                keep_prefix=0,
                keep_recent=0,
            ),
        }
    if name == "research_asym4_sinks":
        return {
            "name": name,
            "compressor": sb.ResearchCacheCompressor(
                key_codec=sb.TurboQuantKVCodec(dim=64, bits=4, mode="prod"),
                value_codec=sb.TurboQuantKVCodec(dim=64, bits=4, mode="mse"),
                keep_prefix=1,
                keep_recent=1,
            ),
        }
    raise ValueError(f"unknown profile {name}")


def benchmark_prompt(
    session: ort.InferenceSession,
    output_names: list[str],
    tokenizer,
    prompt: str,
    steps: int,
    compressor: sb.ResearchCacheCompressor,
) -> dict[str, object]:
    prompt_ids = tokenizer.encode(prompt).ids
    base_cache = sb.init_empty_cache(session)
    first_feed = sb.make_feed(session, prompt_ids, len(prompt_ids), base_cache)
    first_outputs = dict(zip(output_names, session.run(None, first_feed)))

    first_logits = sb.last_logits(first_outputs["logits"])
    first_token = sb.argmax(first_logits)
    raw_tokens = [first_token]
    turbo_tokens = [first_token]
    raw_cache = sb.update_raw_cache(first_outputs)
    compressed_cache = sb.update_compressed_cache(first_outputs, compressor)

    logit_cosine = []
    logit_rmse = []
    cache_mse = []
    prev_raw = first_token
    prev_turbo = first_token
    total_length = len(prompt_ids) + 1

    for _step in range(1, steps):
        raw_feed = sb.make_feed(session, [prev_raw], total_length, raw_cache)
        raw_outputs = dict(zip(output_names, session.run(None, raw_feed)))
        raw_logits = sb.last_logits(raw_outputs["logits"])
        next_raw = sb.argmax(raw_logits)
        raw_tokens.append(next_raw)

        turbo_feed_cache = sb.decompress_cache(compressed_cache, compressor)
        turbo_feed = sb.make_feed(session, [prev_turbo], total_length, turbo_feed_cache)
        turbo_outputs = dict(zip(output_names, session.run(None, turbo_feed)))
        turbo_logits = sb.last_logits(turbo_outputs["logits"])
        next_turbo = sb.argmax(turbo_logits)
        turbo_tokens.append(next_turbo)

        next_compressed_cache = sb.update_compressed_cache(turbo_outputs, compressor)
        raw_present = sb.extract_present(raw_outputs)
        turbo_present = sb.extract_present(
            {
                k.replace("past_key_values.", "present."): v
                for k, v in sb.decompress_cache(next_compressed_cache, compressor).items()
                if k.startswith("past_key_values.")
            }
        )

        logits_stats = sb.compare_arrays(raw_logits, turbo_logits)
        cache_stats = sb.compare_cache_maps(raw_present, turbo_present)
        logit_cosine.append(logits_stats["cosine"])
        logit_rmse.append(logits_stats["rmse"])
        cache_mse.append(cache_stats["relativeMse"])

        raw_cache = sb.update_raw_cache(raw_outputs)
        compressed_cache = next_compressed_cache
        prev_raw = next_raw
        prev_turbo = next_turbo
        total_length += 1

    raw_cache_size = sb.cache_bytes(raw_cache)
    turbo_cache_size = sb.compressed_cache_bytes(compressed_cache)
    return {
        "prompt": prompt,
        "rawText": tokenizer.decode(raw_tokens, skip_special_tokens=True),
        "turboText": tokenizer.decode(turbo_tokens, skip_special_tokens=True),
        "rawTokens": raw_tokens,
        "turboTokens": turbo_tokens,
        "exactTokenMatch": raw_tokens == turbo_tokens,
        "matchingPrefixLength": sb.matching_prefix(raw_tokens, turbo_tokens),
        "compressionRatio": raw_cache_size["kvBytes"] / max(turbo_cache_size, 1),
        "rawCacheKvBytes": raw_cache_size["kvBytes"],
        "turboCacheKvBytes": turbo_cache_size,
        "avgLogitCosine": sb.average(logit_cosine),
        "avgLogitRmse": sb.average(logit_rmse),
        "avgCacheRelativeMse": sb.average(cache_mse),
    }


def aggregate_prompt_results(results: list[dict[str, object]]) -> dict[str, object]:
    exact_matches = sum(1 for item in results if item["exactTokenMatch"])
    return {
        "promptCount": len(results),
        "exactMatchCount": exact_matches,
        "exactMatchRate": exact_matches / max(len(results), 1),
        "avgMatchingPrefix": sb.average([item["matchingPrefixLength"] for item in results]),
        "avgCompressionRatio": sb.average([item["compressionRatio"] for item in results]),
        "avgLogitCosine": sb.average([item["avgLogitCosine"] for item in results]),
        "avgLogitRmse": sb.average([item["avgLogitRmse"] for item in results]),
        "avgCacheRelativeMse": sb.average([item["avgCacheRelativeMse"] for item in results]),
        "failedPrompts": [
            {
                "prompt": item["prompt"],
                "rawText": item["rawText"],
                "turboText": item["turboText"],
                "matchingPrefixLength": item["matchingPrefixLength"],
            }
            for item in results
            if not item["exactTokenMatch"]
        ],
    }


def main() -> None:
    tokenizer = sb.ensure_tokenizer()
    prompts = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    variants = ["q4", "q4f32"]
    profiles = [make_profile("uniform_prod4"), make_profile("research_asym4_sinks")]
    steps = 4

    bundle: dict[str, object] = {
        "modelId": sb.MODEL_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": steps,
        "promptFile": str(PROMPT_FILE),
        "prompts": prompts,
        "profiles": {},
    }

    for profile in profiles:
        profile_name = profile["name"]
        compressor = profile["compressor"]
        bundle["profiles"][profile_name] = {"variants": {}}
        print(f"\n=== Profile {profile_name} ===")

        for variant in variants:
            model_path = sb.ensure_variant_files(variant)
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            output_names = [out.name for out in session.get_outputs()]
            prompt_results = [
                benchmark_prompt(session, output_names, tokenizer, prompt, steps, compressor)
                for prompt in prompts
            ]
            aggregate = aggregate_prompt_results(prompt_results)
            bundle["profiles"][profile_name]["variants"][variant] = {
                "aggregate": aggregate,
                "prompts": prompt_results,
            }
            print(
                f"{variant}: exact {aggregate['exactMatchCount']}/{aggregate['promptCount']} "
                f"avgPrefix={aggregate['avgMatchingPrefix']:.2f} "
                f"ratio={aggregate['avgCompressionRatio']:.2f}x "
                f"logitCos={aggregate['avgLogitCosine']:.4f}"
            )

    out_name = f"prompt-sweep-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    out_path = sb.RESULT_ROOT / out_name
    out_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved prompt sweep to {out_path}")


if __name__ == "__main__":
    main()
