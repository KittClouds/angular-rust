#!/usr/bin/env python3
"""
Export a cross-encoder NLI model to ONNX for phoenix-rel-post.

Default model:
    cross-encoder/nli-deberta-v3-small

Outputs:
    <out>/
      tokenizer.json
      tokenizer_config.json
      special_tokens_map.json
      config.json
      export_metadata.json
      onnx/model.onnx
      onnx/model.onnx_data   (if external data is used)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from optimum.exporters.onnx.__main__ import main_export
from transformers import AutoConfig, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _find_label_index(id2label: dict[int, str], target: str) -> int | None:
    target = target.lower()
    for idx, label in id2label.items():
        if label.lower() == target:
            return int(idx)
    return None


def export_nli(repo_id: str, out_dir: Path, opset: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    print(f"[nli-export] loading tokenizer/config for {repo_id}")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    config = AutoConfig.from_pretrained(repo_id)

    tokenizer.save_pretrained(str(out_dir))
    config.save_pretrained(str(out_dir))

    print("[nli-export] exporting ONNX with optimum")
    try:
        main_export(
            model_name_or_path=repo_id,
            output=onnx_dir,
            task="text-classification",
            opset=opset,
            device="cpu",
            do_validation=False,
        )
    except FileNotFoundError as exc:
        if not (onnx_dir / "model.onnx").exists():
            raise
        print(f"[nli-export] cleanup warning: {exc}")

    metadata = {
        "repo_id": repo_id,
        "contradiction_idx": _find_label_index(config.id2label, "contradiction"),
        "entailment_idx": _find_label_index(config.id2label, "entailment"),
        "neutral_idx": _find_label_index(config.id2label, "neutral"),
        "max_length": getattr(config, "max_position_embeddings", None),
    }
    (out_dir / "export_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"[nli-export] wrote bundle to {out_dir}")
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(out_dir)
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  - {rel} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export NLI cross-encoder to ONNX")
    parser.add_argument(
        "--repo",
        default="cross-encoder/nli-deberta-v3-small",
        help="HF repo id (default: cross-encoder/nli-deberta-v3-small)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output directory for the exported bundle",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset (default: 17)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="remove the existing output directory before exporting",
    )
    args = parser.parse_args()

    if args.clean and args.out.exists():
        shutil.rmtree(args.out, ignore_errors=True)

    export_nli(args.repo, args.out, args.opset)


if __name__ == "__main__":
    main()
