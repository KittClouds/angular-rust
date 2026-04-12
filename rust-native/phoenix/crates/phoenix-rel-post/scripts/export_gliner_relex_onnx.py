import argparse
import shutil
from pathlib import Path

import torch
from gliner import GLiNER


CARD_TEXT = (
    "The Eiffel Tower, located in Paris, France, was designed by engineer "
    "Gustave Eiffel and completed in 1889."
)
CARD_ENTITIES = ["location", "person", "date", "structure"]
CARD_RELATIONS = ["located in", "designed by", "completed in"]


class RelexExportWrapper(torch.nn.Module):
    def __init__(self, core, threshold: float, adjacency_threshold: float):
        super().__init__()
        self.core = core
        self.threshold = threshold
        self.adjacency_threshold = adjacency_threshold

    def forward(self, input_ids, attention_mask, words_mask, text_lengths):
        out = self.core(
            input_ids=input_ids,
            attention_mask=attention_mask,
            words_mask=words_mask,
            text_lengths=text_lengths,
            threshold=self.threshold,
            adjacency_threshold=self.adjacency_threshold,
        )
        return out.logits, out.rel_idx, out.rel_logits, out.rel_mask, out.entity_spans


def build_batch(model, text: str, entity_labels: list[str], relation_labels: list[str]):
    tokens, _, _ = model.prepare_inputs([text])
    input_x = model.prepare_base_input(tokens)
    collator = model.data_collator_class(
        model.config,
        data_processor=model.data_processor,
        return_tokens=True,
        return_entities=True,
        return_id_to_classes=True,
        return_rel_id_to_classes=True,
        prepare_labels=False,
    )
    batch = collator(input_x, entity_types=entity_labels, relation_types=relation_labels)
    return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def copy_model_assets(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("gliner_config.json", "tokenizer.json", "tokenizer_config.json"):
        path = src / name
        if path.exists():
            shutil.copy2(path, dst / name)


def main():
    parser = argparse.ArgumentParser(description="Export GLiNER-relex with live relation prompts.")
    parser.add_argument("--model-root", default="gliner-relex-onnx")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--adjacency-threshold", type=float, default=0.3)
    parser.add_argument("--text", default=CARD_TEXT)
    parser.add_argument("--entity-labels", default=",".join(CARD_ENTITIES))
    parser.add_argument("--relation-labels", default=",".join(CARD_RELATIONS))
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    model_root = Path(args.model_root)
    out_dir = Path(args.out_dir)
    entity_labels = [v.strip() for v in args.entity_labels.split(",") if v.strip()]
    relation_labels = [v.strip() for v in args.relation_labels.split(",") if v.strip()]

    model = GLiNER.from_pretrained(str(model_root), load_tokenizer=True, map_location="cpu")
    model.eval()
    batch = build_batch(model, args.text, entity_labels, relation_labels)
    wrapper = RelexExportWrapper(model.model.to("cpu").eval(), args.threshold, args.adjacency_threshold)

    copy_model_assets(model_root, out_dir)
    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(
        wrapper,
        (
            batch["input_ids"],
            batch["attention_mask"],
            batch["words_mask"],
            batch["text_lengths"],
        ),
        f=str(onnx_path),
        input_names=["input_ids", "attention_mask", "words_mask", "text_lengths"],
        output_names=["logits", "rel_idx", "rel_logits", "rel_mask", "entity_spans"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "words_mask": {0: "batch_size", 1: "sequence_length"},
            "text_lengths": {0: "batch_size", 1: "value"},
            "logits": {0: "batch_size", 1: "word_count", 2: "num_ent_classes"},
            "rel_idx": {0: "batch_size", 1: "num_pairs"},
            "rel_logits": {0: "batch_size", 1: "num_pairs", 2: "num_rel_classes"},
            "rel_mask": {0: "batch_size", 1: "num_pairs"},
            "entity_spans": {0: "batch_size", 1: "num_entities"},
        },
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"exported {onnx_path}")

    if args.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quant_path = out_dir / "model_quantized.onnx"
        quantize_dynamic(str(onnx_path), str(quant_path), weight_type=QuantType.QUInt8)
        print(f"quantized {quant_path}")


if __name__ == "__main__":
    main()
