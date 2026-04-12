import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic


WEIGHT_TYPES = {
    "qint8": QuantType.QInt8,
    "quint8": QuantType.QUInt8,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize an existing GLiNER relex ONNX export with alternate recipes."
    )
    parser.add_argument("--model-input", required=True, help="Path to the fp32 ONNX model.")
    parser.add_argument("--model-output", required=True, help="Path to the quantized ONNX model.")
    parser.add_argument(
        "--weight-type",
        default="qint8",
        choices=sorted(WEIGHT_TYPES),
        help="Dynamic weight quantization type.",
    )
    parser.add_argument(
        "--op-type",
        action="append",
        dest="op_types",
        help="Operator type to quantize. Repeat to pass multiple values.",
    )
    parser.add_argument(
        "--exclude-node",
        action="append",
        default=[],
        help="Exact node name to exclude from quantization. Repeat as needed.",
    )
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=[],
        help="Case-insensitive substring pattern for node names to exclude.",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Enable per-channel weight quantization when supported.",
    )
    parser.add_argument(
        "--reduce-range",
        action="store_true",
        help="Enable reduced-range quantization.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run ORT quantization pre-processing before quantizing.",
    )
    parser.add_argument(
        "--preprocess-output",
        help="Optional path for the pre-processed ONNX file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve exclusions and print the recipe without writing a model.",
    )
    return parser.parse_args()


def unique_keep_order(values):
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def resolve_excluded_nodes(model_path: Path, exact_names: list[str], patterns: list[str]) -> list[str]:
    excluded = list(exact_names)
    if not patterns:
        return unique_keep_order(excluded)

    model = onnx.load(str(model_path), load_external_data=False)
    lowered_patterns = [pattern.lower() for pattern in patterns if pattern]
    for node in model.graph.node:
        node_name = node.name or ""
        if not node_name:
            continue
        if any(pattern in node_name.lower() for pattern in lowered_patterns):
            excluded.append(node_name)
    return unique_keep_order(excluded)


def main():
    args = parse_args()
    model_input = Path(args.model_input)
    model_output = Path(args.model_output)
    quant_input = model_input
    excluded_nodes = resolve_excluded_nodes(
        model_input,
        args.exclude_node,
        args.exclude_pattern,
    )

    print(f"model_input={model_input}")
    print(f"model_output={model_output}")
    print(f"weight_type={args.weight_type}")
    print(f"op_types={args.op_types or ['<ort-default>']}")
    print(f"per_channel={args.per_channel}")
    print(f"reduce_range={args.reduce_range}")
    print(f"preprocess={args.preprocess}")
    print(f"excluded_nodes={len(excluded_nodes)}")
    for node_name in excluded_nodes[:32]:
        print(f"  exclude: {node_name}")
    if len(excluded_nodes) > 32:
        print(f"  ... {len(excluded_nodes) - 32} more")

    if args.dry_run:
        return

    model_output.parent.mkdir(parents=True, exist_ok=True)
    if args.preprocess:
        preprocess_output = (
            Path(args.preprocess_output)
            if args.preprocess_output
            else model_output.with_name(f"{model_output.stem}_preprocessed.onnx")
        )
        preprocess_output.parent.mkdir(parents=True, exist_ok=True)
        quant_pre_process(
            input_model=str(model_input),
            output_model_path=str(preprocess_output),
            save_as_external_data=False,
        )
        quant_input = preprocess_output
        print(f"preprocessed_model={quant_input}")
    quantize_dynamic(
        str(quant_input),
        str(model_output),
        op_types_to_quantize=args.op_types,
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
        weight_type=WEIGHT_TYPES[args.weight_type],
        nodes_to_exclude=excluded_nodes or None,
    )
    print("quantization complete")


if __name__ == "__main__":
    main()
