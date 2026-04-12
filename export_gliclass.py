"""
GLiClass ONNX Export v2 -- Decomposed Architecture

The monolithic tracing approach collapsed logits because:
  1. _create_segment_ids() iterates batch items with .item() calls -> frozen by tracer
  2. _extract_class_features() uses data-dependent indexing (torch.where on class_token_index) -> frozen
  3. The tracer bakes the token routing from the dummy input as CONSTANTS

Fix: Export 3 small, fully-traceable ONNX graphs:
  1. encoder.onnx       -- DeBERTa backbone: (input_ids+segment_embeds, attention_mask) -> hidden_states
  2. projectors.onnx    -- text_projector + classes_projector: (text_embed, class_embeds) -> projected
  3. scorer.onnx        -- MLPScorer: (text_projected, classes_projected) -> logits

The data-dependent routing (segment ID creation, class token extraction, first-token pooling)
is done in Rust/Python at inference time, NOT inside the ONNX graph.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import pathlib
import onnxruntime as ort
from gliclass import GLiClassModel
from transformers import AutoTokenizer

# ──────────────────────────────────────────────
# Wrapper modules for clean tracing
# ──────────────────────────────────────────────

class EncoderWithSegments(nn.Module):
    """DeBERTa encoder that takes pre-computed inputs_embeds + segment_embeds."""
    def __init__(self, encoder_model, word_embeddings, segment_embeddings):
        super().__init__()
        self.encoder_model = encoder_model
        self.word_embeddings = word_embeddings
        self.segment_embeddings = segment_embeddings

    def forward(self, input_ids, attention_mask, segment_ids):
        token_embeds = self.word_embeddings(input_ids)
        segment_embeds = self.segment_embeddings(segment_ids)
        inputs_embeds = token_embeds + segment_embeds
        outputs = self.encoder_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state


class ProjectorHead(nn.Module):
    """Projects text and class embeddings independently."""
    def __init__(self, text_projector, classes_projector, dropout):
        super().__init__()
        self.text_projector = text_projector
        self.classes_projector = classes_projector
        self.dropout = dropout

    def forward(self, text_embedding, class_embeddings):
        # text_embedding: (batch, hidden)
        # class_embeddings: (batch, num_classes, hidden)
        text_proj = self.text_projector(text_embedding)
        text_proj = self.dropout(text_proj)
        class_proj = self.classes_projector(class_embeddings)
        return text_proj, class_proj


class ScorerHead(nn.Module):
    """MLPScorer: concatenates text+class, runs MLP -> logits."""
    def __init__(self, scorer):
        super().__init__()
        self.scorer = scorer

    def forward(self, text_projected, class_projected):
        # text_projected: (batch, hidden)
        # class_projected: (batch, num_classes, hidden)
        return self.scorer(text_projected, class_projected)


# ──────────────────────────────────────────────
# Export functions
# ──────────────────────────────────────────────

def export_encoder(model, out_dir, opset=18):
    print("\n=== Exporting encoder.onnx ===")
    inner = model.model  # GLiClassUniEncoder
    
    wrapper = EncoderWithSegments(
        inner.encoder_model,
        inner.encoder_model.get_input_embeddings(),
        inner.segment_embeddings,
    )
    wrapper.eval()

    batch, seq = 1, 64
    dummy_ids = torch.randint(1, 1000, (batch, seq), dtype=torch.long)
    dummy_mask = torch.ones(batch, seq, dtype=torch.long)
    dummy_segs = torch.zeros(batch, seq, dtype=torch.long)

    path = out_dir / "encoder.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask, dummy_segs),
        str(path),
        input_names=["input_ids", "attention_mask", "segment_ids"],
        output_names=["hidden_states"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "segment_ids": {0: "batch", 1: "seq"},
            "hidden_states": {0: "batch", 1: "seq"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  Saved: {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    return path


def export_projectors(model, out_dir, opset=18):
    print("\n=== Exporting projectors.onnx ===")
    inner = model.model

    wrapper = ProjectorHead(
        inner.text_projector,
        inner.classes_projector,
        inner.dropout,
    )
    wrapper.eval()

    hidden = model.config.hidden_size
    batch, n_classes = 1, 3
    dummy_text = torch.randn(batch, hidden)
    dummy_classes = torch.randn(batch, n_classes, hidden)

    path = out_dir / "projectors.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_text, dummy_classes),
        str(path),
        input_names=["text_embedding", "class_embeddings"],
        output_names=["text_projected", "class_projected"],
        dynamic_axes={
            "text_embedding": {0: "batch"},
            "class_embeddings": {0: "batch", 1: "num_classes"},
            "text_projected": {0: "batch"},
            "class_projected": {0: "batch", 1: "num_classes"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  Saved: {path} ({path.stat().st_size / 1024:.1f} KB)")
    return path


def export_scorer(model, out_dir, opset=18):
    print("\n=== Exporting scorer.onnx ===")
    inner = model.model

    wrapper = ScorerHead(inner.scorer)
    wrapper.eval()

    hidden = model.config.hidden_size
    batch, n_classes = 1, 3
    dummy_text = torch.randn(batch, hidden)
    dummy_classes = torch.randn(batch, n_classes, hidden)

    path = out_dir / "scorer.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_text, dummy_classes),
        str(path),
        input_names=["text_projected", "class_projected"],
        output_names=["logits"],
        dynamic_axes={
            "text_projected": {0: "batch"},
            "class_projected": {0: "batch", 1: "num_classes"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  Saved: {path} ({path.stat().st_size / 1024:.1f} KB)")
    return path


# ──────────────────────────────────────────────
# Verification: run the decomposed pipeline in Python
# to confirm it matches PyTorch baseline
# ──────────────────────────────────────────────

def verify_against_pytorch(model, tokenizer, out_dir):
    """Run the same input through PyTorch and decomposed ONNX, compare logits."""
    from gliclass import ZeroShotClassificationPipeline
    from gliclass.pipeline import UniEncoderZeroShotClassificationPipeline

    text = "NASA launched a new Mars rover to search for signs of ancient life."
    labels = ["space", "politics", "sports", "technology", "health"]

    # -- PyTorch baseline --
    pipeline = ZeroShotClassificationPipeline(model, tokenizer, classification_type='multi-label', device='cpu')
    pt_results = pipeline(text, labels, threshold=0.0)[0]
    pt_scores = {r["label"]: r["score"] for r in pt_results}
    print("\n=== PyTorch Baseline ===")
    for lbl, score in pt_scores.items():
        print(f"  {lbl}: {score:.4f}")

    # -- ONNX decomposed pipeline --
    config = model.config
    CLASS_TOKEN_ID = config.class_token_index   # 128001
    TEXT_TOKEN_ID = config.text_token_index      # 128002

    # Use the UniEncoder pipeline's prepare_inputs to get exact same tokenized input
    uni_pipeline = UniEncoderZeroShotClassificationPipeline(
        model, tokenizer, classification_type='multi-label', device='cpu'
    )
    tokenized = uni_pipeline.prepare_inputs([text], labels, same_labels=True)
    input_ids = tokenized["input_ids"].cpu()
    attention_mask = tokenized["attention_mask"].cpu()

    # Build segment_ids (replicate _create_segment_ids logic)
    segment_ids = torch.zeros_like(input_ids)
    for b in range(input_ids.shape[0]):
        text_positions = (input_ids[b] == TEXT_TOKEN_ID).nonzero(as_tuple=True)[0]
        if len(text_positions) > 0:
            text_start = text_positions[0].item()
            segment_ids[b, text_start:] = 1
    
    # Extract class token positions
    class_positions_list = []
    for b in range(input_ids.shape[0]):
        positions = (input_ids[b] == CLASS_TOKEN_ID).nonzero(as_tuple=True)[0].tolist()
        class_positions_list.append(positions)

    print(f"\n  Input shape: {input_ids.shape}")
    print(f"  Class token positions: {class_positions_list}")

    # Session creation
    enc_sess = ort.InferenceSession(str(out_dir / "encoder.onnx"), providers=["CPUExecutionProvider"])
    proj_sess = ort.InferenceSession(str(out_dir / "projectors.onnx"), providers=["CPUExecutionProvider"])
    scorer_sess = ort.InferenceSession(str(out_dir / "scorer.onnx"), providers=["CPUExecutionProvider"])

    # Step 1: Encoder
    hidden_states = enc_sess.run(
        None,
        {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
            "segment_ids": segment_ids.numpy(),
        },
    )[0]  # (batch, seq, hidden)

    # Step 2: Extract class embeddings and text embedding (first-token pooling)
    batch_size = hidden_states.shape[0]
    hidden_size = hidden_states.shape[2]
    
    # First-token pooling for text (extract_text_features=False, pooler=FirstTokenPooling1D)
    text_embedding = hidden_states[:, 0, :]  # (batch, hidden)
    
    # Class token extraction (embed_class_token=True => use class token position directly)
    num_classes = len(labels)
    class_embeddings = np.zeros((batch_size, num_classes, hidden_size), dtype=np.float32)
    for b in range(batch_size):
        for c_idx, pos in enumerate(class_positions_list[b]):
            class_embeddings[b, c_idx] = hidden_states[b, pos]

    # Step 3: Projectors
    text_proj, class_proj = proj_sess.run(
        None,
        {
            "text_embedding": text_embedding.astype(np.float32),
            "class_embeddings": class_embeddings,
        },
    )

    # Step 4: Scorer
    logits = scorer_sess.run(
        None,
        {
            "text_projected": text_proj,
            "class_projected": class_proj,
        },
    )[0]  # (batch, num_classes)

    print("\n=== ONNX Decomposed ===")
    for i, lbl in enumerate(labels):
        score = 1.0 / (1.0 + np.exp(-logits[0, i]))  # sigmoid
        print(f"  {lbl}: {score:.4f}")

    # Check correlation
    pt_vec = np.array([pt_scores[l] for l in labels])
    onnx_vec = 1.0 / (1.0 + np.exp(-logits[0]))
    correlation = np.corrcoef(pt_vec, onnx_vec)[0, 1]
    print(f"\n  Pearson correlation: {correlation:.4f}")
    if correlation > 0.95:
        print("  PASS -- ONNX output matches PyTorch baseline!")
    else:
        print("  WARN -- correlation is low, investigating...")


# ──────────────────────────────────────────────
# Save config JSON for Rust runner
# ──────────────────────────────────────────────

def save_config(model, tokenizer, out_dir):
    config = model.config
    runtime_config = {
        "class_token_index": config.class_token_index,
        "text_token_index": config.text_token_index,
        "embed_class_token": config.embed_class_token,
        "extract_text_features": config.extract_text_features,
        "use_segment_embeddings": config.use_segment_embeddings,
        "hidden_size": config.hidden_size,
        "scorer_type": config.scorer_type,
        "normalize_features": config.normalize_features,
        "architecture_type": config.architecture_type,
        "onnx_files": {
            "encoder": "encoder.onnx",
            "projectors": "projectors.onnx",
            "scorer": "scorer.onnx",
        }
    }
    path = out_dir / "gliclass_config.json"
    with open(path, "w") as f:
        json.dump(runtime_config, f, indent=2)
    print(f"\n  Config saved: {path}")


def main():
    model_id = "knowledgator/gliclass-instruct-base-v1.0"
    out_dir = pathlib.Path("gliclass-instruct-onnx-v2")
    out_dir.mkdir(exist_ok=True)

    print(f"Loading {model_id}...")
    model = GLiClassModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()

    # Export the 3 decomposed graphs
    export_encoder(model, out_dir)
    export_projectors(model, out_dir)
    export_scorer(model, out_dir)

    # Save tokenizer + config
    tokenizer.save_pretrained(str(out_dir))
    save_config(model, tokenizer, out_dir)

    # Verify
    verify_against_pytorch(model, tokenizer, out_dir)

    print("\n=== DONE ===")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
