from gliner import GLiNER
import torch

model = GLiNER.from_pretrained("knowledgator/gliner-bi-base-v2.0")
text = "Barack Obama visited the White House."
labels = ["Person", "Location", "Organization"]

tokens, starts, ends = model.prepare_inputs([text])
input_x = model.prepare_base_input(tokens)

collator = model.data_collator_class(
    model.config, data_processor=model.data_processor,
    return_tokens=True, return_entities=True,
    return_id_to_classes=True, prepare_labels=False,
)
batch = collator(input_x, entity_types=labels)

print("=== Batch keys ===")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
    else:
        tp = type(v).__name__
        print(f"  {k}: {tp}")

print()
for k in ["input_ids", "attention_mask", "words_mask", "text_lengths"]:
    if k in batch:
        print(f"{k}: {batch[k].tolist()}")

print()
lid = "labels_input_ids"
lam = "labels_attention_mask"
if lid in batch:
    print(f"{lid}: {batch[lid].tolist()}")
    print(f"{lam}: {batch[lam].tolist()}")
else:
    print("NO labels_input_ids in batch")

print()
print(f"span_idx: shape={list(batch['span_idx'].shape)}")
print(f"span_mask: shape={list(batch['span_mask'].shape)}")
print()
print(f"config max_width={model.config.max_width}")
print(f"config class_token_index={model.config.class_token_index}")
print(f"config words_splitter_type={model.config.words_splitter_type}")
print(f"data_processor type={type(model.data_processor).__name__}")
print(f"tokenizer vocab size={model.data_processor.transformer_tokenizer.vocab_size}")
