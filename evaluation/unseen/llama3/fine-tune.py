"""
QLoRA fine-tuning of Meta-Llama-3.1-8B-Instruct for multi-class vendor classification.
- 4-bit NF4 quantization + LoRA (r=8, all-linear)
- Microbatch 1 × gradient-accumulation 8 → effective batch size 8
- Gradient checkpointing enabled
- Right-truncation at 1024 tokens
- Targeted loss masking: only the Vendor value tokens are supervised
"""
import os
import glob
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerBase,
)

# ──────────────────────────── Paths ────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = os.path.join(SCRIPT_DIR, "Meta-Llama-3.1-8B-Instruct")
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")

# ──────────────────────────── Prompt Template ────────────────────────────
INSTRUCTION_TEMPLATE = """\
<|im_start|>user
Below is information about a device. Think step-by-step, then predict the vendor. 
Make sure the explanations are meaningful, easy to understand, succinct, and grounded in the input data.
Respond with a JSON object containing your explanation and predicted vendor.

The JSON must have two keys: "Explanation" and "Vendor".
Do not include placeholders like '<explanation>' or '...'.

Device Metadata:
{metadata}
<|im_end|>
<|im_start|>assistant
{{"Explanation": "Based on the device network metadata characteristics.", "Vendor": "{vendor}"}}
<|im_end|>"""

VENDOR_MARKER = '"Vendor": "'


# ──────────────────────────── Data helpers ────────────────────────────
def load_hf_dataset():
    """Load all finetune CSVs via HuggingFace load_dataset (Arrow, memory-mapped)."""
    csv_files = sorted(glob.glob(os.path.join(DATASET_DIR, "finetune_data_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No finetune_data CSVs found in {DATASET_DIR}")
    for f in csv_files:
        print(f"  Found {os.path.basename(f)}")
    ds = load_dataset("csv", data_files=csv_files, split="train")
    print(f"  Total training samples: {len(ds)}")
    return ds


def format_and_tokenize(examples, tokenizer, max_length=1024):
    """
    Single-pass: build instruction text from raw columns, tokenize, and create
    vendor-only labels.  Avoids materialising all texts in memory at once.
    """
    exclude = {"ip", "vendor"}
    feature_cols = [k for k in examples.keys() if k not in exclude]

    all_input_ids, all_attn, all_labels = [], [], []
    batch_size = len(examples["vendor"])

    for i in range(batch_size):
        # ── build metadata ──
        parts = []
        for col in feature_cols:
            val = examples[col][i]
            if val is None:
                continue
            val = str(val)
            if val and val != "nan" and val != "":
                parts.append(f"{col}: {val}")
        metadata = "\n".join(parts)
        vendor = str(examples["vendor"][i])
        text = INSTRUCTION_TEMPLATE.format(metadata=metadata, vendor=vendor)

        # ── tokenize ──
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        labels = [-100] * len(input_ids)

        # ── vendor-only supervision ──
        marker_pos = text.rfind(VENDOR_MARKER)
        if marker_pos >= 0:
            v_start = marker_pos + len(VENDOR_MARKER)
            v_end = text.find('"', v_start)
            if v_end >= 0:
                for j, (s, e) in enumerate(offsets):
                    if s >= v_start and e <= v_end and s < e:
                        labels[j] = input_ids[j]

        all_input_ids.append(input_ids)
        all_attn.append(enc["attention_mask"])
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attn,
        "labels": all_labels,
    }


# ──────────────────────────── Collator ────────────────────────────
@dataclass
class VendorPaddingCollator:
    """Pad input_ids / attention_mask / labels to the longest sequence in the batch."""
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attn, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ──────────────────────────── Main ────────────────────────────
def main():
    # ── Tokenizer ──
    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset (Arrow memory-mapped → low RAM) ──
    print("Loading dataset …")
    raw_ds = load_hf_dataset()
    all_cols = raw_ds.column_names
    dataset = raw_ds.map(
        lambda ex: format_and_tokenize(ex, tokenizer, max_length=1024),
        batched=True,
        batch_size=256,
        remove_columns=all_cols,
        desc="Formatting + tokenizing",
    )
    print(f"  Tokenized dataset: {len(dataset)} samples")

    # ── QLoRA 4-bit quantization ──
    print("Loading model with 4-bit quantization …")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA adapters ──
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training arguments ──
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,          # effective batch size = 8
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2e-4,
        fp16=True,
        logging_dir=LOG_DIR,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        report_to="none",
    )

    # ── Trainer ──
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=VendorPaddingCollator(tokenizer=tokenizer),
    )

    print("Starting training …")
    trainer.train()

    # ── Save final model + tokenizer ──
    save_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Training complete. Model saved to {save_path}")


if __name__ == "__main__":
    main()