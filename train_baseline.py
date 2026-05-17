"""
Few-shot NER voi PhoBERT tren PhoNER_COVID19

Usage:
    python train_baseline.py --mode full        # train toan bo
    python train_baseline.py --mode k5          # few-shot 5-shot
    python train_baseline.py --mode k5 --seed 1 # voi seed khac
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    set_seed,
)
from seqeval.metrics import classification_report, f1_score

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "vinai/phobert-base"
BASE        = Path(__file__).parent
MAX_LEN     = 128
BATCH_SIZE  = 32
EPOCHS_FULL = 5
EPOCHS_FEW  = 30
LR          = 5e-5

LABEL_LIST = [
    "O",
    "B-AGE",          "I-AGE",
    "B-DATE",         "I-DATE",
    "B-GENDER",       "I-GENDER",
    "B-JOB",          "I-JOB",
    "B-LOCATION",     "I-LOCATION",
    "B-NAME",         "I-NAME",
    "B-ORGANIZATION", "I-ORGANIZATION",
    "B-PATIENT_ID",   "I-PATIENT_ID",
    "B-SYMPTOM_AND_DISEASE", "I-SYMPTOM_AND_DISEASE",
    "B-TRANSPORTATION", "I-TRANSPORTATION",
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL  = {i: l for l, i in LABEL2ID.items()}
# ─────────────────────────────────────────────────────────────────────────────


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return super().default(obj)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_and_align(examples, tokenizer):
    """BIO label alignment tuong thich voi slow tokenizer cua PhoBERT."""
    all_input_ids, all_attention_mask, all_labels = [], [], []

    for words, tags in zip(examples["words"], examples["tags"]):
        input_ids = [tokenizer.cls_token_id]
        label_ids = [-100]

        for word, tag in zip(words, tags):
            sub = tokenizer.encode(word, add_special_tokens=False)
            if not sub:
                continue
            if len(input_ids) + len(sub) > MAX_LEN - 1:
                break
            input_ids.extend(sub)
            label_ids.append(LABEL2ID.get(tag, 0))
            if len(sub) > 1:
                i_tag = ("I-" + tag[2:]) if tag.startswith("B-") else tag
                label_ids.extend([LABEL2ID.get(i_tag, 0)] * (len(sub) - 1))

        input_ids.append(tokenizer.sep_token_id)
        label_ids.append(-100)

        pad_len        = MAX_LEN - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids      = input_ids + [tokenizer.pad_token_id] * pad_len
        label_ids      = label_ids + [-100] * pad_len

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(label_ids)

    return {
        "input_ids":      all_input_ids,
        "attention_mask": all_attention_mask,
        "labels":         all_labels,
    }


def decode_predictions(preds_ids, label_ids):
    """Giai ma predictions va labels, bo qua token dac biet (-100)."""
    true_labels, true_preds = [], []
    for pred_row, label_row in zip(preds_ids, label_ids):
        tl, tp = [], []
        for p, l in zip(pred_row, label_row):
            if l != -100:
                tl.append(ID2LABEL[l])
                tp.append(ID2LABEL[p])
        true_labels.append(tl)
        true_preds.append(tp)
    return true_labels, true_preds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    true_labels, true_preds = decode_predictions(preds, labels)
    return {"f1": f1_score(true_labels, true_preds)}


def build_dataset(data, tokenizer):
    hf = Dataset.from_list(data)
    return hf.map(
        lambda ex: tokenize_and_align(ex, tokenizer),
        batched=True,
        remove_columns=["words", "tags"],
    )


def main(mode, seed):
    set_seed(seed)

    print(f"\n{'='*55}")
    print(f"  Mode: {mode}  |  Seed: {seed}  |  Model: {MODEL_NAME}")
    print(f"{'='*55}\n")

    # -- Duong dan du lieu
    if mode == "full":
        train_path = BASE / "data/word/train_word.json"
        out_dir    = BASE / f"output/baseline_full_seed{seed}"
        epochs     = EPOCHS_FULL
    else:
        train_path = BASE / f"data/few_shot/{mode}/support.json"
        out_dir    = BASE / f"output/baseline_{mode}_seed{seed}"
        epochs     = EPOCHS_FEW

    # Dev dung de chon best model trong khi train (tranh data leakage tu test)
    dev_path  = BASE / "data/word/dev_word.json"
    test_path = BASE / "data/word/test_word.json"

    train_data = load_jsonl(train_path)
    dev_data   = load_jsonl(dev_path)
    test_data  = load_jsonl(test_path)
    print(f"Train : {len(train_data)} sentences")
    print(f"Dev   : {len(dev_data)} sentences  (dung de chon best model)")
    print(f"Test  : {len(test_data)} sentences  (danh gia cuoi cung)")

    # -- Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # -- Build datasets
    train_ds = build_dataset(train_data, tokenizer)
    dev_ds   = build_dataset(dev_data,   tokenizer)
    test_ds  = build_dataset(test_data,  tokenizer)

    collator = DataCollatorForTokenClassification(tokenizer)

    # -- Training arguments
    # Chi luu checkpoint khi train full (few-shot khong luu de tiet kiem disk)
    is_full = (mode == "full")
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch" if is_full else "no",
        load_best_model_at_end=is_full,
        metric_for_best_model="f1",
        logging_steps=50,
        seed=seed,
        data_seed=seed,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,       # dev set — khong phai test set
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # -- Train
    trainer.train()

    # -- Danh gia cuoi cung tren TEST set
    print("\n--- Final evaluation on TEST set ---")
    preds_output = trainer.predict(test_ds)
    preds = np.argmax(preds_output.predictions, axis=-1)
    true_labels, true_preds = decode_predictions(preds, preds_output.label_ids)

    print(classification_report(true_labels, true_preds, digits=4))

    # -- Luu ket qua
    report     = classification_report(true_labels, true_preds, digits=4, output_dict=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {"mode": mode, "seed": seed, "train_size": len(train_data), "report": report},
            f, indent=2, cls=NumpyEncoder,
        )
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-shot NER voi PhoBERT")
    parser.add_argument("--mode", default="full",
                        choices=["full", "k1", "k5", "k10", "k20"],
                        help="Training setting")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()
    main(args.mode, args.seed)
