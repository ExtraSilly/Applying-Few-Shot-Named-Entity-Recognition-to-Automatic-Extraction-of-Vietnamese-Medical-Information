"""
PhoBERT + CRF cho Few-shot NER
Cai tien so voi train_baseline.py: them CRF layer de giam BOUNDARY errors

Usage:
    python train_crf.py --mode full
    python train_crf.py --mode k20 --seed 42
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, set_seed
from TorchCRF import CRF
from seqeval.metrics import classification_report, f1_score

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "vinai/phobert-base"
BASE        = Path(__file__).parent
MAX_LEN     = 128
BATCH_SIZE  = 16      # CRF tieu thu nhieu VRAM hon, giam batch size
EPOCHS_FULL = 5
EPOCHS_FEW  = 30
LR          = 3e-5    # CRF hop voi LR nho hon mot chut

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
NUM_LABELS = len(LABEL_LIST)
# ─────────────────────────────────────────────────────────────────────────────


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return super().default(obj)


# ── Model ─────────────────────────────────────────────────────────────────────
class PhoBERTCRF(nn.Module):
    """PhoBERT encoder + Linear + CRF decoder (TorchCRF API)."""

    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        # TorchCRF: CRF(num_labels, pad_idx=None, use_gpu=True)
        self.crf = CRF(num_labels, use_gpu=torch.cuda.is_available())

    def forward(self, input_ids, attention_mask, labels=None):
        hidden = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state
        emissions = self.classifier(self.dropout(hidden))  # (B, T, C)

        # Mask: True cho token hop le (khong phai padding, khong phai -100)
        if labels is not None:
            mask = (labels != -100) & attention_mask.bool()
        else:
            mask = attention_mask.bool()

        if labels is not None:
            # Doi -100 thanh 0 (O) de CRF khong bi loi
            crf_labels = labels.clone()
            crf_labels[~mask] = 0
            # TorchCRF.forward tra ve tensor [batch] → lay mean lam loss
            loss = -self.crf(emissions, crf_labels, mask).mean()
            return loss
        else:
            # Viterbi decoding → list of lists
            return self.crf.viterbi_decode(emissions, mask)


# ── Dataset ───────────────────────────────────────────────────────────────────
class NERDataset(TorchDataset):
    def __init__(self, data, tokenizer):
        self.samples = []
        for item in data:
            enc = _encode(item["words"], item["tags"], tokenizer)
            self.samples.append(enc)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _encode(words, tags, tokenizer):
    input_ids  = [tokenizer.cls_token_id]
    label_ids  = [-100]

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

    pad_len = MAX_LEN - len(input_ids)
    attn    = [1] * len(input_ids) + [0] * pad_len
    input_ids  = input_ids  + [tokenizer.pad_token_id] * pad_len
    label_ids  = label_ids  + [-100] * pad_len

    return {
        "input_ids":      torch.tensor(input_ids,  dtype=torch.long),
        "attention_mask": torch.tensor(attn,        dtype=torch.long),
        "labels":         torch.tensor(label_ids,   dtype=torch.long),
    }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    true_labels, true_preds = [], []

    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"]

            pred_seqs = model(ids, mask)   # list of lists (Viterbi)

            for pred_seq, label_row in zip(pred_seqs, lbls.tolist()):
                tl, tp = [], []
                valid_positions = [i for i, l in enumerate(label_row) if l != -100]
                for pos in valid_positions:
                    tl.append(ID2LABEL[label_row[pos]])
                    tp.append(ID2LABEL[pred_seq[pos]] if pos < len(pred_seq) else "O")
                true_labels.append(tl)
                true_preds.append(tp)

    return f1_score(true_labels, true_preds), true_labels, true_preds


# ── Main ──────────────────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main(mode, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*55}")
    print(f"  PhoBERT-CRF | Mode: {mode} | Seed: {seed} | {device}")
    print(f"{'='*55}\n")

    if mode == "full":
        train_path = BASE / "data/word/train_word.json"
        out_dir    = BASE / f"output/crf_full_seed{seed}"
        epochs     = EPOCHS_FULL
    else:
        train_path = BASE / f"data/few_shot/{mode}/support.json"
        out_dir    = BASE / f"output/crf_{mode}_seed{seed}"
        epochs     = EPOCHS_FEW

    train_data = load_jsonl(train_path)
    dev_data   = load_jsonl(BASE / "data/word/dev_word.json")
    test_data  = load_jsonl(BASE / "data/word/test_word.json")
    print(f"Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = PhoBERTCRF(MODEL_NAME, NUM_LABELS).to(device)

    train_ds = NERDataset(train_data, tokenizer)
    dev_ds   = NERDataset(dev_data,   tokenizer)
    test_ds  = NERDataset(test_data,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate)
    dev_loader   = DataLoader(dev_ds,   batch_size=64,          shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=64,          shuffle=False, collate_fn=collate)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 50, total_steps)
    scaler = GradScaler()

    best_dev_f1, best_epoch = 0.0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)

            optimizer.zero_grad()
            with autocast():
                loss = model(ids, mask, lbls)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        dev_f1, _, _ = evaluate(model, dev_loader, device)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:02d}/{epochs}  loss={avg_loss:.4f}  dev_f1={dev_f1:.4f}")

        if dev_f1 > best_dev_f1:
            best_dev_f1, best_epoch = dev_f1, epoch
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    # -- Final eval on test
    print(f"\nBest dev F1: {best_dev_f1:.4f} (epoch {best_epoch})")
    print("--- Final evaluation on TEST set ---")
    model.load_state_dict(torch.load(out_dir / "best_model.pt"))
    test_f1, true_labels, true_preds = evaluate(model, test_loader, device)
    print(classification_report(true_labels, true_preds, digits=4))

    report = classification_report(true_labels, true_preds, digits=4, output_dict=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"mode": mode, "seed": seed, "model": "PhoBERT-CRF",
                   "train_size": len(train_data), "report": report},
                  f, indent=2, cls=NumpyEncoder)
    print(f"Saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhoBERT + CRF NER")
    parser.add_argument("--mode", default="full",
                        choices=["full", "k1", "k5", "k10", "k20"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.mode, args.seed)
