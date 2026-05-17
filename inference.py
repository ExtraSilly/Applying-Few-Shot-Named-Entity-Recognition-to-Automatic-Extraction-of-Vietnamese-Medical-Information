"""
Demo inference: nhan dien thuc the trong cau bat ky

Usage:
    python inference.py
    python inference.py --model output/baseline_full_seed42
    python inference.py --text "Benh nhan 91 la phi cong nguoi Anh, 43 tuoi"
"""

import argparse
import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

BASE = Path(__file__).parent

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
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
MAX_LEN  = 128

COLORS = {
    "LOCATION":           "\033[94m",   # xanh duong
    "PATIENT_ID":         "\033[93m",   # vang
    "DATE":               "\033[96m",   # xanh nhat
    "SYMPTOM_AND_DISEASE":"\033[91m",   # do
    "ORGANIZATION":       "\033[95m",   # tim
    "AGE":                "\033[92m",   # xanh la
    "GENDER":             "\033[97m",   # trang
    "NAME":               "\033[33m",   # cam
    "TRANSPORTATION":     "\033[36m",   # xanh biet
    "JOB":                "\033[35m",   # tim nhat
}
RESET = "\033[0m"


def find_checkpoint(model_dir):
    """Tim checkpoint moi nhat trong model_dir."""
    checkpoints = sorted(
        Path(model_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    return checkpoints[-1] if checkpoints else Path(model_dir)


def predict(text, tokenizer, model, device):
    """Tra ve list (word, predicted_tag) cho mot cau da tach tu."""
    words = text.strip().split()

    input_ids  = [tokenizer.cls_token_id]
    word_spans = []   # (start_idx, end_idx) trong input_ids

    for word in words:
        subs = tokenizer.encode(word, add_special_tokens=False)
        if not subs:
            continue
        start = len(input_ids)
        input_ids.extend(subs[:MAX_LEN - len(input_ids) - 1])
        word_spans.append((start, len(input_ids)))
        if len(input_ids) >= MAX_LEN - 1:
            break

    input_ids.append(tokenizer.sep_token_id)
    input_ids = input_ids[:MAX_LEN]

    with torch.no_grad():
        output = model(
            input_ids=torch.tensor([input_ids]).to(device),
            attention_mask=torch.ones(1, len(input_ids), dtype=torch.long).to(device),
        )

    pred_ids = np.argmax(output.logits[0].cpu().numpy(), axis=-1)

    result = []
    for i, (word, (start, end)) in enumerate(zip(words, word_spans)):
        if start < len(pred_ids):
            tag = ID2LABEL[pred_ids[start]]
        else:
            tag = "O"
        result.append((word, tag))

    return result


def extract_entities(word_tags):
    """Gom cac token thanh entities theo BIO."""
    entities = []
    i = 0
    while i < len(word_tags):
        word, tag = word_tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            span  = [word]
            j = i + 1
            while j < len(word_tags) and word_tags[j][1] == f"I-{etype}":
                span.append(word_tags[j][0])
                j += 1
            entities.append((" ".join(span), etype))
            i = j
        else:
            i += 1
    return entities


def colorize(word_tags):
    """In cau voi mau sac theo entity type."""
    out = []
    i = 0
    while i < len(word_tags):
        word, tag = word_tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            span  = [word]
            j = i + 1
            while j < len(word_tags) and word_tags[j][1] == f"I-{etype}":
                span.append(word_tags[j][0])
                j += 1
            color = COLORS.get(etype, "")
            out.append(f"{color}[{' '.join(span)}|{etype}]{RESET}")
            i = j
        else:
            out.append(word)
            i += 1
    return " ".join(out)


def main():
    parser = argparse.ArgumentParser(description="NER inference demo")
    parser.add_argument("--model", type=str,
                        default=None,
                        help="Thu muc chua model (mac dinh: tu dong tim)")
    parser.add_argument("--text", type=str,
                        default=None,
                        help="Cau can nhan dien (neu bo qua se vao che do tuong tac)")
    args = parser.parse_args()

    # Tim model
    if args.model:
        model_path = find_checkpoint(args.model)
    else:
        # Tim checkpoint moi nhat trong output/
        candidates = sorted(BASE.glob("output/baseline_full_seed*/checkpoint-*"),
                            key=lambda p: p.stat().st_mtime)
        if candidates:
            model_path = candidates[-1]
        else:
            print("Khong tim thay model. Chay: python train_baseline.py --mode full")
            return

    print(f"Loading model: {model_path}")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    model.eval()
    print(f"Model loaded on {device}\n")

    if args.text:
        # Single text mode
        word_tags = predict(args.text, tokenizer, model, device)
        print("Input :", args.text)
        print("Output:", colorize(word_tags))
        print("\nEntities:")
        for text, etype in extract_entities(word_tags):
            print(f"  [{etype}] {text}")
    else:
        # Interactive mode
        print("Nhap cau can nhan dien (cau da tach tu, Enter de thoat):")
        print("-" * 55)
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                break
            word_tags = predict(text, tokenizer, model, device)
            print(colorize(word_tags))
            entities = extract_entities(word_tags)
            if entities:
                for ent_text, etype in entities:
                    print(f"  -> [{etype}] {ent_text}")
            print()


if __name__ == "__main__":
    main()
