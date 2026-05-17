"""
Error Analysis: phan tich loi cua mo hinh baseline (full) tren test set
Phan loai loi va xuat bao cao chi tiet
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

BASE       = Path(__file__).parent
MODEL_DIR  = BASE / "output/baseline_full"
TEST_PATH  = BASE / "data/word/test_word.json"
OUT_PATH   = BASE / "output/error_analysis.json"
MAX_LEN    = 128

LABEL_LIST = [
    "O","B-AGE","I-AGE","B-DATE","I-DATE","B-GENDER","I-GENDER",
    "B-JOB","I-JOB","B-LOCATION","I-LOCATION","B-NAME","I-NAME",
    "B-ORGANIZATION","I-ORGANIZATION","B-PATIENT_ID","I-PATIENT_ID",
    "B-SYMPTOM_AND_DISEASE","I-SYMPTOM_AND_DISEASE",
    "B-TRANSPORTATION","I-TRANSPORTATION",
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL  = {i: l for l, i in LABEL2ID.items()}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def predict_sentence(words, tokenizer, model, device):
    """Predict NER tags for a single sentence."""
    input_ids  = [tokenizer.cls_token_id]
    word_spans = []  # (start, end) in input_ids per word

    for word in words:
        subs = tokenizer.encode(word, add_special_tokens=False)
        if not subs:
            continue
        start = len(input_ids)
        input_ids.extend(subs)
        word_spans.append((start, start + len(subs)))
        if len(input_ids) >= MAX_LEN - 1:
            break

    input_ids.append(tokenizer.sep_token_id)
    input_ids = input_ids[:MAX_LEN]

    with torch.no_grad():
        out = model(
            input_ids=torch.tensor([input_ids]).to(device),
            attention_mask=torch.ones(1, len(input_ids), dtype=torch.long).to(device),
        )
    logits = out.logits[0].cpu().numpy()
    pred_ids = np.argmax(logits, axis=-1)

    pred_tags = []
    for start, end in word_spans:
        if start < len(pred_ids):
            pred_tags.append(ID2LABEL[pred_ids[start]])
        else:
            pred_tags.append("O")

    # align length with original words (in case of truncation)
    pred_tags = pred_tags[:len(words)]
    while len(pred_tags) < len(words):
        pred_tags.append("O")

    return pred_tags


def extract_entities(words, tags):
    """Extract list of (entity_text, entity_type, start_idx) from BIO tags."""
    entities = []
    i = 0
    while i < len(tags):
        if tags[i].startswith("B-"):
            etype = tags[i][2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{etype}":
                j += 1
            text = " ".join(words[i:j])
            entities.append({"text": text, "type": etype, "start": i, "end": j})
            i = j
        else:
            i += 1
    return entities


def classify_error(gold_ents, pred_ents, words):
    """
    Phan loai loi:
    - MISSED:       entity that nhung model khong tim thay
    - SPURIOUS:     model du doan nhung khong co trong gold
    - TYPE_ERROR:   span dung nhung sai loai entity
    - BOUNDARY:     type dung nhung span sai (partial match)
    - CORRECT:      du doan dung
    """
    errors = []

    gold_spans = {(e["start"], e["end"]): e for e in gold_ents}
    pred_spans = {(e["start"], e["end"]): e for e in pred_ents}

    matched_pred = set()

    for (gs, ge), g in gold_spans.items():
        if (gs, ge) in pred_spans:
            p = pred_spans[(gs, ge)]
            if p["type"] == g["type"]:
                pass  # correct
            else:
                errors.append({
                    "error_type": "TYPE_ERROR",
                    "gold_text":  g["text"],
                    "gold_type":  g["type"],
                    "pred_type":  p["type"],
                    "context":    " ".join(words[max(0,gs-3):ge+3]),
                })
            matched_pred.add((gs, ge))
        else:
            # Check boundary error: overlap with any pred span
            overlap = [(ps, pe) for (ps, pe) in pred_spans
                       if ps < ge and pe > gs and (ps, pe) not in matched_pred]
            if overlap:
                ps, pe = overlap[0]
                p = pred_spans[(ps, pe)]
                if p["type"] == g["type"]:
                    errors.append({
                        "error_type": "BOUNDARY",
                        "gold_text":  g["text"],
                        "pred_text":  p["text"],
                        "entity_type": g["type"],
                        "context":    " ".join(words[max(0,gs-3):pe+3]),
                    })
                else:
                    errors.append({
                        "error_type": "TYPE_AND_BOUNDARY",
                        "gold_text":  g["text"],
                        "pred_text":  p["text"],
                        "gold_type":  g["type"],
                        "pred_type":  p["type"],
                        "context":    " ".join(words[max(0,gs-3):pe+3]),
                    })
                matched_pred.add((ps, pe))
            else:
                errors.append({
                    "error_type": "MISSED",
                    "gold_text":  g["text"],
                    "gold_type":  g["type"],
                    "context":    " ".join(words[max(0,gs-3):ge+3]),
                })

    for (ps, pe), p in pred_spans.items():
        if (ps, pe) not in matched_pred and (ps, pe) not in gold_spans:
            errors.append({
                "error_type": "SPURIOUS",
                "pred_text":  p["text"],
                "pred_type":  p["type"],
                "context":    " ".join(words[max(0,ps-3):pe+3]),
            })

    return errors


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Find best checkpoint
    checkpoints = sorted(MODEL_DIR.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[1]))
    ckpt = checkpoints[-1] if checkpoints else MODEL_DIR
    print(f"Loading model from: {ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForTokenClassification.from_pretrained(ckpt).to(device)
    model.eval()

    test_data = load_jsonl(TEST_PATH)
    print(f"Test sentences: {len(test_data)}")

    all_errors = []
    error_counter = Counter()
    per_type_errors = defaultdict(list)

    for item in test_data:
        words     = item["words"]
        gold_tags = item["tags"]
        pred_tags = predict_sentence(words, tokenizer, model, device)

        gold_ents = extract_entities(words, gold_tags)
        pred_ents = extract_entities(words, pred_tags)

        errors = classify_error(gold_ents, pred_ents, words)
        for e in errors:
            error_counter[e["error_type"]] += 1
            gt = e.get("gold_type", e.get("entity_type", e.get("pred_type", "?")))
            per_type_errors[gt].append(e)
            all_errors.append(e)

    # Summary
    print("\n=== Error Type Distribution ===")
    total = sum(error_counter.values())
    for etype, cnt in error_counter.most_common():
        print(f"  {etype:<20} {cnt:>5}  ({cnt/total*100:.1f}%)")

    print("\n=== Errors per Entity Type ===")
    for etype in sorted(per_type_errors.keys()):
        errs = per_type_errors[etype]
        ec = Counter(e["error_type"] for e in errs)
        print(f"  {etype:<25} total={len(errs):>4}  {dict(ec)}")

    # Save
    report = {
        "total_errors": total,
        "error_distribution": dict(error_counter),
        "per_entity_type": {k: [e for e in v[:5]] for k, v in per_type_errors.items()},
        "sample_errors": all_errors[:50],
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
