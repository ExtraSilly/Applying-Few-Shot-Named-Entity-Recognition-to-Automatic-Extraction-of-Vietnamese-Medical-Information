"""
Few-shot NER bang LLM (Claude API) de so sanh voi PhoBERT fine-tuning

Usage:
    python llm_ner.py --k 5                  # 5-shot prompting
    python llm_ner.py --k 0                  # zero-shot
    python llm_ner.py --k 5 --n-eval 200     # chi eval 200 cau test
    python llm_ner.py --dry-run              # test prompt khong goi API

Can set bien moi truong:
    set ANTHROPIC_API_KEY=sk-ant-...
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent

LABEL_LIST = [
    "AGE", "DATE", "GENDER", "JOB", "LOCATION", "NAME",
    "ORGANIZATION", "PATIENT_ID", "SYMPTOM_AND_DISEASE", "TRANSPORTATION",
]

ENTITY_DESCRIPTIONS = {
    "AGE":                   "tuoi cua benh nhan (vi du: 43 tuoi, 67)",
    "DATE":                  "ngay thang (vi du: 18/3, ngay 14-4, thang 7)",
    "GENDER":                "gioi tinh (vi du: nam, nu)",
    "JOB":                   "nghe nghiep, chuc danh (vi du: phi cong, bac si, dieu duong)",
    "LOCATION":              "dia danh, dia diem (vi du: Ha Noi, san bay Tan Son Nhat)",
    "NAME":                  "ten nguoi (vi du: L.T.H., Nguyen Van A)",
    "ORGANIZATION":          "to chuc, co quan (vi du: Bo Y te, Benh vien Bach Mai)",
    "PATIENT_ID":            "ma so benh nhan (vi du: 91, 188, 523)",
    "SYMPTOM_AND_DISEASE":   "trieu chung, benh ly (vi du: sot cao, kho tho, than man)",
    "TRANSPORTATION":        "phuong tien di chuyen (vi du: chuyen bay VN0054, tau hoa)",
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_entities_from_item(item):
    """Tra ve dict {entity_type: [list of entity texts]}."""
    result = defaultdict(list)
    words, tags = item["words"], item["tags"]
    i = 0
    while i < len(tags):
        if tags[i].startswith("B-"):
            etype = tags[i][2:]
            span  = [words[i]]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{etype}":
                span.append(words[j])
                j += 1
            result[etype].append(" ".join(span))
            i = j
        else:
            i += 1
    return dict(result)


def format_shot(item):
    """Dinh dang 1 example cho prompt."""
    text     = " ".join(item["words"])
    entities = extract_entities_from_item(item)
    ent_str  = json.dumps(entities, ensure_ascii=False) if entities else "{}"
    return f'Input: "{text}"\nOutput: {ent_str}'


def build_prompt(test_sentence, shots):
    """Xay dung prompt few-shot."""
    entity_defs = "\n".join(
        f"- {k}: {v}" for k, v in ENTITY_DESCRIPTIONS.items()
    )

    shot_block = ""
    if shots:
        shot_block = "\n\nVi du:\n" + "\n\n".join(format_shot(s) for s in shots)

    prompt = f"""Ban la he thong nhan dang thuc the co ten (NER) cho van ban COVID-19 tieng Viet.

Cac loai thuc the:
{entity_defs}

Chi trich xuat thuc the co trong van ban. Tra ve JSON object, moi key la ten loai thuc the, value la list cac thuc the tim thay. Neu khong co thuc the nao, tra ve {{}}.{shot_block}

Bay gio hay nhan dang thuc the trong cau sau:
Input: "{test_sentence}"
Output:"""
    return prompt


def parse_llm_output(raw):
    """Parse JSON tu output cua LLM."""
    raw = raw.strip()
    # Tim JSON block
    match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def entities_to_bio(words, entity_dict):
    """Chuyen entity_dict thanh chuoi BIO tags."""
    tags = ["O"] * len(words)

    for etype, ent_list in entity_dict.items():
        if etype not in LABEL_LIST:
            continue
        for ent_text in ent_list:
            ent_words = ent_text.split()
            n = len(ent_words)
            for i in range(len(words) - n + 1):
                if words[i:i+n] == ent_words and tags[i] == "O":
                    tags[i] = f"B-{etype}"
                    for j in range(1, n):
                        tags[i+j] = f"I-{etype}"
                    break
    return tags


def compute_span_f1(true_items, pred_items):
    """Tinh Precision, Recall, F1 o cap do span."""
    tp = fp = fn = 0
    for true, pred in zip(true_items, pred_items):
        true_spans = set()
        pred_spans = set()

        # Extract spans tu true
        w = true["words"]
        t = true["tags"]
        i = 0
        while i < len(t):
            if t[i].startswith("B-"):
                etype = t[i][2:]
                j = i + 1
                while j < len(t) and t[j] == f"I-{etype}":
                    j += 1
                true_spans.add((i, j, etype))
                i = j
            else:
                i += 1

        # Extract spans tu pred
        pt = pred
        i = 0
        while i < len(pt):
            if pt[i].startswith("B-"):
                etype = pt[i][2:]
                j = i + 1
                while j < len(pt) and pt[j] == f"I-{etype}":
                    j += 1
                pred_spans.add((i, j, etype))
                i = j
            else:
                i += 1

        tp += len(true_spans & pred_spans)
        fp += len(pred_spans - true_spans)
        fn += len(true_spans - pred_spans)

    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f, tp, fp, fn


def get_shots(support_data, k, seed=42):
    """Lay k examples tu support data, phan phoi deu theo entity type."""
    rng = random.Random(seed)
    selected = []
    per_type = defaultdict(list)
    for item in support_data:
        ents = extract_entities_from_item(item)
        for etype in ents:
            per_type[etype].append(item)

    shots_per_type = max(1, k // len(LABEL_LIST))
    seen = set()
    for etype in LABEL_LIST:
        candidates = [i for i in per_type.get(etype, [])
                      if id(i) not in seen]
        for item in rng.sample(candidates, min(shots_per_type, len(candidates))):
            seen.add(id(item))
            selected.append(item)
        if len(selected) >= k:
            break

    return selected[:k]


def main():
    parser = argparse.ArgumentParser(description="LLM Few-shot NER")
    parser.add_argument("--k", type=int, default=5,
                        help="So shot (0 = zero-shot)")
    parser.add_argument("--n-eval", type=int, default=100,
                        help="So cau test de danh gia (< 3000 de tiet kiem API)")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001",
                        help="Claude model ID")
    parser.add_argument("--dry-run", action="store_true",
                        help="In prompt mau, khong goi API")
    args = parser.parse_args()

    # -- Load data
    support_data = load_jsonl(BASE / "data/few_shot/k5/support.json")
    test_data    = load_jsonl(BASE / "data/word/test_word.json")

    rng = random.Random(42)
    rng.shuffle(test_data)
    eval_data = test_data[:args.n_eval]

    shots = get_shots(support_data, args.k) if args.k > 0 else []
    print(f"Mode: {args.k}-shot | Eval sentences: {len(eval_data)} | Model: {args.model}")

    # -- Dry run
    if args.dry_run:
        sample = eval_data[0]
        prompt = build_prompt(" ".join(sample["words"]), shots)
        print("\n--- SAMPLE PROMPT ---")
        print(prompt[:2000])
        print("...")
        return

    # -- Setup Anthropic client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY truoc khi chay.")
        print("  Windows: set ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        print("Cai dat: pip install anthropic")
        sys.exit(1)

    # -- Inference
    all_true  = []
    all_preds = []
    errors    = 0

    for i, item in enumerate(eval_data):
        sentence = " ".join(item["words"])
        prompt   = build_prompt(sentence, shots)

        try:
            response = client.messages.create(
                model=args.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
        except Exception as e:
            print(f"  [WARN] API error on sentence {i}: {e}")
            errors += 1
            raw = "{}"

        entity_dict = parse_llm_output(raw)
        pred_tags   = entities_to_bio(item["words"], entity_dict)

        all_true.append(item)
        all_preds.append(pred_tags)

        if (i + 1) % 10 == 0:
            p, r, f, *_ = compute_span_f1(all_true, all_preds)
            print(f"  [{i+1}/{len(eval_data)}] running F1={f:.4f}  P={p:.4f}  R={r:.4f}")

    # -- Final results
    p, r, f, tp, fp, fn = compute_span_f1(all_true, all_preds)
    print(f"\n{'='*50}")
    print(f"LLM {args.k}-shot NER Results ({args.model})")
    print(f"{'='*50}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"Micro F1:  {f:.4f}")
    print(f"TP={tp}  FP={fp}  FN={fn}  API_errors={errors}")

    # Per-entity stats
    per_type_tp = defaultdict(int)
    per_type_fp = defaultdict(int)
    per_type_fn = defaultdict(int)
    for true, pred in zip(all_true, all_preds):
        w = true["words"]
        t = true["tags"]
        # true spans
        true_spans = set()
        i = 0
        while i < len(t):
            if t[i].startswith("B-"):
                etype = t[i][2:]
                j = i + 1
                while j < len(t) and t[j] == f"I-{etype}":
                    j += 1
                true_spans.add((i, j, etype))
                i = j
            else:
                i += 1
        # pred spans
        pred_spans = set()
        i = 0
        while i < len(pred):
            if pred[i].startswith("B-"):
                etype = pred[i][2:]
                j = i + 1
                while j < len(pred) and pred[j] == f"I-{etype}":
                    j += 1
                pred_spans.add((i, j, etype))
                i = j
            else:
                i += 1
        for span in true_spans & pred_spans:
            per_type_tp[span[2]] += 1
        for span in pred_spans - true_spans:
            per_type_fp[span[2]] += 1
        for span in true_spans - pred_spans:
            per_type_fn[span[2]] += 1

    print("\nPer-entity F1:")
    for etype in LABEL_LIST:
        tp_ = per_type_tp[etype]
        fp_ = per_type_fp[etype]
        fn_ = per_type_fn[etype]
        p_  = tp_ / (tp_ + fp_) if tp_ + fp_ > 0 else 0
        r_  = tp_ / (tp_ + fn_) if tp_ + fn_ > 0 else 0
        f_  = 2*p_*r_/(p_+r_) if p_+r_ > 0 else 0
        sup = tp_ + fn_
        print(f"  {etype:<25}  F1={f_:.4f}  P={p_:.4f}  R={r_:.4f}  support={sup}")

    # Luu ket qua
    out_dir = BASE / "output"
    out_dir.mkdir(exist_ok=True)
    result = {
        "method": f"LLM_{args.k}shot",
        "model": args.model,
        "n_eval": len(eval_data),
        "micro_f1": round(f, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
    }
    out_path = out_dir / f"llm_{args.k}shot_results.json"
    with open(out_path, "w", encoding="utf-8") as fp_:
        json.dump(result, fp_, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
