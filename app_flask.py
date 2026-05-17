"""
Flask web app cho Few-shot NER COVID-19 tieng Viet

Chay: python app_flask.py
Truy cap: http://localhost:5000
"""

import json
import numpy as np
import torch
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForTokenClassification

try:
    from underthesea import word_tokenize as _word_tokenize
    _HAS_UNDERTHESEA = True
except ImportError:
    _HAS_UNDERTHESEA = False

# ── Config ────────────────────────────────────────────────────────────────────
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

ENTITY_COLORS = {
    "LOCATION":            "#AED6F1",
    "PATIENT_ID":          "#FAD7A0",
    "DATE":                "#A9DFBF",
    "SYMPTOM_AND_DISEASE": "#F1948A",
    "ORGANIZATION":        "#D7BDE2",
    "AGE":                 "#85C1E9",
    "GENDER":              "#F9E79F",
    "NAME":                "#A3E4D7",
    "TRANSPORTATION":      "#D5D8DC",
    "JOB":                 "#F0B27A",
}

ENTITY_VI = {
    "LOCATION":            "Dia diem",
    "PATIENT_ID":          "Ma benh nhan",
    "DATE":                "Ngay thang",
    "SYMPTOM_AND_DISEASE": "Trieu chung/Benh",
    "ORGANIZATION":        "To chuc",
    "AGE":                 "Tuoi",
    "GENDER":              "Gioi tinh",
    "NAME":                "Ten nguoi",
    "TRANSPORTATION":      "Phuong tien",
    "JOB":                 "Nghe nghiep",
}
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# Global model (load 1 lan khi khoi dong)
tokenizer = None
model     = None
device    = None
model_info = ""


def segment(text: str) -> str:
    if _HAS_UNDERTHESEA:
        return _word_tokenize(text, format="text")
    return text


def find_checkpoint():
    candidates = sorted(
        list(BASE.glob("output/baseline_full_seed*/checkpoint-*")) +
        list(BASE.glob("output/baseline_full/checkpoint-*")),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def load_model():
    global tokenizer, model, device, model_info
    ckpt = find_checkpoint()
    if ckpt is None:
        model_info = "Chua co model"
        return False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer  = AutoTokenizer.from_pretrained(str(ckpt))
    model      = AutoModelForTokenClassification.from_pretrained(str(ckpt)).to(device)
    model.eval()
    model_info = str(ckpt.name)
    print(f"[OK] Model loaded: {ckpt}  ({device})")
    return True


def predict(text):
    words      = text.strip().split()
    input_ids  = [tokenizer.cls_token_id]
    word_spans = []

    for word in words:
        subs = tokenizer.encode(word, add_special_tokens=False)
        if not subs:
            continue
        if len(input_ids) + len(subs) >= MAX_LEN - 1:
            break
        start = len(input_ids)
        input_ids.extend(subs)
        word_spans.append((start, word))

    input_ids.append(tokenizer.sep_token_id)

    with torch.no_grad():
        out = model(
            input_ids=torch.tensor([input_ids]).to(device),
            attention_mask=torch.ones(1, len(input_ids), dtype=torch.long).to(device),
        )

    pred_ids = np.argmax(out.logits[0].cpu().numpy(), axis=-1)

    result = []
    for start, word in word_spans:
        tag = ID2LABEL[pred_ids[start]] if start < len(pred_ids) else "O"
        result.append({"word": word, "tag": tag})

    return result


def extract_entities(word_tags):
    entities = []
    i = 0
    while i < len(word_tags):
        tag = word_tags[i]["tag"]
        if tag.startswith("B-"):
            etype = tag[2:]
            span  = [word_tags[i]["word"]]
            j = i + 1
            while j < len(word_tags) and word_tags[j]["tag"] == f"I-{etype}":
                span.append(word_tags[j]["word"])
                j += 1
            entities.append({
                "text":    " ".join(span),
                "type":    etype,
                "type_vi": ENTITY_VI.get(etype, etype),
                "color":   ENTITY_COLORS.get(etype, "#eee"),
            })
            i = j
        else:
            i += 1
    return entities


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template(
        "index.html",
        entity_colors=ENTITY_COLORS,
        entity_vi=ENTITY_VI,
        model_info=model_info,
        model_ready=(model is not None),
    )


@app.route("/predict", methods=["POST"])
def predict_api():
    if model is None:
        return jsonify({"error": "Model chua duoc tai. Chay train truoc."}), 503

    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "Thieu truong 'text'"}), 400

    segmented = segment(text)
    word_tags = predict(segmented)
    entities  = extract_entities(word_tags)

    return jsonify({
        "word_tags":    word_tags,
        "entities":     entities,
        "colors":       ENTITY_COLORS,
        "entity_vi":    ENTITY_VI,
        "segmented":    segmented,
        "auto_segment": _HAS_UNDERTHESEA,
    })


@app.route("/status")
def status():
    return jsonify({
        "model_ready":  model is not None,
        "model_info":   model_info,
        "device":       str(device) if device else "none",
        "auto_segment": _HAS_UNDERTHESEA,
    })


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading model...")
    ok = load_model()
    if not ok:
        print("[WARN] Model not found. Run: python train_baseline.py --mode full")
    app.run(debug=False, host="0.0.0.0", port=5000)
