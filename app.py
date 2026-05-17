"""
Web demo cho Few-shot NER COVID-19 tieng Viet
Chay: streamlit run app.py
"""

import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
import streamlit as st

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
    "TRANSPORTATION":      "#ABB2B9",
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

EXAMPLES = [
    "Benh_nhan 91 , nam phi_cong nguoi Anh , 43 tuoi , dang dieu_tri tai Benh_vien Cho_Ray tu ngay 18/3 .",
    "Ca nhiem thu 188 la nu , 20 tuoi , du_hoc_sinh vua tro ve tu Anh , hien cach_ly tai Ha_Noi .",
    "Mot dieu_duong 24 tuoi cua Benh_vien Bach_Mai da duoc xac_nhan nhiem COVID-19 sau khi tiep_xuc voi benh_nhan 133 .",
    "Benh_nhan 523 va chong la benh_nhan 522 , 67 tuoi , duoc Bo_Y_te ghi_nhan nhiem nCoV hom 31/7 .",
    "Ca nhiem moi nhat la nam giao_vien 45 tuoi , cu_tru tai quan Hai_Ba_Trung , Ha_Noi , trien_khai cach_ly tu ngay 5/8 .",
]
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource
def load_model():
    """Load model lan dau, cache lai cho cac lan sau."""
    checkpoints = sorted(
        BASE.glob("output/baseline_full_seed*/checkpoint-*"),
        key=lambda p: p.stat().st_mtime,
    )
    # Fallback: thu tim bat ky checkpoint nao
    if not checkpoints:
        checkpoints = sorted(
            BASE.glob("output/*/checkpoint-*"),
            key=lambda p: p.stat().st_mtime,
        )
    if not checkpoints:
        return None, None, "Chua co model. Chay: python train_baseline.py --mode full"

    ckpt = checkpoints[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
        model = AutoModelForTokenClassification.from_pretrained(str(ckpt)).to(device)
        model.eval()
        return tokenizer, model, str(ckpt)
    except Exception as e:
        return None, None, str(e)


def predict(text, tokenizer, model):
    """Nhan dien thuc the trong cau da tach tu."""
    words  = text.strip().split()
    device = next(model.parameters()).device

    input_ids  = [tokenizer.cls_token_id]
    word_spans = []

    for word in words:
        subs = tokenizer.encode(word, add_special_tokens=False)
        if not subs:
            continue
        start = len(input_ids)
        if start + len(subs) >= MAX_LEN - 1:
            break
        input_ids.extend(subs)
        word_spans.append((start, len(input_ids)))

    input_ids.append(tokenizer.sep_token_id)
    input_ids = input_ids[:MAX_LEN]

    with torch.no_grad():
        out = model(
            input_ids=torch.tensor([input_ids]).to(device),
            attention_mask=torch.ones(1, len(input_ids), dtype=torch.long).to(device),
        )

    pred_ids = np.argmax(out.logits[0].cpu().numpy(), axis=-1)

    result = []
    for i, (word, (start, _)) in enumerate(zip(words, word_spans)):
        tag = ID2LABEL[pred_ids[start]] if start < len(pred_ids) else "O"
        result.append((word, tag))

    return result


def extract_entities(word_tags):
    """Gom token BIO thanh entities."""
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


def render_highlighted_text(word_tags):
    """Tra ve HTML chuoi van ban voi entity duoc to mau."""
    html_parts = []
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
            color   = ENTITY_COLORS.get(etype, "#E8DAEF")
            label_v = ENTITY_VI.get(etype, etype)
            text    = " ".join(span)
            html_parts.append(
                f'<mark style="background:{color}; padding:2px 6px; border-radius:4px; '
                f'margin:1px; font-weight:500;" title="{label_v}">'
                f'{text} <sup style="font-size:0.65em; color:#555;">{label_v}</sup></mark>'
            )
            i = j
        else:
            html_parts.append(f'<span>{word}</span>')
            i += 1

    return '<div style="line-height:2.2; font-size:1.1rem; padding:12px; ' \
           'background:#fafafa; border-radius:8px; border:1px solid #eee;">' \
           + " ".join(html_parts) + "</div>"


# ── UI ────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="NER COVID-19 Tieng Viet",
        page_icon="🦠",
        layout="wide",
    )

    # Header
    st.title("Nhan dang Thuc the COVID-19 Tieng Viet")
    st.markdown(
        "**Few-shot Named Entity Recognition** voi PhoBERT-base tren bo du lieu PhoNER_COVID19 "
        "*(Truong et al., NAACL 2021)*"
    )
    st.divider()

    # Load model
    with st.spinner("Dang tai model..."):
        tokenizer, model, ckpt_info = load_model()

    if model is None:
        st.error(f"Khong tai duoc model: {ckpt_info}")
        st.code("python train_baseline.py --mode full --seed 42")
        return

    st.success(f"Model: `{ckpt_info}`")

    # Layout 2 cot
    col_input, col_legend = st.columns([3, 1])

    with col_legend:
        st.subheader("Chu thich mau")
        for etype, color in ENTITY_COLORS.items():
            label_v = ENTITY_VI.get(etype, etype)
            st.markdown(
                f'<span style="background:{color}; padding:3px 10px; '
                f'border-radius:4px; font-size:0.85rem;">{label_v}</span>',
                unsafe_allow_html=True,
            )

    with col_input:
        st.subheader("Nhap van ban")
        st.caption("Luu y: van ban can duoc tach tu (dung gach_duoi cho tu ghep, vi du: phi_cong, Benh_vien)")

        # Vi du nhanh
        selected = st.selectbox(
            "Chon cau vi du:",
            ["(Nhap cau cua ban)"] + EXAMPLES,
        )
        default_text = "" if selected == "(Nhap cau cua ban)" else selected
        text_input = st.text_area(
            "Van ban:",
            value=default_text,
            height=100,
            placeholder="Nhap cau tieng Viet da tach tu...",
        )

        run_btn = st.button("Nhan dang thuc the", type="primary", use_container_width=True)

    # Ket qua
    if run_btn and text_input.strip():
        with st.spinner("Dang xu ly..."):
            word_tags = predict(text_input, tokenizer, model)

        st.divider()
        st.subheader("Ket qua nhan dang")

        # Highlighted text
        html = render_highlighted_text(word_tags)
        st.markdown(html, unsafe_allow_html=True)

        # Entity table
        entities = extract_entities(word_tags)
        if entities:
            st.subheader("Cac thuc the tim thay")
            cols = st.columns(2)
            for i, (text, etype) in enumerate(entities):
                color   = ENTITY_COLORS.get(etype, "#eee")
                label_v = ENTITY_VI.get(etype, etype)
                cols[i % 2].markdown(
                    f'<div style="background:{color}; padding:8px 12px; border-radius:6px; '
                    f'margin:4px 0;"><b>{text}</b><br>'
                    f'<small style="color:#555;">{label_v} ({etype})</small></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Khong tim thay thuc the nao trong cau nay.")

    elif run_btn:
        st.warning("Vui long nhap van ban truoc khi chay.")

    # Stats sidebar
    st.divider()
    with st.expander("Thong tin mo hinh va du lieu"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", "PhoBERT-base")
        c2.metric("Micro F1 (Full)", "94.23%")
        c3.metric("Micro F1 (20-shot)", "88.17%")
        c4.metric("Entity types", "10")

        st.markdown("""
        **Bo du lieu**: PhoNER_COVID19 — 10,027 cau, 35,000+ entities, 10 loai thuc the
        **Tham khao**: Truong et al. *COVID-19 Named Entity Recognition for Vietnamese*, NAACL 2021
        **Phuong phap**: PhoBERT-base fine-tuned voi few-shot splits (K=1,5,10,20)
        """)


if __name__ == "__main__":
    main()
