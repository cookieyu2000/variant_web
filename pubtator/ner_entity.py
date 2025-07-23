# pubtator/ner_entity.py
import os
import html
from collections import Counter
from flask import Blueprint, render_template, request
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from .config import NER_MODEL_DIR, WINDOW_SIZE, STRIDE

# Blueprint
ner_bp = Blueprint("ner_entity", __name__, template_folder="templates")

# 只有真正调用时才加载模型
_ner_pipe = None
_COLOR_MAP = {
    "gene": "lightblue", "disease": "lightgreen", "chemical": "lightpink",
    "variant": "red", "species": "khaki", "cellline": "lightcoral",
    "chromosome": "lightseagreen", "refseq": "plum", "genomicregion": "peachpuff",
}

def get_ner_pipe():
    global _ner_pipe
    if _ner_pipe is None:
        tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_DIR)
        model     = AutoModelForTokenClassification.from_pretrained(NER_MODEL_DIR)
        _ner_pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=0  # 如果没有 GPU，改成 device=-1
        )
    return _ner_pipe

def ner_highlight_html(text: str):
    raw = text or ""
    if not raw.strip():
        return "<div><i>No content provided.</i></div>", "<div><i>No entities.</i></div>"

    spans = []
    length = len(raw)
    for start in range(0, length, STRIDE):
        seg = raw[start:start+WINDOW_SIZE]
        for ent in get_ner_pipe()(seg):
            s, e = ent["start"], ent["end"]
            spans.append((start + s, start + e, ent["entity_group"]))
        if start + WINDOW_SIZE >= length:
            break

    unique_spans = sorted(set(spans), key=lambda x: x[0])

    # 构建高亮 HTML
    last, parts = 0, []
    for s, e, lbl in unique_spans:
        if s < last:
            continue
        parts.append(html.escape(raw[last:s]))
        color = _COLOR_MAP.get(lbl, "gray")
        parts.append(
            f"<span style='color:{color}; font-weight:bold'>{html.escape(raw[s:e])}</span>"
        )
        last = e
    parts.append(html.escape(raw[last:]))
    highlighted = (
        "<div class='output-box'><h4>Annotated Output</h4>"
        + "".join(parts) +
        "</div>"
    )

    # 构建 Annotation Summary
    by_lbl = {}
    for s, e, lbl in unique_spans:
        by_lbl.setdefault(lbl, []).append(raw[s:e])
    summary_parts = ["<div class='output-box'><h4>Annotation Summary</h4>"]
    if not unique_spans:
        summary_parts.append("<p><i>No entities found.</i></p>")
    else:
        for lbl, texts in by_lbl.items():
            color = _COLOR_MAP.get(lbl, "gray")
            badges = ""
            for txt, cnt in Counter(texts).items():
                badges += (
                    f"<span class='entity-badge' style='color:{color}; border-color:{color}'>"
                    f"{html.escape(txt)} ({cnt})</span>"
                )
            summary_parts.append(
                f"<p><strong style='color:{color}; text-transform:capitalize'>{lbl}:</strong> {badges}</p>"
            )
    summary_parts.append("</div>")
    summary = "".join(summary_parts)

    return highlighted, summary

@ner_bp.route("/ner_entity", methods=["GET", "POST"])
def ner_page():
    input_text = ""
    highlighted_html, summary_html = "", ""
    if request.method == "POST":
        input_text = request.form.get("ner_text", "").strip()
        highlighted_html, summary_html = ner_highlight_html(input_text)
    return render_template(
        "ner_entity.html",
        input_text=input_text,
        highlighted_html=highlighted_html,
        summary_html=summary_html
    )

# 为旧版 predict.py 导入兼容：延迟获取 pipe
def ner_pipe(text: str):
    """
    兼容 predict.py 中的调用接口，
    返回 list of entity dicts。
    """
    return get_ner_pipe()(text)
