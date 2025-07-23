import re
import torch
from typing import List
from lime.lime_text import LimeTextExplainer

SENT_TOKEN = "<<<SENT_BREAK>>>"

def custom_sent_tokenize(text: str) -> List[str]:
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)

def lime_sentence_predict(texts, model, tokenizer, device):
    model.eval()
    encoding = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=512,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding["input_ids"].unsqueeze(1).to(device)
    attention_mask = encoding["attention_mask"].unsqueeze(1).to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

def highlight_paragraph(
    paragraph_text: str,
    explainer: LimeTextExplainer,
    model,
    tokenizer,
    class_names: List[str],
    device,
    base_threshold: float = 0.1,
    num_samples: int = 300
) -> str:
    """
    對單一段落做 LIME 解釋，並回傳 HTML 字串。
    num_samples 只傳給 explain_instance()。
    """
    if not paragraph_text.strip():
        return paragraph_text

    sentences = custom_sent_tokenize(paragraph_text)
    if not sentences:
        return paragraph_text

    # join sentences with a unique token
    joined = f" {SENT_TOKEN} ".join(sentences)
    explanation = explainer.explain_instance(
        joined,
        classifier_fn=lambda x: lime_sentence_predict(x, model, tokenizer, device),
        num_samples=num_samples,   # 正確放在這裡
        top_labels=2
    )

    # 取出最重要的標籤
    top_label = explanation.top_labels[0]
    exp_list = explanation.as_list(label=top_label)
    weights = {feat.strip(): w for feat, w in exp_list}

    low, med, high = base_threshold, base_threshold*2, base_threshold*4

    highlighted = []
    for sent in sentences:
        w = weights.get(sent.strip(), 0.0)
        if w >= high:
            color = "darkred"
        elif w >= med:
            color = "red"
        elif w >= low:
            color = "lightcoral"
        elif w <= -high:
            color = "darkgreen"
        elif w <= -med:
            color = "green"
        elif w <= -low:
            color = "lightgreen"
        else:
            color = None

        if color:
            highlighted.append(
                f"<span style='color:{color}; font-weight:bold;'>{sent.strip()}</span>"
            )
        else:
            highlighted.append(sent.strip())

    return " ".join(highlighted)

def highlight_lime_in_paragraphs(
    paragraphs: List[str],
    model,
    tokenizer,
    class_names: List[str],
    device,
    base_threshold: float = 0.1,
    num_samples: int = 300
) -> str:
    """
    對多個段落做 LIME 解釋，回傳整段 HTML。
    num_samples 從前端傳進來，由 highlight_paragraph 使用。
    """
    explainer = LimeTextExplainer(
        split_expression=re.escape(SENT_TOKEN),
        bow=False,
        class_names=class_names
    )

    html_paras = []
    for para in paragraphs:
        html = highlight_paragraph(
            paragraph_text=para,
            explainer=explainer,
            model=model,
            tokenizer=tokenizer,
            class_names=class_names,
            device=device,
            base_threshold=base_threshold,
            num_samples=num_samples
        )
        html_paras.append(f"<p>{html}</p>")

    return "\n".join(html_paras)
