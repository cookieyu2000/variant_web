# pubtator/predict.py
import os, yaml, pickle, json, warnings, inspect
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from .config import CLASSIFIER_CONFIG_YAML
from .ner_entity import ner_pipe
import importlib  # 新增

# 載入本 package 底下的 model.py
model_module = importlib.import_module(f"{__package__}.model")

# 自動偵測所有在 model.py 中定義、且名稱以 "Classifier" 結尾的類別
MODEL_CLASSES = {
    name: cls
    for name, cls in inspect.getmembers(model_module, inspect.isclass)
    if name.lower().endswith("classifier")
}

class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length, max_paragraphs, stride=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_paragraphs = max_paragraphs
        self.stride = stride
        self.samples = []
        for paras in texts:
            proc = []
            tokens_window = max_length - 2
            for p in paras:
                toks = tokenizer.tokenize(p)
                if len(toks) > tokens_window:
                    for i in range(0, len(toks), stride):
                        sub = toks[i:i+tokens_window]
                        if sub:
                            proc.append(tokenizer.convert_tokens_to_string(sub))
                        if len(proc)>=max_paragraphs: break
                else:
                    proc.append(p)
                if len(proc)>=max_paragraphs: break
            if len(proc)<max_paragraphs:
                proc += [""]*(max_paragraphs-len(proc))
            input_ids, attn = [], []
            for txt in proc:
                enc = tokenizer(
                    txt, add_special_tokens=True,
                    max_length=max_length, padding="max_length",
                    truncation=True, return_attention_mask=True,
                    return_tensors="pt"
                )
                input_ids.append(enc["input_ids"].squeeze(0))
                attn.append(enc["attention_mask"].squeeze(0))
            self.samples.append((torch.stack(input_ids), torch.stack(attn)))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        ids, mask = self.samples[idx]
        return {"input_ids":ids, "attention_mask":mask}


def setup_inference(config_path=None):
    cfg_file = config_path or CLASSIFIER_CONFIG_YAML
    with open(cfg_file) as f:
        config = yaml.safe_load(f)
    print("Starting inference setup...")
    split = config['paths']['split_data_dir']
    with open(os.path.join(split,'id2label.pkl'),'rb') as f: id2label = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = config['model']['type']
    ModelCls = MODEL_CLASSES[model_type]
    # 過濾 __init__ 可接受的參數
    common = {k:v for k,v in config['model'].items() if k not in ['type',model_type]}
    spec   = config['model'].get(model_type,{})
    transf = config.get('transformer',{})
    allp   = {**common, **spec, 'transformer_config':transf}
    sig    = inspect.signature(ModelCls.__init__)
    valid  = set(sig.parameters)-{'self','args','kwargs'}
    params = {k:v for k,v in allp.items() if k in valid}
    model  = ModelCls(**params)
    model.load_state_dict(torch.load(config['paths']['best_model_path'],map_location=device))
    model.to(device).eval()
    print("Classification model loaded.")
    return config, model, id2label, None, device


def predict_classification(texts, config, model, id2label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(config['data']['tokenizer_name'])
    ds = InferenceDataset(texts, tokenizer,
                          max_length=config['data']['max_length'],
                          max_paragraphs=config['data']['max_paragraphs'],
                          stride=config['data'].get('stride',128))
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            ids  = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            out  = model(input_ids=ids, attention_mask=attn)
            idxs = torch.argmax(out, dim=1).cpu().tolist()
            for i in idxs:
                preds.append(id2label.get(i,"Unknown"))
    return preds


def ner_inference(paragraphs):
    """
    Return True if any paragraph contains a 'variant' entity.
    """
    for para in paragraphs:
        for ent in ner_pipe(para):
            if ent.get("entity_group","").lower()=="variant":
                return True
    return False