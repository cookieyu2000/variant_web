# pubtator_inference/file_utils.py

import os
import json

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_all_pmids(json_path):
    """
    讀取保存的 { variant: [pmid_list...] } JSON 檔，
    若不存在則回傳空 dict
    """
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_all_pmids(all_pmids, json_path):
    """
    儲存 { variant: [pmid_list...] } 到 JSON 檔
    """
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_pmids, f, indent=4, ensure_ascii=False)

def save_variant_data(variant_data, output_file):
    """
    將 {pmid: {...段落...}, pmid2: {...}} 寫成 JSON 檔
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(variant_data, f, indent=4, ensure_ascii=False)
