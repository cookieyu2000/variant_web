# pubtator_inference/pub_inference.py

from .fetch_utils import fetch_pmid_data, fetch_full_text_via_api
from .parser_utils import parse_biocxml, sanitize_filename
from .file_utils import ensure_dir_exists, load_all_pmids, save_all_pmids, save_variant_data
import os

def do_inference_for_variant(variant, base_output_dir, pmid_list_file):
    """
    給定一個 variant (e.g. 'c.3578G>A')，會：
      1) 用 PubTator 搜尋 pmid_list
      2) 讀取/更新 pmid_list_file (保存所有 variant->pmid_list 的紀錄)
      3) 逐篇 pmid 解析 BioC XML
      4) 存成一個 {pmid: {...}} 的 JSON 到 base_output_dir/<variant>.json
      5) 若同一次執行出現重複 PMIDs，不會重複下載

    回傳: (variant_data, output_file_path)
    """

    # 讀取/初始化 all_pmids
    all_pmids = load_all_pmids(pmid_list_file)

    # 取得指定 variant 的 pmid_list
    pmid_list = fetch_pmid_data(variant)
    if not pmid_list:
        print(f"variant {variant} 未找到任何PMID資料。")
        return {}, None

    # 更新全域的 all_pmids
    all_pmids[variant] = pmid_list
    save_all_pmids(all_pmids, pmid_list_file)
    print(f"variant {variant} 的PMID數據已保存到 {pmid_list_file}")

    # 開始抓取並解析
    variant_data = {}
    already_fetched_pmids = set()  # 本次執行中已經下載過的PMID

    for pmid in pmid_list:
        if pmid in already_fetched_pmids:
            print(f"PMID {pmid} 已經下載過，跳過重複下載。")
            continue

        print(f"正在查詢PMID: {pmid} 的全文資料...")
        xml_data = fetch_full_text_via_api(pmid)
        if xml_data:
            parsed_obj = parse_biocxml(xml_data, pmid)
            if parsed_obj:
                variant_data.update(parsed_obj)

        already_fetched_pmids.add(pmid)

    # 寫檔
    if variant_data:
        safe_variant_name = sanitize_filename(variant)
        ensure_dir_exists(base_output_dir)
        output_file = os.path.join(base_output_dir, f"{safe_variant_name}.json")
        save_variant_data(variant_data, output_file)
        print(f"variant {variant} 的全文數據已保存到 {output_file}")
        return variant_data, output_file
    else:
        print(f"variant {variant} 沒有任何文章符合指定段落。")
        return {}, None


