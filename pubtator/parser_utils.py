# pubtator_inference/parser_utils.py

import os
import requests
import json
import xml.etree.ElementTree as ET
from .fetch_utils import fetch_pmid_data, fetch_full_text_via_api


def parse_biocxml(xml_data, pmid):
    """
    從 PubTator 回傳的 BioC XML 中，抓取各個 section (Title、Abstract、Introduction、Methods、Results、Conclusion、Discussion...)，
    但若 passage 的 type 裡包含 'title' (例如 title_1, title_2) 就跳過不抓(避免抓到小標題)。 
    若 section_type 為 'TITLE'，則視為真正的論文標題。

    最後將同一 section 多段合併為多行字串，保留段落空行與首行縮排。
    """

    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        print(f"解析 PMID {pmid} 的 XML 錯誤: {e}")
        return {}

    # 新增 "title" -> "Title"，以便能抓到 <section_type="TITLE">...
    section_map = {
        "title":        "Title",
        "abstract":     "Abstract",
        "intro":        "Introduction",
        "method":       "Methods",
        "result":       "Results",
        "concl":        "Conclusion",
        "discuss":      "Discussion"
    }

    # 如果 text 僅僅是這些字樣（大小寫不分），就跳過
    skip_standalone = {"title", "abstract", "introduction", "results", "conclusion", "discussion", "methods"}

    # 只要 passage_type 含有 'title' 就跳過 —— 用來排除小標題 (type="title_1"等)
    skip_if_type_contains_title = True

    # 用來暫存各段落
    result_sections = {
        "Title": [],
        "Abstract": [],
        "Introduction": [],
        "Methods": [],
        "Results": [],
        "Conclusion": [],
        "Discussion": []
    }

    has_content = False

    for document in root.findall('document'):
        # 先嘗試從 <infon key="article-id_pmid"> 拿 PMID
        doc_infons = {
            infon.get('key'): (infon.text.strip() if infon.text else '')
            for infon in document.findall('.//infon')
        }
        doc_pmid = doc_infons.get('article-id_pmid', '')

        # 若拿不到，再退而求其次從 <document><id>
        if not doc_pmid:
            doc_id_elem = document.find('id')
            doc_pmid = doc_id_elem.text.strip() if (doc_id_elem is not None and doc_id_elem.text) else ''

        # 只處理符合當前 pmid 的 <document>
        if doc_pmid != pmid:
            continue

        # 走訪所有 <passage>
        for passage in document.findall('passage'):
            # 取得 section_type (小寫)
            section_type_elem = passage.find('infon[@key="section_type"]')
            section_type = (section_type_elem.text.strip().lower()
                            if (section_type_elem is not None and section_type_elem.text)
                            else '')

            # 檢查 <infon key="type"> 以排除 "title_1", "title_2" 等
            type_elem = passage.find('infon[@key="type"]')
            passage_type = (type_elem.text.strip().lower()
                            if (type_elem is not None and type_elem.text)
                            else '')

            # 如果 passage_type 含 'title' 就跳過 (避免小標題)
            if skip_if_type_contains_title and ("title" in passage_type):
                continue

            # 抓取 text
            text_element = passage.find('text')
            text = text_element.text.strip() if (text_element is not None and text_element.text) else ''
            if not text:
                continue

            # map 到對應 section
            section_name = None
            for key_substr, mapped_section in section_map.items():
                if key_substr in section_type:
                    section_name = mapped_section
                    break
            if not section_name:
                # 表示這個 passage 不在我們關心的段落類型裡，跳過
                continue

            # 若 text 本身就是 "title"/"abstract"/"introduction" 等純字樣 => 跳過
            if text.lower() in skip_standalone:
                continue

            # 收進對應 section
            result_sections[section_name].append(text)
            has_content = True

    # 若全程都沒抓到東西
    if not has_content:
        print(f"PMID {pmid} - No articles available。")
        return {}

    # 多段落合併 => 多行字串
    for sec, paragraphs in result_sections.items():
        if paragraphs:
            # 每段前加 4 格空白，段落間以一行空白分隔
            with_indent = ["    " + p for p in paragraphs]
            multiline_text = "\n\n".join(with_indent)
            result_sections[sec] = multiline_text
        else:
            result_sections[sec] = ""

    # 最後加上 PubMed_Link
    result_sections["PubMed_Link"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    
    print(f"PMID {pmid} - got full article")

    return {pmid: result_sections}


def sanitize_filename(name):
    invalid_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    for c in invalid_chars:
        name = name.replace(c, '_')
    return name

if __name__ == "__main__":
    base_output_dir = "PubTator3_data/full_text"
    pmid_list_file = "PubTator3_data/pmid_list.json"
    os.makedirs(base_output_dir, exist_ok=True)

    # 讀取或初始化 all_pmids (variant->pmid_list)
    if os.path.exists(pmid_list_file):
        with open(pmid_list_file, "r", encoding="utf-8") as f:
            all_pmids = json.load(f)
    else:
        all_pmids = {}

    while True:
        variant = input("請輸入查詢的 variant (例如 c.3578G>A, 輸入 'exit' 結束): ").strip()
        if variant.lower() == "exit":
            print("結束程式。")
            break
        if not variant:
            print("未輸入variant, 請重新輸入。")
            continue

        sanitized_variant = sanitize_filename(variant)
        pmid_list = fetch_pmid_data(variant)
        if not pmid_list:
            print(f"variant {variant} 未找到任何PMID資料。")
            continue

        # 更新 PMIDs 至全域檔案
        all_pmids[variant] = pmid_list
        with open(pmid_list_file, "w", encoding="utf-8") as f:
            json.dump(all_pmids, f, indent=4, ensure_ascii=False)
        print(f"variant {variant} 的PMID數據已保存到 {pmid_list_file}")

        # 準備存放解析結果
        variant_data = {}
        already_fetched_pmids = set()  # 本次執行中已經下載過的PMID

        for pmid in pmid_list:
            if pmid in already_fetched_pmids:
                print(f"PMID {pmid} 已經下載過，跳過重複下載。")
                continue

            print(f"正在查詢PMID: {pmid} 的全文資料...")
            full_text_xml = fetch_full_text_via_api(pmid)
            if full_text_xml:
                parsed_obj = parse_biocxml(full_text_xml, pmid)
                if parsed_obj:
                    # parsed_obj = { pmid: {...} }
                    variant_data.update(parsed_obj)

            already_fetched_pmids.add(pmid)

        if variant_data:
            output_file = os.path.join(base_output_dir, f"{sanitized_variant}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(variant_data, f, indent=4, ensure_ascii=False)
            print(f"variant {variant} 的全文數據已保存到 {output_file}")
        else:
            print(f"variant {variant} 沒有任何文章符合指定段落。")
