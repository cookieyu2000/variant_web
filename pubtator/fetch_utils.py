# pubtator_inference/fetch_utils.py

import requests

def fetch_pmid_data(variant):
    """
    以 variant (如 c.3578G>A) 去 PubTator3 API 搜尋相關的 PMID
    """
    base_url = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/search/"
    page = 1
    pmid_list = []
    while True:
        url = f"{base_url}?text=@{variant}&page={page}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"無法連接到API，狀態碼：{response.status_code}")
            break
        data = response.json()
        if 'results' in data:
            for item in data['results']:
                if '_id' in item:
                    pmid_list.append(item['_id'])
        total_pages = data.get('total_pages', 1)
        if page >= total_pages:
            break
        page += 1
    return pmid_list

def fetch_full_text_via_api(pmid):
    """
    以 PubTator3 API 拿到指定 PMID 的 BioC XML (full=true)
    """
    base_url = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocxml"
    params = {
        "pmids": pmid,
        "full": "true"
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print(f"無法抓取全文資料，PMID: {pmid}，狀態碼：{response.status_code}")
        return None
    return response.text
