import requests
from requests.adapters import HTTPAdapter, Retry
from lxml import etree
from bs4 import BeautifulSoup
import urllib3

# 禁用 InsecureRequestWarning 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 定义年数范围
years = list(range(2024, 2014, -1))

# 会话和重试设置
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504, 522, 524])

# 安装适配器
adapter = HTTPAdapter(max_retries=retries)
session.mount('http://', adapter)

def extract_text_from_url(url):
    try:
        response = session.get(url, verify=False)  # 跳过SSL证书验证
        response.raise_for_status()

        # 从响应中获取HTML内容
        html_content = response.content

        # 使用BeautifulSoup解析HTML文档
        soup = BeautifulSoup(html_content, 'html.parser')

        # 使用lxml解析HTML文档
        tree = etree.HTML(html_content)

        # 提取XPath对应的文本
        text_elements = tree.xpath('//*[@id="top_bg"]/div/div[4]/div[2]/div[3]//text()')
        text = ''.join(text_elements).strip()

        return text

    except requests.RequestException as e:
        print(f"请求失败: {e}")
        return None

def save_text_to_file(year, text):
    filename = f"{year}.txt"
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"内容已成功保存到 {filename}")
    except IOError as e:
        print(f"保存文件失败: {e}")

def main():
    links_file = 'urls.txt'
    try:
        with open(links_file, 'r', encoding='utf-8') as file:
            urls = file.readlines()

        for year, url in zip(years, urls):
            url = url.strip()
            if url:
                print(f"正在处理 {year} 对应的URL: {url}")
                text = extract_text_from_url(url)
                if text:
                    save_text_to_file(year, text)
                else:
                    print(f"无法从URL中提取内容: {url}")

    except IOError as e:
        print(f"读取文件失败: {e}")

if __name__ == "__main__":
    main()
