import re
import time
import json
import random
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from fake_useragent import UserAgent


def idcard_regions_map(output=True, result=False, bar=True, fake_ua=True, sleep=False):
    """
    获取身份证前6位对应的行政区划名称
    :param  output: 是否保存字典到文件（当前路径下的 idcard_regions.json 文件）
    :param  result: 函数执行完成后是否返回结果
    :param     bar: 是否开启进度条查看实时进度
    :param fake_ua: 是否使用虚拟请求头
    :param   sleep: 是否在每次请求完成后 sleep 一段时间再进行请求
    """
    url = "http://www.zxinc.org/gb2260.htm"
    query_url = "https://id.8684.cn/ajax.php?act=check"
    
    if fake_ua:
        ua = UserAgent(path="fake_ua.json")
    else:
        ua = None

    response = requests.get(url=url)
    response.encoding = response.apparent_encoding
    soup = BeautifulSoup(response.text)
    regions = soup("areacode")[0].text.strip().split("\r\n")
    regions = [re.sub(r"[\u3000|\[\]]{1,}", "", r) for r in regions]
    regions_list = [re.findall("[\d]+", r)[0] for r in regions]

    if bar:
        regions_iter = tqdm(regions_list, total=len(regions_list))
    else:
        regions_iter = regions_list
    
    regions_map = {}
    for r in regions_iter:
        kwargs = {"url": query_url, "data": {"userId": r}}
        if fake_ua:
            kwargs.update({"headers": {"User-Agent": ua.random}})
        
        place = requests.post(**kwargs).json().get("place")
        if place is not None and place.strip() != "":
            regions_map[r] = place

        if sleep:
            time.sleep(random.random())

    if output:
        with open("idcard_regions.json", "w", encoding="utf8") as f:
            json.dump(regions_map, f, indent=4, ensure_ascii=False)
    
    if result:
        return regions_map
    
    
if __name__ == '__main__':
    idcard_regions_map()
