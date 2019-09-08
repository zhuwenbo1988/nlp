#!/bin/env python26
# coding=UTF-8


import scrapy
import re
import time
import random
from scrapy.http import FormRequest
from scrapy.crawler import CrawlerProcess
import crawler_util
from bs4 import BeautifulSoup as soup
import logging
import os
import url_manager


logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(filename)s %(funcName)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='crawler.log',
                    filemode='a')


HEADER={
    'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'
    }
# HEADER = crawler_util.dianping_headers


# 要爬取大众点评的哪个频道
cat_index = None


output_dir = None


class DianpingSpider(scrapy.Spider):
    name = 'dianping'
    start_urls = []

    def __init__(self):
        self.headers = HEADER
        for i in range(1, 351):
            self.start_urls.append(
                'http://www.dianping.com/search/category/{}/{}'.format(i, cat_index))
        # 获取已经爬取的url
        if cat_index == crawler_util.get_main_categroy_number('周边游'):
            um = url_manager.UrlManager(url_manager.UrlFile.Dianping_travel)
        else :
            um = url_manager.UrlManager(url_manager.UrlFile.Dianping_rest)
        um.load_urls()
        self.url_manager = um

    def start_requests(self):
        for url in self.start_urls:
            yield FormRequest(url, headers=self.headers)

    def parse(self, response):
        self._sleep()
        if not response.status == 200:
            return
        html_text = response.body
        if not html_text:
            return
        HTML = soup(html_text)
        city_name = crawler_util.parse_city_name(HTML)
        cook_dict = self._parse_city_cook(HTML)
        for cook_name in cook_dict:
            # 菜品
            cook_index = cook_dict[cook_name]
            cook_url = '{}/{}'.format(response.url, cook_index)
            yield scrapy.Request(cook_url, headers=self.headers, callback=self.parse_cook)

    def parse_cook(self, response):
        self._sleep()
        if not response.status == 200:
            return
        html_text = response.body
        if not html_text:
            return
        HTML = soup(html_text)
        #
        area_dict = self._parse_city_cook_area(HTML)
        for area_name in area_dict:
            # 区域
            area_index = area_dict[area_name]
            area_url = '{}{}'.format(response.url, area_index)
            yield scrapy.Request(area_url, headers=self.headers, callback=self.parse_page)

    def parse_page(self, response):
        self._sleep()
        if not response.status == 200:
            return
        html_text = response.body
        if not html_text:
            return
        HTML = soup(html_text)
        #
        pages = crawler_util.parse_max_page(HTML)
        for page_num in range(1, pages + 1):
            page_url = '{}p{}'.format(response.url, page_num)
            yield scrapy.Request(page_url, headers=self.headers, callback=self.parse_shop_index)

    def parse_shop_index(self, response):
        self._sleep()
        if not response.status == 200:
            return
        html_text = response.body
        if not html_text:
            return
        HTML = soup(html_text)
        shop_list = self._parse_shop_list(HTML)
        for shop_id in shop_list:
            shop_comment_url = 'http://www.dianping.com/shop/{}/review_all_flower'.format(
                shop_id)
            if not shop_comment_url in self.url_manager.url_container:
                yield scrapy.Request(shop_comment_url, headers=self.headers, callback=self.save_shop_comment)
	    else : 
		logging.warning('has:'+shop_comment_url)
            self._sleep()
            shop_base_url = 'http://www.dianping.com/shop/{}'.format(shop_id)
            if not shop_base_url in self.url_manager.url_container:
                yield scrapy.Request(shop_base_url, headers=self.headers, callback=self.save_shop_base_info)
	    else :
		logging.warning('has:'+shop_base_url)

    def save_shop_comment(self, response):
        self._sleep()
        if not response.status == 200:
            return
        # 输入验证码的重定向
        if response.url.startswith('http://h5.dianping.com/platform/secure/index.html'):
	    logging.warning('yan zheng ma '+response.url)
            return
        html_text = response.body
        if not html_text:
            return
        filename_cmt = 'dianping_comment_{}'.format(
            re.search('\d+', response.url).group())
        try:
            outfile = os.path.join(output_dir, filename_cmt)
            f = open(outfile, 'w')
            f.write(html_text)
            f.close()
        except Exception as e:
            f.close()
        self.url_manager.add_url(response.url)

    def save_shop_base_info(self, response):
        self._sleep()
        if not response.status == 200:
            return
        # 输入验证码的重定向
        if response.url.startswith('http://h5.dianping.com/platform/secure/index.html'):
	    logging.warning('yan zheng ma '+response.url)
            return
        html_text = response.body
        if not html_text:
            return
        filename_base = 'dianping_base_{}'.format(
            re.search('\d+', response.url).group())
        try:
            outfile = os.path.join(output_dir, filename_base)
            f = open(outfile, 'w')
            f.write(html_text)
            f.close()
        except Exception as e:
            f.close()
        self.url_manager.add_url(response.url)

    def _parse_city_cook(self, HTML):
        cook_dict = {}
        # 城市的菜品类别
        for div in HTML.find_all('div', 'nc-items', id='classfy'):
            for a in div.find_all('a'):
                m = re.search('g\d+', a['href'])
                if m:
                    cook_dict[a.string] = m.group()
        return cook_dict

    def _parse_city_cook_area(self, HTML):
        # 提取当前菜品存在的区域
        area_dict = {}
        for div in HTML.find_all('div', id='bussi-nav'):
            for a in div.find_all('a'):
                m = re.search('r\d+', a['href'])
                if m:
                    area_dict[a.string] = m.group()
        return area_dict

    def _parse_city_cook_sub_cook(self, HTML):
        # 提取当前菜品的子类别
        sub_cook_dict = {}
        for div in HTML.find_all('div', id='classfy-sub'):
            for a in div.find_all('a'):
                m = re.search('g\d+', a['href'])
                if m:
                    sub_cook_dict[a.string] = m.group()
        return sub_cook_dict

    def _parse_shop_list(self, HTML):
        shop_list = []
        for div in HTML.find_all('div', ['pic']):
            m = re.search('\d+', div.a['href'])
            if m:
                shop_list.append(m.group())
        return shop_list

    def _sleep(self):
        time.sleep(random.uniform(2, 3))


def process(out, main_ctg=crawler_util.get_main_categroy_number('美食')):
    global output_dir, cat_index
    # 输出路径
    output_dir = out
    # 默认是大众点评的美食频道
    cat_index = main_ctg
    # 启动爬虫
    process = CrawlerProcess()
    process.crawl(DianpingSpider)
    process.start()


if __name__ == '__main__':
    process(out='out')
