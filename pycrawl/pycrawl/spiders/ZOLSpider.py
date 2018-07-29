'''
Created on 2018年5月20日

@author: Administrator
'''

import scrapy
from pycrawl.items import PycrawlItem

class ZOLSprider(scrapy.Spider):
    name = "hao"
    allowed_domains = ["pic.hao123.com"]
    start_urls = [
        "http://pic.hao123.com/"
    ]
    
    def parse(self, response):
        for sel in response.xpath('//ul/li/a'):
            item = PycrawlItem()
            item['title'] = sel.xpath('img/@title').extract()
            item['width'] = sel.xpath('img/@width').extract()
            item['height'] = sel.xpath('img/@height').extract()
            item['image_urls'] = sel.xpath('img/@src').extract()
            yield item
        
        for sel in response.xpath('//div/div/div/div/div/div'):
            item['image_urls'] = sel.xpath('div/@src').extract()
            yield item
        