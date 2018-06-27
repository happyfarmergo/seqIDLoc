# -*- coding: utf-8 -*-
import datetime
import json
import logging
import logging.config
import math

import time

import requests


class CellTower:
    def __init__(self):
        self.base_url = 'http://api.gpsspg.com/bs/?oid=7763&key=0D22FE4AB2AF8B9CAD286C016924DD1E&type=gsm&'
        self.headers = {
            "User-Agent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.10; rv:41.0) Gecko/20100101 Firefox/41.0',
            "Referer": 'http://api.gpsspg.com/',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Host': 'api.gpsspg.com',
        }

        self.session = requests.session()

    def get_tower(self, towers):
        result = []
        failed = []
        cnt = 0 
        for rncid, cellid in towers:
            if cellid == -1 or math.isnan(cellid) or cellid==-999:
                print 'id error'
                continue
            url = self.base_url + 'bs=460,01,%d,%d&output=json' % (rncid, cellid)
            req = self.session.get(url=url, headers=self.headers)
            data = json.loads(req.text, encoding='utf-8')
            if data['status']!=200:
                print 'error:(%d,%d)' % (rncid, cellid)
                failed.append((rncid, cellid))
                continue
            lat, lng = float(data['result'][0]['lats']), float(data['result'][0]['lngs'])
            result.append((rncid, cellid, lat, lng, -1, -1))
            print cnt
            time.sleep(3)
            cnt += 1
        return result, failed
