# coding=utf-8

from bs4 import BeautifulSoup
import json

#xml = BeautifulSoup(open('Laptops_Train.xml'))
xml = BeautifulSoup(open('Restaurants_Train.xml'))

for s in xml.find_all('sentence'):
  id = s['id']
  text = s.text.strip()
  aspects = s.aspectterms
  if not aspects:
    d = {}
    d['text'] = text
    print(json.dumps(d, ensure_ascii=False)).encode('utf-8')
    continue
  a = []
  for aspect in aspects.find_all('aspectterm'):
    a.append((aspect['term'], int(aspect['from']), int(aspect['to'])))
  d = {}
  d['text'] = text
  d['aspect'] = a
  print(json.dumps(d, ensure_ascii=False)).encode('utf-8')
