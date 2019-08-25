# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import csv
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import requests
import codecs
import cStringIO
import time
from datetime import datetime
import sys
from datetime import datetime


ONE_WEEK_LENGTH = 1000 * 60 * 60 * 24 * 7
ONE_DAY_LENGTH = 1000 * 60 * 60 * 24

domain = '手表'
crawler_date_url = 'http://mobvoi-oss/v1/ufile/mobvoi-search-public/crawler/tmall.json'


class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """
    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf-8") for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


def send_email(web_api, start_time, end_time):
    mail_username = 'wbzhu@mobvoi.com'
    mail_password = 'hzthlvkbhpzndvsl'
    from_addr = mail_username
    to_addrs = ['wbzhu@mobvoi.com', 'linyili@mobvoi.com', 'bzhang@mobvoi.com', 'leyang@mobvoi.com', 'wshzhang@mobvoi.com']
    email_attach_path = 'email_attach_temp.csv'
    csv_lines = 0
    try:
        r = requests.get(web_api['getCSV'].format(domain, start_time, end_time))
        r_getcsv = json.loads(r.content.decode('utf-8'))
        if r_getcsv['code'] == 500:
            print('Web server internal error, api:getCSV')
            exit()
        csv_lines = len(r_getcsv['data'])
        with open(email_attach_path, 'wb') as csvfile:
            csv_writer = UnicodeWriter(csvfile)
            csv_writer.writerows(r_getcsv['data'])
    except Exception as e:
        print('web_api:getCSV error:{}'.format(e))
        exit()

    message = MIMEMultipart()
    message['From'] = Header('mobvoi opinion mining', 'utf-8')
    message['To'] = Header('', 'utf-8')
    subject = '手表评论分析_{}-{}'.format(timestamp2date(start_time), timestamp2date(end_time))
    message['Subject'] = Header(subject, 'utf-8')

    message.attach(MIMEText('这是一封自动发送的邮件。', 'plain', 'utf-8'))
    att1 = MIMEText(open(email_attach_path, 'rb').read(), 'base64', 'utf-8')
    att1['Content-Type'] = 'application/octet-stream'
    att_name = "comments_{}-{}.csv".format(timestamp2date(start_time), timestamp2date(end_time))
    att1['Content-Disposition'] = 'attachment; filename={}'.format(att_name)
    message.attach(att1)

    def send():
        try:
            smtpObj = smtplib.SMTP()
            smtpObj.connect('smtp.gmail.com')
            smtpObj.ehlo()
            smtpObj.starttls()
            smtpObj.login(mail_username, mail_password)
            smtpObj.sendmail(from_addr, to_addrs, message.as_string())
            smtpObj.quit()
            print('邮件发送成功,附件:{}({}lines),收件人:{}\t{}'.format(att_name, csv_lines, to_addrs, datetime.now()))
            return True
        except Exception as e:
            print('邮件发送失败,{}\t{}'.format(e, datetime.now()))
            return False
    flag = send()
    while flag is False:
        print('等待5s尝试重新发送邮件...\t{}'.format(datetime.now()))
        time.sleep(5)
        flag = send()


def timestamp2date(timestr):
    time_array = time.localtime(int(timestr) / 1000)
    time_str = time.strftime("%Y/%m/%d", time_array)
    return time_str


if __name__ == '__main__':
    host = sys.argv[1]
    port = sys.argv[2]
    address = 'http://{}:{}'.format(host, port)
    web_api = {
        'saveRawData': address + '/saveRawData?domain={}&filePath={}',
        'preprocess': address + '/preprocess?domain={}&spu={}&startTime={}',
        'batchInfer': address + '/batchInfer?domain={}&spu={}',
        'getCSV': address + '/getCSV?domain={}&startTime={}&endTime={}'
    }

    latestTime = ''

    try:
        r_save_raw_data = requests.get(web_api['saveRawData'].format(domain, crawler_date_url))
        r_save_raw_data = json.loads(r_save_raw_data.content.decode('utf-8'))
        if r_save_raw_data['code'] == 500:
            print('Web server internal error, api:saveRawData')
            exit()
        latestTime = r_save_raw_data['latestTime']
        print('web_api:saveRawData:{}\t{}'.format(r_save_raw_data,datetime.now()))
    except Exception as e:
        print('web_api:saveRawData error:{}\t{}'.format(e,datetime.now()))

    if latestTime == '':
        exit()

    try:
        r_preprocess = requests.get(web_api['preprocess'].format(domain, 'none', latestTime))
        r_preprocess = json.loads(r_preprocess.content.decode('utf-8'))
        if r_preprocess['code'] == 500:
            print('Web server internal error, api: preprocess\t{}'.format(datetime.now()))
            exit()
        print('web_api:preprocess:{}\t{}'.format(r_preprocess, datetime.now()))
    except Exception as e:
        print('web_api:preprocess error:{}\t{}'.format(e, datetime.now()))

    try:
        r_batch_infer = requests.get(web_api['batchInfer'].format(domain, 'none'))
        r_batch_infer = json.loads(r_batch_infer.content.decode('utf-8'))
        if r_batch_infer['code'] == 500:
            print('Web server internal error, api:batchInfer\t{}'.format(datetime.now()))
            exit()
        print('web_api:batchInfer:{}\t{}'.format(r_batch_infer, datetime.now()))
    except Exception as e:
        print('web_api:batchInfer error:{}\t{}'.format(e, datetime.now()))

    this_thurs_str = time.strftime("%Y-%m-%d")
    this_thurs_date = datetime.strptime(this_thurs_str, "%Y-%m-%d")
    t = this_thurs_date.timetuple()
    this_thurs_timestamp = int(time.mktime(t)) * 1000
    last_sun_timestamp = this_thurs_timestamp - ONE_DAY_LENGTH * 4
    last_mon_timestamp = this_thurs_timestamp - ONE_DAY_LENGTH * 10

    send_email(web_api, last_mon_timestamp, last_sun_timestamp)
