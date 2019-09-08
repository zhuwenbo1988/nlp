# coding=utf-8

import sys
import json
import requests
import time
import hashlib
import base64
import urllib
from http_server import HttpServer
from service_result_obj import ResponseObj
import ConfigParser


TULING = "tuling"
TENCENT = "tencent"
IFLY = "ifly"
UNISOUND = "unisound"
config = ConfigParser.ConfigParser()


def tuling_api(params):
    print(params)

    query = None
    api_key = None
    user_id = None
    # parameters
    if 'query' in params:
        query = params['query']
        if type(query) is list:
            query = query[0]
        if type(query) is unicode:
            query = query.encode('utf-8')
    if 'api_key' in params:
        api_key = params['api_key']
    if 'user_id' in params:
        user_id = params['user_id']
    # 参数检查
    if not query:
        return ResponseObj('error', 'parse query failed').format()
    if not api_key:
        api_key = config.get(TULING, "default_api_key")
    if not user_id:
        user_id = config.get(TULING, "default_user_id")

    print("query: {}".format(query))
    print("api_key: {}".format(api_key))
    print("user_id: {}".format(user_id))

    url = config.get(TULING, "service_url")
    input = {"reqType": 0, "perception": {"inputText": {"text": ""}},
             "userInfo": {"apiKey": "", "userId": ""}}
    # 设置参数
    input['perception']['inputText']['text'] = query.decode('utf-8')
    input['userInfo']['apiKey'] = api_key
    input['userInfo']['userId'] = user_id
    try:
        r = requests.post(url, data=json.dumps(input))
    except Exception as e:
        # 失败
        return ResponseObj('error', 'call tuling api failed').format()
    # 成功
    return ResponseObj('success', json.loads(r.text)).format()


def tencent_api(params):
    print(params)

    query = None
    app_key = None
    app_id = None
    # parameters
    if 'query' in params:
        query = params['query']
        if type(query) is list:
            query = query[0]
        if type(query) is unicode:
            query = query.encode('utf-8')
    if 'app_key' in params:
        app_key = params['app_key']
    if 'app_id' in params:
        app_id = params['app_id']
    # 参数检查
    if not query:
        return ResponseObj('error', 'parse query failed').format()
    if not app_key:
        app_key = config.get(TENCENT, "default_app_key")
    if not app_id:
        app_id = config.get(TENCENT, "default_app_id")

    print("query: {}".format(query))
    print("app_key: {}".format(app_key))
    print("app_id: {}".format(app_id))

    url = config.get(TENCENT, "service_url")
    # 请求参数
    req_params = {}
    req_params['app_id'] = app_id
    req_params['session'] = '10000'
    req_params['question'] = query
    req_params['time_stamp'] = int(time.time())
    req_params['nonce_str'] = "fa577ce340859f9fe"
    # 生成签名
    def get_sign(req_params, app_key):
        l = []
        for k in sorted(req_params):
            l.append((k, req_params[k]))
        l.append(('app_key', app_key))
        # 保证utf-8编码
        s = urllib.urlencode(l).encode('utf-8')
        m2 = hashlib.md5()
        m2.update(s)
        checkSum = m2.hexdigest()
        sign = checkSum.upper()
        return sign
    req_params['sign'] = get_sign(req_params, app_key)

    header = {'Content-Type': 'application/x-www-form-urlencoded'}

    try:
        r = requests.post(url, headers=header, data=req_params)
    except Exception as e:
        # 失败
        return ResponseObj('error', 'call tencent api failed').format()
    # 成功
    return ResponseObj('success', json.loads(r.text)).format()


def ifly_api(params):
    print(params)

    query = None
    app_key = None
    app_id = None
    auth_id = None
    # parameters
    if 'query' in params:
        query = params['query']
        if type(query) is list:
            query = query[0]
        if type(query) is unicode:
            query = query.encode('utf-8')
    if 'app_key' in params:
        app_key = params['app_key']
    if 'app_id' in params:
        app_id = params['app_id']
    if 'auth_id' in params:
        auth_id = params['auth_id']
    # 参数检查
    if not query:
        return ResponseObj('error', 'parse query failed').format()
    if not app_key:
        app_key = config.get(IFLY, "default_app_key")
    if not app_id:
        app_id = config.get(IFLY, "default_app_id")
    if not auth_id:
        auth_id = config.get(IFLY, "default_auth_id")

    print("query: {}".format(query))
    print("app_key: {}".format(app_key))
    print("app_id: {}".format(app_id))
    print("auth_id: {}".format(auth_id))

    url = config.get(IFLY, "service_url")

    def getHeader(app_key, app_id, auth_id):
        curTime = str(int(time.time()))
        param = "{\"result_level\":\"complete\",\"auth_id\":\"" + auth_id + "\",\"data_type\":\"text\",\"scene\":\"main\"}"
        paramBase64 = base64.b64encode(param)

        m2 = hashlib.md5()
        m2.update(app_key + curTime + paramBase64)
        checkSum = m2.hexdigest()

        header = {
            'X-CurTime': curTime,
            'X-Param': paramBase64,
            'X-Appid': app_id,
            'X-CheckSum': checkSum,
        }
        return header

    try:
        r = requests.post(url, headers=getHeader(app_key, app_id, auth_id), data=query)
    except Exception as e:
        # 失败
        return ResponseObj('error', 'call ifly api failed').format()
    # 成功
    return ResponseObj('success', json.loads(r.text)).format()


def unisound_api(params):
    print(params)

    query = None
    app_key = None
    sess_id = None
    user_id = None
    visitor_id = None

    # parameters
    if 'query' in params:
        query = params['query']
        if type(query) is list:
            query = query[0]
        if type(query) is unicode:
            query = query.encode('utf-8')
    if 'app_key' in params:
        app_key = params['app_key']
    if 'session_id' in params:
        sess_id = params['session_id']
    if 'creator_id' in params:
        user_id = params['creator_id']
    if 'visitor_id' in params:
        visitor_id = params['visitor_id']
    # 参数检查
    if not query:
        return ResponseObj('error', 'parse query failed').format()
    if not app_key:
        app_key = config.get(UNISOUND, "default_app_key")
    if not sess_id:
        sess_id = config.get(UNISOUND, "JSESSIONID")
    if not user_id:
        user_id = config.get(UNISOUND, "creatorId")
    if not visitor_id:
        visitor_id = config.get(UNISOUND, "visitorUid")

    print("query: {}".format(query))
    print("app_key: {}".format(app_key))
    print("session_id: {}".format(sess_id))
    print("creator_id: {}".format(user_id))
    print("visitor_id: {}".format(visitor_id))

    request_url = config.get(UNISOUND, "service_url")

    cookies = dict(JSESSIONID=sess_id, visitorUid=visitor_id)
    headers = {'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'}
    # 请求参数
    req_params = {}
    req_params['appKey'] = app_key
    req_params['question'] = query
    req_params['creatorId'] = user_id

    try:
        r = requests.post(url=request_url, headers=headers, data=req_params, cookies=cookies)
    except Exception as e:
        # 失败
        return ResponseObj('error', 'call unisound api failed').format()
    # 成功
    return ResponseObj('success', json.loads(r.text)).format()


if __name__ == "__main__":
    # 服务端口
    try:
        port = int(sys.argv[1])
    except:
        print("USAGE: service port")
        sys.exit(-1)
    # 加载配置文件
    config_file = sys.argv[2]
    config.read(config_file)
    # 注册服务
    HOST, PORT = "", port
    server = HttpServer(HOST, PORT)
    server.Register("/tuling", tuling_api)
    server.Register("/tencent", tencent_api)
    server.Register("/ifly", ifly_api)
    server.Register("/unisound", unisound_api)
    # TODO
    server.Register("/tuling/", tuling_api)
    server.Register("/tencent/", tencent_api)
    server.Register("/ifly/", ifly_api)
    server.Register("/unisound/", unisound_api)
    server.Start()