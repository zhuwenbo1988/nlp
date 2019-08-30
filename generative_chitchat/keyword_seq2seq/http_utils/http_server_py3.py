#!/usr/bin/env python
# coding=UTF-8


import socketserver
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse
import urllib
import json


class ThreadedHTTPRequestHandler(BaseHTTPRequestHandler):
    dispatchers = {}


    def _set_headers(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Request-Method", "*")
        self.end_headers()

    def do_HEAD(self):
        self._set_headers()

    @classmethod
    def Register(self, path, fun):
        self.dispatchers[path] = fun

    def do_GET(self):
        self.log_message("in get method")
        response = {}
        response['status'] = 'error'
        response['msg'] = 'get method is invalid'
        response = json.dumps(response, ensure_ascii=False)
        response = response.encode('utf-8')
        # send data
        self._set_headers()
        self.wfile.write(response)

    def do_POST(self):
        self.log_message("in post method")
        data_string = self.rfile.read(int(self.headers['Content-Length']))
        print('post input :', data_string)
        # parse path
        o = urlparse(self.path)
        # parse data
        data_string = data_string.decode('utf-8')
        params = json.loads(data_string)
        # process
        if o.path in self.dispatchers:
            fun = self.dispatchers[o.path]
            response = fun(params)
        else:
            result = 'error url path: {}'.format(data_string)
            response = {}
            response['status'] = 'error'
            response['msg'] = result
            response = json.dumps(response, ensure_ascii=False)
        print('post output :', response)
        response = response.encode('utf-8')
        # send data
        self._set_headers()
        self.wfile.write(response)


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


class HttpServer:

    def __init__(self, host, port):
        self.server = ThreadedTCPServer(
            (host, port), ThreadedHTTPRequestHandler)

    def Register(self, path, fun):
        ThreadedHTTPRequestHandler.Register(path, fun)

    def Start(self):
        self.server.serve_forever()

    def ShutDown(self, *args):
        self.server.shutdown()