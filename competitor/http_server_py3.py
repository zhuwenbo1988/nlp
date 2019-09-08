#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 22:02:46 2019

@author: huataowang@mobvoi.com
"""

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
        o = urlparse(self.path)
        params = urllib.parse.parse_qs(o.query)
        fun = self.dispatchers[o.path]
        response = fun(params)
        self._set_headers()
        self.wfile.write(response.encode('utf-8'))

    def do_POST(self):
        self.log_message("in post method")
        data_string = self.rfile.read(int(self.headers['Content-Length']))
        data_string = data_string.decode('utf-8')
        self.log_message('log: {}'.format(data_string))
        # parse path
        o = urlparse(self.path)
        # parse data
        params = json.loads(data_string)
        # process
        fun = self.dispatchers[o.path]
        response = fun(params)
        self._set_headers()
        self.wfile.write(response.encode('utf-8'))

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

def Test(params):
    return "Test:" + str(params)

if __name__ == "__main__":
    HOST, PORT = "", 38000
    server = HttpServer(HOST, PORT)
    server.Register("/test", Test)
    server.Start()
