#!/usr/bin/env python
# coding=UTF-8
# chaoli cli@mobvoi.com

import sys
import socket
import SocketServer
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import urlparse
import json
from service_result_obj import ResponseObj


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
        o = urlparse.urlparse(self.path)
        params = urlparse.parse_qs(o.query)
        if o.path in self.dispatchers:
            fun = self.dispatchers[o.path]
            response = fun(params)
        else:
            result = 'error url path: {}'.format(params)
            response = ResponseObj('error', result)
            response = response.format()
        # send data
        self._set_headers()
        self.wfile.write(response)

    def do_POST(self):
        self.log_message("in post method")
        data_string = self.rfile.read(int(self.headers['Content-Length']))
        self.log_message('{}'.format(data_string))
        # parse path
        o = urlparse.urlparse(self.path)
        # parse data
        params = json.loads(data_string)
        # process
        if o.path in self.dispatchers:
            fun = self.dispatchers[o.path]
            response = fun(params)
        else:
            result = 'error url path: {}'.format(data_string)
            response = ResponseObj('error', result)
            response = response.format()
        self.log_message('{}'.format(response))
        # send data
        self._set_headers()
        self.wfile.write(response)


class ThreadedTCPServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
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
