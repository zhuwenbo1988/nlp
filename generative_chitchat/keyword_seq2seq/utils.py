#!/usr/bin/env python
# coding: utf-8


import sys
import time
import os
import subprocess
import shlex
import math
from time import time as ttime
 
import tensorflow as tf
 

def print_time(s, start_time):
    """Take a start time, print elapsed duration, and return a new time."""

    print("%s, time %ds, %s." % (s, (time.time() - start_time), time.ctime()))
    return time.time()


def print_out(s, f=None, new_line=True):
    """Similar to print but with support to flush and output to a file."""

    if isinstance(s, bytes):
        s = s.decode("utf-8")

    if f:
        f.write(s.encode("utf-8"))
        if new_line:
            f.write(b"\n")

    # stdout
    out_s = s.encode("utf-8")
    if not isinstance(out_s, str):
        out_s = out_s.decode("utf-8")
    print(out_s, end="", file=sys.stdout)

    if new_line:
        print()


def add_summary(summary_writer, global_step, tag, value):
  """Add a new summary to the current summary_writer.
  Useful to log things that are not part of the training graph, e.g., tag=BLEU.
  """
  summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
  summary_writer.add_summary(summary, global_step)


def split3(path):
    dir, f = os.path.split(path)
    fname, ext = os.path.splitext(f)

    return dir, fname, ext


def file_name(path):
    _, fname, _ = split3(path)
    return fname


def mkdir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        
        
def rm_if_exists(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred
            
        
def count_lines(file_path):
    if sys.platform != 'win32':
        process = subprocess.Popen(shlex.split("wc -l {}".format(file_path)), stdout=subprocess.PIPE)

        output = process.stdout.readline()
        if output:
            return int(output.split()[0])

    # https://gist.github.com/zed/0ac760859e614cd03652
    with open(file_path, 'rbU') as f:
        return sum(1 for _ in f)
    
    time
def safe_mod(dividend, divisor):
    return (dividend % divisor) if divisor != 0 else 0


def safe_exp(value):
    """Exponentiation with catching of overflow error."""
    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans


class Stopwatch:
    def __init__(self):
        self.start()

    def start(self):
        self.__start = ttime()
        
    def elapsed(self):
        return round(ttime() - self.__start, 3)

    def print(self, log_text):
        print(log_text, 'elapsed: {}s'.format(self.elapsed()))