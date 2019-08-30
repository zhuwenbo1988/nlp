package com.mobvoi.sentiment_analysis.utils;

import java.net.URLDecoder;
import java.util.concurrent.ExecutorService;

import org.apache.log4j.Logger;

import com.alibaba.fastjson.JSONObject;
import com.mobvoi.be.common.http.multi.HttpMultiCaller;
import com.mobvoi.be.utils.MultiCallerUtil;

public class MultiCallerWrapper {
  
  private static final Logger logger = Logger.getLogger(MultiCallerWrapper.class.getName());
  
  public static JSONObject get(String url, HttpMultiCaller caller, long timeout, ExecutorService executor) throws Exception {
    long start = System.currentTimeMillis();
    JSONObject response = MultiCallerUtil.getJson(url, caller, timeout, executor);
    long timeCost = System.currentTimeMillis() - start;
    logger.info(String.format("[%s] total elapsed time(milliseconds): %d", URLDecoder.decode(url, "UTF-8"), timeCost));
    return response;
  }
}
