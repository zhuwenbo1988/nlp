// Copyright (c) 2016 Mobvoi Inc. All Rights Reserved.
package com.mobvoi.sentiment_analysis.servlet;

import java.io.IOException;

import javax.servlet.ServletConfig;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;
import org.json.JSONObject;

/**
 * @Author Cong Yue, <congyue@mobvoi.com>
 * @Date 6/14/16
 */
public class StatusCheckServlet extends HttpServlet {
  public static String WEB_PATH = "src/main/webapp/";

  private Logger logger = Logger.getLogger(StatusCheckServlet.class.getName());

  public void init(ServletConfig config) throws ServletException {
    super.init(config);
    WEB_PATH = config.getServletContext().getRealPath("/");
    if (!WEB_PATH.endsWith("/")) {
      WEB_PATH = WEB_PATH + "/";
    }
    PropertyConfigurator.configure(WEB_PATH + "WEB-INF/config/log4j.properties");
  }

  public void doGet(HttpServletRequest request, HttpServletResponse response)
      throws IOException {
    request.setCharacterEncoding("UTF-8");
    String queryStr = request.getParameter("query");

    long start = System.currentTimeMillis();
    if (queryStr == null){
      logger.info("status checked.");
    }
    response.setContentType("text/html; charset=UTF-8");
    response.setCharacterEncoding("UTF8");
    
    JSONObject obj = new JSONObject();
    long end = System.currentTimeMillis();
    String time = (end - start)  + "ms";

    obj.put("timeCost", time);
    obj.put("status", "ok");
   
    response.setHeader("Content-type", "application/json");
    response.getWriter().print(obj + "\n");
  }
}