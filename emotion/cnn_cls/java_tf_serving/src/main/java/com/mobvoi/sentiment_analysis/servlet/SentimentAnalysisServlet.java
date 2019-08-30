// Copyright (c) 2016 Mobvoi Inc. All Rights Reserved.
package com.mobvoi.sentiment_analysis.servlet;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.Arrays;

import javax.servlet.ServletConfig;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;
import org.json.JSONObject;

import com.mobvoi.sentiment_analysis.SentimentAnalysis;
import com.mobvoi.sentiment_analysis.SentimentAnalysis.InputObj;

/**
 * sentiment analysis servlet
 * 
 * @author wbzhu <wbzhu@mobvoi.com>
 * @Date 11 18, 2018
 */
public class SentimentAnalysisServlet extends HttpServlet{
  public static String WEB_PATH = "src/main/webapp/";
  private SentimentAnalysis sentimentAnalysis;
  private Logger logger = Logger.getLogger(SentimentAnalysisServlet.class.getName());
  public void init(ServletConfig config) throws ServletException {
    super.init(config);
    WEB_PATH = config.getServletContext().getRealPath("/");
    if (!WEB_PATH.endsWith("/")) {
      WEB_PATH = WEB_PATH + "/";
    }
    PropertyConfigurator.configure(WEB_PATH + "WEB-INF/config/log4j.properties");
    sentimentAnalysis = new SentimentAnalysis();
  }
  
  public void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
    request.setCharacterEncoding("UTF-8");
    // get parameters
    String queryStr = request.getParameter("query").trim();
    String wordsStr = request.getParameter("words");
    String type = request.getParameter("type");
    // set input
    InputObj in = sentimentAnalysis.new InputObj();
    in.setRawQuery(queryStr);
    if (!(wordsStr == null || "".equals(wordsStr))) {
      wordsStr = wordsStr.trim();
      String[] words = wordsStr.split(" ");
      in.setWords(Arrays.asList(words));
    }
    in.setTaskModelType(type);
    // use model
    JSONObject obj  = sentimentAnalysis.query(in);
    obj.put("query", queryStr);
    obj.put("model", type);
    logger.info("servlet get result: " + obj);

    response.setHeader("Content-type", "application/json");
    response.getWriter().print(obj + "\n");

  }
  
  public void doPost(HttpServletRequest request, HttpServletResponse response)
          throws ServletException, IOException {
    String postContent = getPostRequestAsString(request);
    logger.info(postContent);
    JSONObject postObj = new JSONObject(postContent);
    
    String queryStr = postObj.getString("query").trim();
    String type = postObj.getString("type");
    
    InputObj in = sentimentAnalysis.new InputObj();
    in.setRawQuery(queryStr);
    in.setTaskModelType(type);
    JSONObject obj  = sentimentAnalysis.processLongText(in);
    obj.put("query", queryStr);
    obj.put("model", type);
    logger.info(obj);
    
    response.setCharacterEncoding("UTF-8");
    response.setHeader("Content-type", "application/json");
    response.getWriter().print(obj + "\n");
  }

  public static String getPostRequestAsString(HttpServletRequest request) throws IOException {
    request.setCharacterEncoding("UTF8");
    StringBuilder sb = new StringBuilder();
    String line = null;
    BufferedReader br = request.getReader();
    while ((line = br.readLine()) != null) {
      sb.append(line);
    }
    return sb.toString();
  }
}
