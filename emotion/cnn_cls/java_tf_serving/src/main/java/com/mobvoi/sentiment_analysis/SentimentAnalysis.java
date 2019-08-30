// Copyright (c) 2016 Mobvoi Inc. All Rights Reserved.
package com.mobvoi.sentiment_analysis;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.log4j.Logger;
import org.json.JSONArray;
import org.json.JSONObject;

import com.mobvoi.sentiment_analysis.business.businessSentimentModel;
import com.mobvoi.sentiment_analysis.common.CommonSentimentModel;
import com.mobvoi.sentiment_analysis.servlet.SentimentAnalysisServlet;
import com.mobvoi.sentiment_analysis.utils.PunctuationRecognizer;
import com.mobvoi.sentiment_analysis.utils.ServiceResultForReuse;
import com.mobvoi.sentiment_analysis.utils.StanfordResultParser;

/**
 * sentiment analysis Controller
 * 
 * @author wbzhu <wbzhu@mobvoi.com>
 * @Date 11 18, 2018
 */
public class SentimentAnalysis {
  private static Logger logger = Logger.getLogger(SentimentAnalysis.class.getName());
  private static final String SENTIMENT_DICT_PATH = SentimentAnalysisServlet.WEB_PATH
          + "WEB-INF/classes/sentiment_dict.txt";

  private Map<String, Integer> word2sentimentMap;
  List<String> sentList = Arrays.asList("None", "Happy", "Sad", "Angry", "Fear", "Surprise");

  private void loadSentimentMap() {
    word2sentimentMap = new HashMap<>();
    BufferedReader bf;
    try {
      bf = new BufferedReader(new FileReader(SENTIMENT_DICT_PATH));
      String tmpLine;
      while ((tmpLine = bf.readLine()) != null) {
        String[] parts = tmpLine.split("\t");
        if (parts.length == 2) {
          word2sentimentMap.put(parts[0], Integer.valueOf(parts[1]));
        }
      }
      bf.close();
    } catch (NumberFormatException | IOException e) {
      logger.error("Load Sentiment Map error");
    }
  }

  public SentimentAnalysis() {
    init();
  }

  public void init() {
    logger.info("sentiment analysis initing...");
    // 加载情感词典
    loadSentimentMap();
    // 加载闲聊(通用)情感分类模型
    CommonSentimentModel.getInstance().init();
    // 加载商业化部门(限定域)情感分类模型
    businessSentimentModel.getInstance().init();
  }

  private int getSentimentFromDict(String string) {
    int rlt = 0;
    int maxlen = 0;
    for (Entry<String, Integer> e : word2sentimentMap.entrySet()) {
      if (string.contains(e.getKey())) {
        if (e.getKey().length() > maxlen) {
          rlt = e.getValue();
          maxlen = e.getKey().length();
        }
      }
    }
    return rlt;
  }

  /*
   * return sentiment for string, 0: None, 1: happy, 2: sad, 3: angry "rule_id":"id" }
   */
  public int getSentiment(String string) {
    int rlt = 0;
    rlt = getSentimentFromDict(string);
    return rlt;
  }

  public JSONObject query(InputObj in) {
    long start = System.currentTimeMillis();
    
    String query = in.getRawQuery();
    if (query == null || query.length() == 0) {
      logger.warn("query must contains something");
      return generateAnswerJson(null, start);
    }
    // analysis sentiment from dict has been deprecated
    /*
     * int senId = 0; senId = getSentiment(query); String senStr = sentList.get(senId);
     */

    List<String> wordsegs = in.getWords();
    if (wordsegs == null) {
      ServiceResultForReuse sr = StanfordResultParser.httpServiceGet(query, false);
      if (sr == null || sr.wordSegments == null || sr.wordSegments.length == 0) {
        logger.warn("can not get word segments from stanford service");
        return generateAnswerJson(null, start);
      }
      wordsegs = Arrays.asList(sr.wordSegments);
    }
    if (wordsegs.size() == 0) {
      logger.warn("word segments is nothing");
      return generateAnswerJson(null, start);
    }
    
    JSONObject result = null;
    // 通用情感分类
    if (CommonSentimentModel.MODEL_NAME.equals(in.getTaskModelType())) {
      result = CommonSentimentModel.getInstance().classicify(query, wordsegs);
    } else if (businessSentimentModel.MODEL_NAME.equals(in.getTaskModelType())) {
      result = businessSentimentModel.getInstance().classicify(query, wordsegs);
    } else {
      logger.warn("model type is not specified");
      return generateAnswerJson(null, start);
    }
    
    logger.info(String.format("%s : %s", query, result));
    return generateAnswerJson(result, start);
  }
  
  private JSONObject processShortText(String s, String modelType) {
    InputObj in = new InputObj();
    in.rawQuery = s;
    in.taskModelType = modelType;
    JSONObject res = query(in);
    return res;
  }
  
  private List<String> splitDoc(String doc) {
    logger.info(String.format("split long text: %s", doc));
    
    List<String> shortSentenceList = new ArrayList<String>();
    StringBuilder s = new StringBuilder();
    for (char c : doc.toCharArray()) {
      if (PunctuationRecognizer.isChinesePunctuation(c) || PunctuationRecognizer.isEnglishPunctuation(c)) {
        String shortText = s.toString();
        logger.info(String.format("short text : %s", shortText));
        shortSentenceList.add(shortText);
        // 清空
        s.delete(0, s.length());
        continue;
      }
      s.append(c);
    }
    // 最后一句
    if (s.length() > 0) {
      String shortText = s.toString();
      logger.info(String.format("short text : %s", shortText));
      shortSentenceList.add(shortText);
    }
    return shortSentenceList;
  }
  
  public JSONObject processLongText(InputObj in) {
    long start = System.currentTimeMillis();
    
    String query = in.getRawQuery();
    if (query == null || query.length() == 0) {
      logger.warn("query must contains something");
      return generateAnswerJson(null, start);
    }
    
    JSONArray all = new JSONArray();
    List<String> ss = splitDoc(query);
    for (String s : ss) {
      JSONObject partResult = processShortText(s, in.getTaskModelType());
      all.put(partResult);
    }
    
    return generateLongAnswerJson(all, start);
  }
  
  private JSONObject generateAnswerJson(JSONObject answer, long startTime) {
    JSONObject obj = new JSONObject();

    long end = System.currentTimeMillis();
    String time = (end - startTime) + "ms";
    
    if (answer != null) {
      obj.put("answer", answer.get("label"));
      obj.put("score", answer.get("score"));
    }
    obj.put("timeCost", time);
    obj.put("status", "ok");
    return obj;
  }
  
  private JSONObject generateLongAnswerJson(JSONArray answer, long startTime) {
    JSONObject obj = new JSONObject();

    long end = System.currentTimeMillis();
    String time = (end - startTime) + "ms";
    
    obj.put("answer", answer);
    obj.put("timeCost", time);
    obj.put("status", "ok");
    return obj;
  }
  
  public class InputObj {
    private String rawQuery;
    private List<String> words;
    private String taskModelType;
    
    public InputObj() {
    }

    public String getRawQuery() {
      return rawQuery;
    }

    public void setRawQuery(String rawQuery) {
      this.rawQuery = rawQuery;
    }

    public List<String> getWords() {
      return words;
    }

    public void setWords(List<String> words) {
      this.words = words;
    }

    public String getTaskModelType() {
      return taskModelType;
    }

    public void setTaskModelType(String taskModelType) {
      this.taskModelType = taskModelType;
    }
  }
}
