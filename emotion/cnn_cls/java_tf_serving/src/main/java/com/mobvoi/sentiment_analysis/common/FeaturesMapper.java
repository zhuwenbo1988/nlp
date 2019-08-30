// Copyright (c) 2016 Mobvoi Inc. All Rights Reserved.
package com.mobvoi.sentiment_analysis.common;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.mobvoi.sentiment_analysis.servlet.SentimentAnalysisServlet;
import com.mobvoi.sentiment_analysis.utils.SentimentResourceReader;

/**
 * features extractor
 * 
 * @author wbzhu <wbzhu@mobvoi.com>
 * @Date 11 18, 2018
 */
public class FeaturesMapper {

  public static final String SENTIMENT_MODEL_PATH = SentimentAnalysisServlet.WEB_PATH
          + "WEB-INF/classes/export/chatbot_model/model";
  public static final String SENTIMENT_RESOURCES_PATH = SentimentAnalysisServlet.WEB_PATH
          + "WEB-INF/classes/export/chatbot_model/resources";

  private Map<String, Integer> label2idx = new HashMap<String, Integer>();
  private Map<Integer, String> idx2label = new HashMap<Integer, String>();

  private Map<String, Integer> word2idx = new HashMap<String, Integer>();
  private Map<String, String> sentimentWord2SentimentLabel = new HashMap<String, String>();
  private Map<String, Pattern> regexName2Pattern = new HashMap<String, Pattern>();
  private Map<String, Pattern> negationName2Pattern = new HashMap<String, Pattern>();

  public Map<String, Integer> configParams = new HashMap<String, Integer>();

  public static final int DEFAULT_WORD_IDX = 1;

  private static FeaturesMapper mapper;
  static {
    mapper = new FeaturesMapper();
    mapper.init();
  }

  private FeaturesMapper() {
  }

  public static FeaturesMapper getInstance() {
    return mapper;
  }

  public void init() {
    loadLabel();

    loadVocab();
    loadRegexPattern();
    loadSentimentDict();
    loadNegationPattern();

    loadConfig();
  }

  private void loadVocab() {
    String vocabLocation = SENTIMENT_RESOURCES_PATH + "/vocab/vocab";
    List<String> lines = SentimentResourceReader.read(vocabLocation);
    int idx = 0;
    for (String line : lines) {
      word2idx.put(line, idx);
      idx++;
    }
  }

  private void loadLabel() {
    String labelLocation = SENTIMENT_RESOURCES_PATH + "/vocab/vocab_label";
    List<String> lines = SentimentResourceReader.read(labelLocation);
    int idx = 0;
    for (String line : lines) {
      label2idx.put(line, idx);
      idx2label.put(idx, line);
      idx++;
    }
  }

  private void loadRegexPattern() {
    String regexLocation = SENTIMENT_RESOURCES_PATH + "/regex/regex_pattern";
    List<String> lines = SentimentResourceReader.read(regexLocation);
    for (String line : lines) {
      String[] ss = line.split("=");
      String name = ss[0];
      String patternStr = ss[1];
      Pattern pattern = Pattern.compile(patternStr);
      regexName2Pattern.put(name, pattern);
    }
  }

  private void loadSentimentDict() {
    String dictLocation = SENTIMENT_RESOURCES_PATH + "/sentiment_dict/word2sentiment";
    List<String> lines = SentimentResourceReader.read(dictLocation);
    for (String line : lines) {
      String[] ss = line.split("\t");
      String word = ss[0];
      String label = ss[1];
      sentimentWord2SentimentLabel.put(word, label);
    }
  }

  private void loadNegationPattern() {
    String negationRegexLocation = SENTIMENT_RESOURCES_PATH + "/negation_regex/neg_regex";
    List<String> lines = SentimentResourceReader.read(negationRegexLocation);
    for (String line : lines) {
      String[] ss = line.split("=");
      String name = ss[0];
      String patternStr = ss[1];
      Pattern pattern = Pattern.compile(patternStr);
      negationName2Pattern.put(name, pattern);
    }
  }

  private void loadConfig() {
    String configLocation = SENTIMENT_RESOURCES_PATH + "/config";
    List<String> lines = SentimentResourceReader.read(configLocation);
    for (String line : lines) {
      String[] ss = line.split("=");
      String name = ss[0];
      int val = Integer.valueOf(ss[1]);
      configParams.put(name, val);
    }
  }

  public long[][] mapWordIds(List<String> words) {
    int modelSequenceLength = configParams.get("word_vector_length");
    long[][] vector = new long[1][modelSequenceLength];
    int max = modelSequenceLength > words.size() ? words.size() : modelSequenceLength;
    for (int i = 0; i < max; i++) {
      String word = words.get(i);
      if (word2idx.containsKey(word)) {
        vector[0][i] = word2idx.get(word);
      } else {
        vector[0][i] = DEFAULT_WORD_IDX;
      }
    }
    return vector;
  }

  public long[][] mapSentimentWordIds(List<String> words) {
    int modelSequenceLength = configParams.get("sentiment_vector_length");
    long[][] vector = new long[1][modelSequenceLength];
    // padding
    int numClasses = label2idx.size();
    int idx = 0;
    for (String word : words) {
      if (idx >= modelSequenceLength)
        break;
      if (sentimentWord2SentimentLabel.containsKey(word)) {
        vector[0][idx] = label2idx.get(sentimentWord2SentimentLabel.get(word));
        idx++;
      }
    }
    for (int i = idx; i < modelSequenceLength; i++) {
      vector[0][i] = numClasses;
    }
    return vector;
  }

  public long[][] mapRegexIds(String sent) {
    long[][] vector = new long[1][configParams.get("regex_vector_length")];
    int numClasses = label2idx.size();
    List<Integer> tmp = new ArrayList<Integer>();
    for (Entry<String, Pattern> entry : regexName2Pattern.entrySet()) {
      String name = entry.getKey();
      Pattern pattern = entry.getValue();
      Matcher result = pattern.matcher(sent);
      if (result.find()) {
        String label = name.split("_")[0].toLowerCase();
        int idx = label2idx.get(label);
        if (!tmp.contains(idx)) {
          tmp.add(idx);
        }
      }
    }
    for (int i = 0; i < configParams.get("regex_vector_length"); i++) {
      if (i < tmp.size()) {
        vector[0][i] = tmp.get(i);
      } else {
        vector[0][i] = numClasses;
      }
    }
    return vector;
  }

  public long[] mapNegationIds(String sent) {
    long[] vector = new long[1];
    for (Entry<String, Pattern> entry : negationName2Pattern.entrySet()) {
      String name = entry.getKey();
      Pattern pattern = entry.getValue();
      Matcher result = pattern.matcher(sent);
      if (result.find()) {
        String type = name.split("_")[0];
        if ("NEG".equals(type)) {
          vector[0] = 1;
          return vector;
        }
        if ("POS".equals(type)) {
          vector[0] = 2;
          return vector;
        }
      }
    }
    vector[0] = 0;
    return vector;
  }

  public String mapLabel(long idx) {
    int i = (int) idx;
    if (idx2label.containsKey(i)) {
      return idx2label.get(i);
    } else {
      return null;
    }
  }

  public static void main(String[] args) {
    String s = "我 是 傻逼 你 是 傻逼 我 是 傻逼 你 是 笨猪";
    List<String> words = Arrays.asList(s.split(" "));
    s = s.replace(" ", "");
    System.out.println(FeaturesMapper.getInstance().mapWordIds(words));
    System.out.println(FeaturesMapper.getInstance().mapSentimentWordIds(words));
    System.out.println(FeaturesMapper.getInstance().mapRegexIds(s));
    System.out.println(FeaturesMapper.getInstance().mapNegationIds(s));
  }

}
