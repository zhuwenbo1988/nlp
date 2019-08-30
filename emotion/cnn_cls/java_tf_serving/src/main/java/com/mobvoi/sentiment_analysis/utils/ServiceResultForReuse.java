// Copyright (c) 2016 Mobvoi Inc. All Rights Reserved.
package com.mobvoi.sentiment_analysis.utils;

/**
 * ServiceResult becomes a public class, for reusing under the whole deepqa frame.
 * 
 * @author Cong Yue, <congyue@mobvoi.com>
 * @date 2016/6/2
 */
public class ServiceResultForReuse {
  public String query;  // query after filtering stop_words
  public String originalQuery;
  public String serviceLine;
  public int[] dependencyInfos;
  public String[] posTags;
  public String[] namedEntitys;
  public String[] wordSegments;

  public ServiceResultForReuse() {
  }

  public ServiceResultForReuse(String query, String serviceLine, int[] dependencyInfos,
      String[] posTags, String[] namedEntitys, String[] wordSegments) {
    this.query = query;
    this.serviceLine = serviceLine;
    this.dependencyInfos = dependencyInfos;
    this.posTags = posTags;
    this.namedEntitys = namedEntitys;
    this.wordSegments = wordSegments;
  }

  public String toString() {
    String result = "";
    result += "query: " + this.query + "\t" + this.originalQuery + "\n";
    for (String word : this.wordSegments) {
      result += word + " ";
    }
    result += "\n";
    for (String pos : this.posTags) {
      result += pos + " ";
    }
    result += "\n";
    for (String ner : this.namedEntitys) {
      result += ner + " ";
    }
    result += "\n";
    for (int dependencyRelation : this.dependencyInfos) {
      result += dependencyRelation + " ";
    }
    result += "\n";
    return result;
  }
}
