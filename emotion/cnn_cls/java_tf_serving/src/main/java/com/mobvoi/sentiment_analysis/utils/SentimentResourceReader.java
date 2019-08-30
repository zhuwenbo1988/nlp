package com.mobvoi.sentiment_analysis.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;

public class SentimentResourceReader {
  private static Logger logger = Logger.getLogger(SentimentResourceReader.class.getName());
  
  public static List<String> read(String fileLocation) {
    List<String> lines = new ArrayList<String>();
    
    BufferedReader br = null;
    try {
      br = new BufferedReader(new FileReader(fileLocation));
      String s = "";
      while ((s = br.readLine()) != null) {// 一行一行读
        lines.add(s);
      }
    } catch (IOException e) {
      logger.info("load resource failed");
    } finally {
      try {
        br.close();
      } catch (IOException e) {
        logger.info("close resource failed");
      }
    }
    return lines;
  }
}
