package com.mobvoi.sentiment_analysis.utils;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

import org.apache.log4j.Logger;

public class ConfigReader {
  protected static Logger logger = Logger.getLogger(ConfigReader.class);
  private static Properties prop;
  static {
    InputStream is = ConfigReader.class.getResourceAsStream("/sentiment.properties");

    prop = new Properties();
    if (is == null) {
      logger.warn("sentiment.config not found.");
    } else {
      try {
        prop.load(is);
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }

  public static String getProp(String key) {
    return prop.getProperty(key);
  }

}
