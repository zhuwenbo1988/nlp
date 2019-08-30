
package com.mobvoi.sentiment_analysis.utils;

import java.io.UnsupportedEncodingException;
import java.net.URISyntaxException;
import java.net.URLEncoder;
import java.util.concurrent.ThreadPoolExecutor;

import org.apache.http.client.utils.URIBuilder;
import org.apache.log4j.Logger;

import com.mobvoi.be.common.http.multi.HttpMultiCaller;
import com.mobvoi.be.utils.MultiCallerUtil;

/**
 * Build a class ServieceResult to represent the information of the processing result of one query,
 * including segment, PosTag, NER, and dependency parsing.
 * 
 * @author Cong Yue, congyue@mobvoi.com
 * @Date Apr 16, 2016
 */

public class StanfordResultParser {
  private static Logger logger = Logger.getLogger(StanfordResultParser.class.getName());
  private static String serviceURL;
  private static final long TIMEOUT = 1000;
  private static ThreadPoolExecutor executor = MultiCallerUtil.makeExecutor();
  private static HttpMultiCaller caller = MultiCallerUtil.makeMultiCaller("stanford_wordseg");

  static {
    serviceURL = ConfigReader.getProp("serviceURL");
    if (!serviceURL.startsWith("http://"))
      serviceURL = "http://" + serviceURL;
  }

  private static int[] parseDenpendencyTree(String dependencyLine, boolean isDebugMode) {
    // \t split below will result in split error, so replace root so that we can split by 't
    dependencyLine = dependencyLine.replace("root", "roor");
    String[] dependencies = dependencyLine.split("    "); // t[A-Z]|troor

    int[] result = new int[dependencies.length + 1];
    for (String dependency : dependencies) {
      // if (isDebugMode)
      // System.out.println("dependency: " + dependency);
      int firstSplitIndex = dependency.indexOf("(");
      if (firstSplitIndex < 0)
        firstSplitIndex = 0;
      int lastSplitIndex = dependency.length() - 1;
      String wordToWord = dependency.substring(firstSplitIndex + 1, lastSplitIndex);
      String[] twoWords = wordToWord.split(",");
      if (twoWords.length == 2) {
        int i = Integer.parseInt(twoWords[0].split("-")[1]);
        // training data may have character '-'
        int j = Integer.parseInt(twoWords[1].split("-")[1]);
        result[j] = i;
      } else {
        // System.out.println("dependency parse Line Format Error!");
      }
      // if (isDebugMode)
      // System.out.println(wordToWord);
    }
    if (isDebugMode) {
      for (int i : result) {
        System.out.print(i);
      }
    }
    return result;
  }

  public static ServiceResultForReuse httpServiceGet(String query, boolean isDebugMode) {
    ServiceResultForReuse serviceResult = null;
    try {
      String[] replaceStopCharacter = { ";", "-", ".", ",", "。", "，", "{", "}", "[", "]", ":", "：",
          "<", ">", "|", "\\", "/", "'", "+", "=", "_", "`", "—", "(", ")", "-", "*", "&", "~", "!",
          "！", "？", "?", "～", "^", "%", "#", "（", "）", "”", "‘", "、", "@", " ", " " };
      for (String str : replaceStopCharacter) {
        query = query.replace(str, "");
      }
      String line = fetchStanfordRlt(query);
      if (line == null) {
        return serviceResult;
      }
      int[] dependencyInfos = null;
      String[] posTags = null;
      String[] namedEntitys = null;
      String[] wordSegments = null;
      String content = line.replace("{\"", "").replace("\"}", "");
      String[] services = content.split("[\"],[\"]");
      for (int i = 0; i < services.length; i++) {
        String[] serviceContent = services[i].split("\":\"");
        if (serviceContent[0].equals("dep_praser")) {
          dependencyInfos = parseDenpendencyTree(serviceContent[1], isDebugMode);
        }
        if (serviceContent[0].equals("pos_tag")) {
          posTags = serviceContent[1].split(" ");
        }
        if (serviceContent[0].equals("named_entity")) {
          namedEntitys = serviceContent[1].split(" ");
        }
        if (serviceContent[0].equals("word_segments")) {
          wordSegments = serviceContent[1].split(" ");
        }
        // Caution: dependencyInfos length = wordSegments length + 1
        serviceResult = new ServiceResultForReuse(query, line, dependencyInfos, posTags,
                namedEntitys, wordSegments);
      }
    } catch (Exception e) {
      logger.error("stanford service failed: " + e);
      return null;
    }
    return serviceResult;
  }

  public static String fetchStanfordRlt(String query) {
    if (query == null || query.isEmpty()) {
      return null;
    }

    URIBuilder uriBuilder;
    try {
      String url = serviceURL + URLEncoder.encode(query, "UTF-8");
      uriBuilder = new URIBuilder(url);
    } catch (URISyntaxException e) {
      logger.error("new URI builder fail: " + e);
      return null;
    } catch (UnsupportedEncodingException e) {
      logger.error("new URI builder fail: " + e);
      return null;
    }

    String stanfordRlt = null;
    try {
      stanfordRlt = MultiCallerWrapper.get(uriBuilder.toString(), caller, TIMEOUT, executor).toString();
    } catch (Exception e) {
      logger.error("get response error: " + e);
      return null;
    }
    return stanfordRlt;
  }

  public static void main(String[] args) {
    ServiceResultForReuse sr = httpServiceGet("如何查看Ticwatch系统的版本号", true);
    // parseDenpendencyTree("NMOD(总统-4, 美国-1)\tAMOD(任-3, 第一-2)\tNMOD(总统-4, 任-3)\tSUB(是-5,
    // 总统-4)\troot(ROOT-0, 是-5)\tPRD(是-5, 谁-6)")----- Ticwatch
    System.out.println("--------------");
    System.out.println(sr.toString());
  }

}
