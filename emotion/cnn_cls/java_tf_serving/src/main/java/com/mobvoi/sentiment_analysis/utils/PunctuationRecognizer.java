// Copyright (c) 2016 Mobvoi Inc. All Rights Reserved.

package com.mobvoi.sentiment_analysis.utils;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 识别中文和英文标点符号
 * 
 * @author wbzhu <wbzhu@mobvoi.com>
 * @Date 11 18, 2018
 */
public class PunctuationRecognizer {

  // 根据UnicodeBlock方法判断中文标点符号
  public static boolean isChinesePunctuation(char c) {
    Character.UnicodeBlock ub = Character.UnicodeBlock.of(c);
    if (ub == Character.UnicodeBlock.GENERAL_PUNCTUATION
            || ub == Character.UnicodeBlock.CJK_SYMBOLS_AND_PUNCTUATION
            || ub == Character.UnicodeBlock.HALFWIDTH_AND_FULLWIDTH_FORMS
            || ub == Character.UnicodeBlock.CJK_COMPATIBILITY_FORMS
            || ub == Character.UnicodeBlock.VERTICAL_FORMS) {
      return true;
    } else {
      return false;
    }
  }

  public static boolean isEnglishPunctuation(char c) {
    String regEx = "[,.?!;:]";
    Pattern p = Pattern.compile(regEx);
    Matcher m = p.matcher("" + c);
    return m.matches();
  }

}
