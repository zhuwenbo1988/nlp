package com.mycompany.myproject.utils;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class NumberValidationUtils {
    /**
     * integer (-MAX, MAX)
     */
    public final static String REGEX_INTEGER = "^[-\\+]?\\d+$"; //$NON-NLS-1$
    /**
     * integer [1, MAX)
     */
    public final static String REGEX_POSITIVE_INTEGER = "^\\+?[1-9]\\d*$"; //$NON-NLS-1$
    /**
     * integer (-MAX, -1]
     */
    public final static String REGEX_NEGATIVE_INTEGER = "^-[1-9]\\d*$"; //$NON-NLS-1$
    /**
     * integer [0, MAX), only numeric
     */
    public final static String REGEX_NUMERIC = "^\\d+$"; //$NON-NLS-1$
    /**
     * decimal (-MAX, MAX)
     */
    public final static String REGEX_DECIMAL = "^[-\\+]?\\d+\\.\\d+$"; //$NON-NLS-1$
    /**
     * decimal (0.0, MAX)
     */
    public final static String REGEX_POSITIVE_DECIMAL = "^\\+?([1-9]+\\.\\d+|0\\.\\d*[1-9])$"; //$NON-NLS-1$
    /**
     * decimal (-MAX, -0.0)
     */
    public final static String REGEX_NEGATIVE_DECIMAL = "^-([1-9]+\\.\\d+|0\\.\\d*[1-9])$"; //$NON-NLS-1$
    /**
     * decimal + integer (-MAX, MAX)
     */
    public final static String REGEX_REAL_NUMBER = "^[-\\+]?(\\d+|\\d+\\.\\d+)$"; //$NON-NLS-1$
    /**
     * decimal + integer [0, MAX)
     */
    public final static String REGEX_NON_NEGATIVE_REAL_NUMBER = "^\\+?(\\d+|\\d+\\.\\d+)$"; //$NON-NLS-1$
    public static boolean isMatch(String regex, String orginal) {
        if (orginal == null || orginal.trim().equals("")) { //$NON-NLS-1$
            return false;
        }
        Pattern pattern = Pattern.compile(regex);
        Matcher isNum = pattern.matcher(orginal);
        return isNum.matches();
    }
    /**
     * 非负整数[0,MAX)
     *
     * @param orginal
     * @return boolean
     */
    public static boolean isNumeric(String orginal) {
        return isMatch(REGEX_NUMERIC, orginal);
    }
    /**
     * 正整数[1,MAX)
     *
     * @param orginal
     * @return boolean
     * @Description: 是否为正整数
     */
    public static boolean isPositiveInteger(String orginal) {
        return isMatch(REGEX_POSITIVE_INTEGER, orginal);
    }
    /**
     * 负整数 (-MAX,-1]
     *
     * @param orginal
     * @return boolean
     */
    public static boolean isNegativeInteger(String orginal) {
        return isMatch(REGEX_NEGATIVE_INTEGER, orginal);
    }
    /**
     * 整数 (-MAX,MAX)
     *
     * @param orginal
     * @return boolean
     */
    public static boolean isInteger(String orginal) {
        return isMatch(REGEX_INTEGER, orginal);
    }
    /**
     * 正小数 (0.0, MAX)
     *
     * @param orginal
     * @return boolean
     */
    public static boolean isPositiveDecimal(String orginal) {
        return isMatch(REGEX_POSITIVE_DECIMAL, orginal);
    }
    /**
     * 负小数 (-MAX, -0.0)
     *
     * @param orginal
     * @return boolean
     */
    public static boolean isNegativeDecimal(String orginal) {
        return isMatch(REGEX_NEGATIVE_DECIMAL, orginal);
    }
    /**
     * 小数 (-MAX, MAX)
     *
     * @param orginal
     * @return boolean
     */
    public static boolean isDecimal(String orginal) {
        return isMatch(REGEX_DECIMAL, orginal);
    }
    /**
     * 实数，包括所有的整数与小数 (-MAX, MAX)
     *
     * @param orginal
     * @return boolean
     */
    public static boolean isRealNumber(String orginal) {
        return isMatch(REGEX_REAL_NUMBER, orginal);
    }
    /**
     * 非负实数 [0, MAX)
     *
     * @param orginal
     * @return boolean
     */
    public static boolean isNonNegativeRealNumber(String orginal) {
        return isMatch(REGEX_NON_NEGATIVE_REAL_NUMBER, orginal);
    }
    /**
     * 正实数
     *
     * @param orginal
     * @return boolean
     */
    public static boolean isPositiveRealNumber(String orginal) {
        return isPositiveDecimal(orginal) || isPositiveInteger(orginal);
    }
}