package com.mycompany.myproject.utils;

public class StringValidationUtils {
    public final static String REGEX_CN_NAME = "^[a-zA-Z\\u4e00-\\u9fa5][_a-zA-Z\\d\\u4e00-\\u9fa5]+$";


    public final static String REGEX_EN_NAME = "^[a-zA-Z][_a-zA-Z\\d]+$";

    public static boolean isValidCnName(String name) {
        return NumberValidationUtils.isMatch(REGEX_CN_NAME, name);
    }

    public static boolean isValidEnName(String name) {
        return NumberValidationUtils.isMatch(REGEX_EN_NAME, name);
    }

}
