package com.mycompany.myproject.utils;

import org.joda.time.DateTime;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;

import java.util.Date;

public class DateTimeFormatUtils {
    public static final String STANDARD_FORMAT = "yyyy-MM-dd HH:mm:ss";
    private static DateTimeFormatter dateTimeFormatter;

    static {
        dateTimeFormatter = DateTimeFormat.forPattern(STANDARD_FORMAT);
    }

    public static Date strToDate(String str){
        DateTime dateTime = dateTimeFormatter.parseDateTime(str);
        return dateTime.toDate();
    }

    public static String dateToStr(Date date) {
        DateTime dateTime = new DateTime(date);
        return dateTime.toString(STANDARD_FORMAT);
    }

}
