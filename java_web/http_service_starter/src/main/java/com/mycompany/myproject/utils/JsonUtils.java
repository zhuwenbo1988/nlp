package com.mycompany.myproject.utils;

import com.google.gson.*;

import java.lang.reflect.Type;

public class JsonUtils {
    public static final Gson gson = new GsonBuilder().create();
    public static final Gson gsonWithUnderscore = new GsonBuilder().setFieldNamingPolicy(FieldNamingPolicy.LOWER_CASE_WITH_UNDERSCORES).create();
    /**
     * 将对象转换成json字符串。
     */
    public static String objectToJsonStr(Object data, boolean needUnderscore) {
        if (needUnderscore) {
            return gsonWithUnderscore.toJson(data);
        }
        return gson.toJson(data);
    }
    /**
     * 将对象转换成json对象。
     */
    public static JsonObject objectToJsonObj(Object data, boolean needUnderscore) {
        String str;
        if (needUnderscore) {
            str = gsonWithUnderscore.toJson(data);
        } else {
            str = gson.toJson(data);
        }
        return JsonParser.parseString(str).getAsJsonObject();
    }
    /**
     * 将json结果集转化为对象
     */
    public static <T> T jsonToPoJo(String jsonData, Class<T> beanType, boolean needUnderscore) {
        if (needUnderscore) {
            return gsonWithUnderscore.fromJson(JsonParser.parseString(jsonData), beanType);
        }
        return gson.fromJson(JsonParser.parseString(jsonData), beanType);
    }
    /**
     * 将json数据转换成pojo对象list
     */
    public static <T> T jsonToList(String jsonData, Type typeReference, boolean needUnderscore) {
        if (needUnderscore) {
            return gsonWithUnderscore.fromJson(JsonParser.parseString(jsonData), typeReference);
        }
        return gson.fromJson(JsonParser.parseString(jsonData), typeReference);
    }
}
