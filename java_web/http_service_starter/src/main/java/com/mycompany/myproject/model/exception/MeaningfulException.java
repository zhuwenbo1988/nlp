package com.mycompany.myproject.model.exception;

import com.mycompany.myproject.model.api.response.base.ExceptionInfo;
import lombok.val;

/**
 * @author tony.zhuby
 */
public class MeaningfulException extends RuntimeException {
    private static final long serialVersionUID = -4793045666862414043L;

    private final int code;

    private final String msg;

    public MeaningfulException(int code, String msg) {
        this.code = code;
        this.msg = msg;
    }

    public ExceptionInfo transform() {
        val ret = new ExceptionInfo();
        ret.setCode(code);
        ret.setMsg(msg);
        return ret;
    }

    public int getCode() {
        return code;
    }

    public String getMsg() {
        return msg;
    }
}
