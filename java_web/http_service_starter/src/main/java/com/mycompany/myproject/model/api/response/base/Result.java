package com.mycompany.myproject.model.api.response.base;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.lang.Nullable;

import java.io.Serializable;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Result<T> implements Serializable {
    private static final long serialVersionUID = -2295515565311212372L;

    private int status;
    private String msg;
    private T data;

    public static <T> Result<T> ok() {
        return ok(null);
    }

    public static <T> Result<T> ok(@Nullable T data) {
        return new Result<>(HttpStatus.OK.value(), HttpStatus.OK.getReasonPhrase(), data);
    }

    public static Result<ExceptionInfo> internalException(Exception ex) {
        ExceptionInfo info = new ExceptionInfo();
        info.setCode(1000);
        info.setMsg(ex.getLocalizedMessage());
        return new Result<>(HttpStatus.INTERNAL_SERVER_ERROR.value(), ex.getLocalizedMessage(), info);
    }
}
