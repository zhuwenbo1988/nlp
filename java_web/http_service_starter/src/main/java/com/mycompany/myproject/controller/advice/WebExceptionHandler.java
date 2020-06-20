package com.mycompany.myproject.controller.advice;

import com.mycompany.myproject.model.api.response.base.Result;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
@Slf4j
public class WebExceptionHandler {

    @ExceptionHandler(Exception.class)
    public Result<?> common(Exception ex) {
        log.error("exception handler record", ex);
        return Result.internalException(ex);
    }
}
