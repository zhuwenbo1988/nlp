package com.mycompany.myproject.controller.advice;

import org.springframework.beans.propertyeditors.StringTrimmerEditor;
import org.springframework.web.bind.WebDataBinder;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.InitBinder;

@ControllerAdvice
public class ControllerSetup {

    @InitBinder
    public void initBinder(WebDataBinder binder) {
        // 对@RequestBody这种基于消息转换器的请求参数无效
        binder.registerCustomEditor(String.class, new StringTrimmerEditor(true));
    }
}
