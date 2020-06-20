package com.mycompany.myproject.common;

import com.mycompany.myproject.model.anno.SampleType;
import com.mycompany.myproject.support.converter.IntegerToWithCodeEnumConverter;
import com.mycompany.myproject.support.converter.StringToWithCodeEnumConverter;
import org.springframework.core.convert.support.GenericConversionService;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;

@Component
public class WebViewConfig {
    private final GenericConversionService service;

    public WebViewConfig(GenericConversionService service) {
        this.service = service;
    }

    @PostConstruct
    public void custom() {
        service.addConverter(String.class, SampleType.class, new StringToWithCodeEnumConverter<>(SampleType.class));
        service.addConverter(Integer.class, SampleType.class, new IntegerToWithCodeEnumConverter<>(SampleType.class));
    }
}
