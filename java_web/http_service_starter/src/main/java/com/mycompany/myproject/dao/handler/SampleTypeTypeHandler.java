package com.mycompany.myproject.dao.handler;

import com.mycompany.myproject.dao.handler.base.BaseWithCodeEnumTypeHandler;
import com.mycompany.myproject.model.anno.SampleType;

public class SampleTypeTypeHandler extends BaseWithCodeEnumTypeHandler<SampleType> {
    public SampleTypeTypeHandler() {
        super(SampleType.class);
    }
}
