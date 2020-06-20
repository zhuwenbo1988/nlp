package com.mycompany.myproject.model.anno;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.mycompany.myproject.model.anno.trait.WithCode;
import com.mycompany.myproject.model.anno.trait.WithDisplay;

import java.util.stream.Stream;

public enum SampleType implements WithCode, WithDisplay {
    A(1, "类型A"),
    B(2, "类型B");

    private final int code;
    private final String display;

    SampleType(int code, String display) {
        this.code = code;
        this.display = display;
    }

    @Override
    public int code() {
        return code;
    }

    @Override
    public String display() {
        return display;
    }

    @JsonCreator
    static SampleType code2Enum(int code){
        return Stream.of(SampleType.values()).filter(state -> state.code == code).findFirst().get();
    }
}
