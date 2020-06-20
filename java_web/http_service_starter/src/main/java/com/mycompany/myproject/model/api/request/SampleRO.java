package com.mycompany.myproject.model.api.request;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.mycompany.myproject.model.anno.SampleType;
import lombok.Data;

import java.io.Serializable;

@Data
public class SampleRO implements Serializable {
    @JsonProperty("id")
    private Long id;
    @JsonProperty("sample_type")
    private SampleType sampleType;
}
