package com.mycompany.myproject.model.api.response;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.mycompany.myproject.model.anno.SampleType;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDate;
import java.time.LocalTime;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class SampleVO {
    @JsonProperty("id")
    private Long id;
    @JsonProperty("sample_type")
    private SampleType sampleType;
    @JsonProperty("create_date")
    @JsonFormat(pattern = "yyyy-MM-dd", timezone = "GMT+8")
    private LocalDate createDate;
    @JsonProperty("create_time")
    @JsonFormat(pattern = "hh:mm:ss", timezone = "GMT+8")
    private LocalTime createTime;
}
