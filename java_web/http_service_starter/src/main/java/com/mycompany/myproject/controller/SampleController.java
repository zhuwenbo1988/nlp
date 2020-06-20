package com.mycompany.myproject.controller;

import com.google.common.collect.Lists;
import com.mycompany.myproject.model.anno.SampleType;
import com.mycompany.myproject.model.api.request.SampleRO;
import com.mycompany.myproject.model.api.response.SampleVO;
import com.mycompany.myproject.model.api.response.base.Result;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.time.LocalTime;
import java.util.List;

@RequestMapping("/api/")
@RestController
@Slf4j
public class SampleController {

    @GetMapping("/sample")
    public Result<List<SampleVO>> getSampleList() {
        List<SampleVO> ret = Lists.newArrayList();
        ret.add(new SampleVO((long) 1, SampleType.A, LocalDate.now(), LocalTime.now()));
        ret.add(new SampleVO((long)2, SampleType.A, LocalDate.now(), LocalTime.now()));
        ret.add(new SampleVO((long)3, SampleType.A, LocalDate.now(), LocalTime.now()));
        return Result.ok(ret);
    }

    @GetMapping("/sample/{id}")
    public Result<SampleVO> getSample(@PathVariable(value = "id") Long id) {
        SampleVO ret = new SampleVO();
        ret.setId(id);
        return Result.ok(ret);
    }

    @PostMapping("/sample")
    public Result<SampleVO> saveSample(@RequestBody SampleRO sample) {
        SampleVO ret = new SampleVO();
        ret.setId(sample.getId());
        ret.setSampleType(sample.getSampleType());
        return Result.ok(ret);
    }
}
