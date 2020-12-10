package com.mycompany.myproject.support.mapstruct;

import com.github.pagehelper.PageInfo;

import java.util.List;

/**
 * @author tony.zhuby
 */
public interface BaseMapper<SOURCE, TARGET> {

    List<TARGET> listCopy(List<SOURCE> orig);

    PageInfo<TARGET> pageInfoCopy(PageInfo<SOURCE> orig);
}
