package com.mycompany.myproject.support.mapstruct;

import com.github.pagehelper.PageInfo;
import org.mapstruct.MapperConfig;
import org.mapstruct.Mapping;
import org.mapstruct.MappingInheritanceStrategy;
import org.mapstruct.NullValueCheckStrategy;

/**
 * @author tony.zhuby
 */
@MapperConfig(nullValueCheckStrategy = NullValueCheckStrategy.ALWAYS, mappingInheritanceStrategy = MappingInheritanceStrategy.AUTO_INHERIT_FROM_CONFIG)
public interface StructMapperConfig {
    @Mapping(target = "list", ignore = true)
    PageInfo<?> pageConvert(PageInfo<?> orig);
}
