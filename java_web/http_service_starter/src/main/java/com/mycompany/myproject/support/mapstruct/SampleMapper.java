package com.mycompany.myproject.support.mapstruct;

import com.mycompany.myproject.model.api.response.SampleVO;
import com.mycompany.myproject.model.entity.SamplePO;
import org.mapstruct.Mapper;
import org.mapstruct.factory.Mappers;

/**
 * @author tony.zhuby
 */
@Mapper(config = StructMapperConfig.class)
public interface SampleMapper extends BaseMapper<SamplePO, SampleVO> {
    SampleMapper MAPPER = Mappers.getMapper(SampleMapper.class);

    SampleVO toVo(SamplePO source);
}
