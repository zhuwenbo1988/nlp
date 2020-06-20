package com.mycompany.myproject.support.converter;

import com.mycompany.myproject.model.anno.trait.WithCode;
import lombok.val;
import org.apache.commons.lang3.StringUtils;
import org.springframework.core.convert.converter.Converter;

import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

public class StringToWithCodeEnumConverter<E extends Enum<E> & WithCode> implements Converter<String, E> {

    private final Map<Integer, E> map;

    public StringToWithCodeEnumConverter(Class<E> clazz) {
        map = Arrays.stream(Objects.requireNonNull(clazz).getEnumConstants())
                .collect(Collectors.toMap(it -> it.code(), Function.identity()));
    }

    @Override
    public E convert(String source) {
        if (StringUtils.isBlank(source)) {
            return null;
        }
        try {
            val code = Integer.parseInt(source);
            val ret = map.get(code);
            if (ret == null) {
                throw new RuntimeException("no mapping to " + source);
            }
            return ret;
        } catch (Exception e) {
            throw new RuntimeException("source convert error", e);
        }
    }
}
