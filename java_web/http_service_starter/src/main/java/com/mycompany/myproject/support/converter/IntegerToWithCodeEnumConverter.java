package com.mycompany.myproject.support.converter;

import com.mycompany.myproject.model.anno.trait.WithCode;
import lombok.val;
import org.springframework.core.convert.converter.Converter;

import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

public class IntegerToWithCodeEnumConverter<E extends Enum<E> & WithCode> implements Converter<Integer, E> {

    private final Map<Integer, E> map;

    public IntegerToWithCodeEnumConverter(Class<E> clazz) {
        map = Arrays.stream(Objects.requireNonNull(clazz).getEnumConstants())
                .collect(Collectors.toMap(it -> it.code(), it -> it));
    }

    @Override
    public E convert(Integer source) {
        try {
            val ret = map.get(source);
            if (ret == null) {
                throw new RuntimeException("no mapping to " + source);
            }
            return ret;
        } catch (Exception e) {
            throw new RuntimeException("source convert error", e);
        }
    }
}
