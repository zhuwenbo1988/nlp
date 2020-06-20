package com.mycompany.myproject.dao.handler.base;

import com.mycompany.myproject.model.anno.trait.WithCode;
import org.apache.ibatis.type.BaseTypeHandler;
import org.apache.ibatis.type.JdbcType;

import java.sql.CallableStatement;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

public abstract class BaseWithCodeEnumTypeHandler<T extends Enum<T> & WithCode> extends BaseTypeHandler<T> {

    private final Map<Integer, T> mapping;

    public BaseWithCodeEnumTypeHandler(Class<T> clazz) {
        mapping = Arrays.stream(Objects.requireNonNull(clazz).getEnumConstants()).collect(Collectors.toMap(it -> it.code(), Function.identity(), (u, v) -> u));
    }

    @Override
    public void setNonNullParameter(PreparedStatement preparedStatement, int i, T t, JdbcType jdbcType) throws SQLException {
        preparedStatement.setInt(i, t.code());
    }

    @Override
    public T getNullableResult(ResultSet resultSet, String s) throws SQLException {
        int code = resultSet.getInt(s);
        return mapping.get(code);
    }

    @Override
    public T getNullableResult(ResultSet resultSet, int i) throws SQLException {
        int code = resultSet.getInt(i);
        return mapping.get(code);
    }

    @Override
    public T getNullableResult(CallableStatement callableStatement, int i) throws SQLException {
        int code = callableStatement.getInt(i);
        return mapping.get(code);
    }
}
