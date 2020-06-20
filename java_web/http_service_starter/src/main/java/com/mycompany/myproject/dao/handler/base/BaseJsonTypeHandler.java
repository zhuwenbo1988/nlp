package com.mycompany.myproject.dao.handler.base;

import com.mycompany.myproject.utils.JsonUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.ibatis.type.BaseTypeHandler;
import org.apache.ibatis.type.JdbcType;

import java.sql.CallableStatement;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

/**
 * POJO里有个属性是非基本数据类型，在DB存储时我们想存的是json格式的字符串，从DB拿出来时想直接映射成目标类型，也即json格式的字符串字段与Java类的相互类型转换。
 */
public abstract class BaseJsonTypeHandler<T extends Object> extends BaseTypeHandler<T> {
    private Class<T> clazz;
 
    public BaseJsonTypeHandler(Class<T> clazz) {
        if (clazz == null) throw new IllegalArgumentException("Type argument cannot be null");
        this.clazz = clazz;
    }
 
    @Override
    public void setNonNullParameter(PreparedStatement ps, int i, T parameter, JdbcType jdbcType) throws SQLException {
        ps.setString(i, this.toJson(parameter));
    }
 
    @Override
    public T getNullableResult(ResultSet rs, String columnName) throws SQLException {
        return this.toObject(rs.getString(columnName), clazz);
    }
 
    @Override
    public T getNullableResult(ResultSet rs, int columnIndex) throws SQLException {
        return this.toObject(rs.getString(columnIndex), clazz);
    }
 
    @Override
    public T getNullableResult(CallableStatement cs, int columnIndex) throws SQLException {
        return this.toObject(cs.getString(columnIndex), clazz);
    }
 
    private String toJson(T object) {
        try {
            return JsonUtils.objectToJsonStr(object, false);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
 
    private T toObject(String content, Class<?> clazz) {
        if (!StringUtils.isBlank(content)) {
            try {
                return (T) JsonUtils.jsonToPoJo(content, clazz, false);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        } else {
            return null;
        }
    }
}