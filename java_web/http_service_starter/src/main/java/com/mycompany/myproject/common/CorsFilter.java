package com.mycompany.myproject.common;

import com.google.common.collect.Sets;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.filter.OncePerRequestFilter;

import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.Set;

@Slf4j
public class CorsFilter extends OncePerRequestFilter {
    public static final String ACCESS_CONTROL_ALLOW_ORIGIN = "Access-Control-Allow-Origin";
    public static final String ACCESS_CONTROL_ALLOW_CREDENTIALS = "Access-Control-Allow-Credentials";
    public static final String ACCESS_CONTROL_ALLOW_METHODS = "Access-Control-Allow-Methods";
    public static final String ACCESS_CONTROL_REQUEST_METHOD = "Access-Control-Request-Method";
    public static final String ACCESS_CONTROL_ALLOW_HEADERS = "Access-Control-Allow-Headers";
    public static final String ORIGIN = "Origin";
    public static final String OPTIONS = "OPTIONS";
    public static final String METHODS = "GET, POST, PUT, DELETE, OPTIONS";
    public static final String HEADERS = "X-Requested-With,Origin,Content-Type, Accept, swimlane,Access-token,client-id";
    public static final String MSG = "请求来源不合法";
    public static final Set<String> allowOrigins = Sets.newHashSet("");

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain) throws IOException, ServletException {
        String origin = request.getHeader(ORIGIN);
        if (null == origin || !allowOrigins.contains(origin)) {
            log.warn("{}[{}]", MSG, origin);
            filterChain.doFilter(request, response);
            return;
        }
        response.addHeader(ACCESS_CONTROL_ALLOW_ORIGIN, origin);
        response.addHeader(ACCESS_CONTROL_ALLOW_CREDENTIALS, Boolean.TRUE.toString());
        response.addHeader(ACCESS_CONTROL_ALLOW_METHODS, METHODS);
        if (request.getHeader(ACCESS_CONTROL_REQUEST_METHOD) != null && OPTIONS.equals(request.getMethod())) {
            // CORS "pre-flight" request
            response.addHeader(ACCESS_CONTROL_ALLOW_HEADERS, HEADERS);
        }
        filterChain.doFilter(request, response);
    }
}