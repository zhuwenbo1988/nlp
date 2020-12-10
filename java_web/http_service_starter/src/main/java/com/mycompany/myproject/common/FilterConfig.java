package com.mycompany.myproject.common;

import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * 配置拦截器
 */
@Configuration
public class FilterConfig {
    /**
     * 拦截器注册
     *
     * @return
     */
    @Bean
    public FilterRegistrationBean corsFilterRegistration() {
        FilterRegistrationBean registration = new FilterRegistrationBean();
        registration.setFilter(new CorsFilter());
        registration.addUrlPatterns("/api/*");// 拦截路径
        // 注册处理跨域的拦截器
        registration.setName("CorsFilter");// 拦截器名称
        registration.setOrder(1);// 顺序
        return registration;
    }
}