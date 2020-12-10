package com.mycompany.myproject.bean;

import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;

import java.util.concurrent.ForkJoinPool;

@Configuration
@EnableAutoConfiguration
public class BeanConfig {
    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }

    /**
     * 用法
     * @Async("ThreadPool")
     */
    @Bean
    public ForkJoinPool ThreadPool() {
        return new ForkJoinPool(Runtime.getRuntime().availableProcessors());
    }
}
