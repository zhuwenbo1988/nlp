package com.mycompany.myproject;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.context.annotation.ImportResource;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.transaction.annotation.EnableTransactionManagement;

@SpringBootApplication
@ImportResource("classpath:applicationContext.xml")
@EnableCaching
@EnableAsync
@EnableTransactionManagement
public class HttpApp {
    public static void main(String[] args) {
        SpringApplication.run(HttpApp.class, args);
    }
}