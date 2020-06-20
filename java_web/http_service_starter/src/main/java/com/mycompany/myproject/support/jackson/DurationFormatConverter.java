package com.mycompany.myproject.support.jackson;

import com.fasterxml.jackson.databind.util.StdConverter;
import lombok.val;
import org.springframework.stereotype.Component;

import java.time.Duration;

@Component
public class DurationFormatConverter extends StdConverter<Number, String> {
    @Override
    public String convert(Number value) {
        val duration = Duration.ofMillis(value.longValue());
        if (duration == Duration.ZERO) {
            return "0";
        }
        val sb = new StringBuilder();
        val days = duration.toDays();
        if (days > 0) {
            sb.append(days).append("天");
        }
        val hours = duration.toHours() % 24;
        if (hours > 0) {
            sb.append(hours).append("小时");
        }
        val minutes = duration.toMinutes() % 60;
        if (minutes > 0) {
            sb.append(minutes).append("分");
        }
        val seconds = duration.getSeconds() % 60;
        if (seconds > 0) {
            sb.append(seconds).append("秒");
        }
        if (sb.length() == 0) {
            sb.append("0");
        }
        return sb.toString();
    }
}
