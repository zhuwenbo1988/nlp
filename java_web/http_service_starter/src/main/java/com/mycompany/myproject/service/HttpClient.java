package com.mycompany.myproject.service;

import lombok.val;
import org.apache.commons.codec.digest.HmacAlgorithms;
import org.apache.commons.codec.digest.HmacUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;

import javax.annotation.Resource;
import java.util.Base64;
import java.util.List;

import static org.springframework.http.HttpHeaders.AUTHORIZATION;
import static org.springframework.http.HttpHeaders.DATE;

@Service
public class HttpClient {
    @Resource
    private RestTemplate restTemplate;

    @Value("${client_id}")
    private String clientId;
    @Value("${secret}")
    private String secret;

    private List<?> httpGet(String url) {
        HttpHeaders headers = createMwsBasicAuthHeaders(clientId, secret, HttpMethod.GET.name(), url);
        ResponseEntity<?> resp = restTemplate.exchange(url,
                HttpMethod.GET,
                new HttpEntity<MultiValueMap<String, Object>>(null, headers),
                ?.class);
        if (HttpStatus.OK.equals(resp.getStatusCode())) {
            ? body = resp.getBody();
            return body.getData();
        }
        return null;
    }

    private HttpHeaders createMwsBasicAuthHeaders(String clientId, String secret, String httpVerb, String requestUri) {
        val headers = new HttpHeaders();
        headers.setDate(System.currentTimeMillis());
        val dateString = headers.getFirst(DATE);
        val toSign = httpVerb + " " + requestUri + "\n" + dateString;
        val sign = Base64.getEncoder().encodeToString(new HmacUtils(HmacAlgorithms.HMAC_SHA_1, secret).hmac(toSign));
        headers.set(AUTHORIZATION, "MWS " + clientId + ":" + sign);
        return headers;
    }
}
