package com.mycompany.myproject.model.api.response.base;

import lombok.Data;

import java.io.Serializable;

@Data
public class ExceptionInfo implements Serializable {
    private static final long serialVersionUID = 8418717151745120791L;

    private int code;
    private String msg;
}
