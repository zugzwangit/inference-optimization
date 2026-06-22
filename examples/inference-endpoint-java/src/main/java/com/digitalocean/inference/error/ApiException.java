package com.digitalocean.inference.error;

import org.springframework.http.HttpStatus;

/**
 * Base exception that carries enough information to render an OpenAI-style error envelope. The
 * {@code GlobalExceptionHandler} maps these to the correct HTTP status and JSON body.
 */
public class ApiException extends RuntimeException {

    private final HttpStatus status;
    private final String type;
    private final String code;

    public ApiException(HttpStatus status, String type, String code, String message) {
        super(message);
        this.status = status;
        this.type = type;
        this.code = code;
    }

    public HttpStatus status() {
        return status;
    }

    public String type() {
        return type;
    }

    public String code() {
        return code;
    }

    public static ApiException unauthorized(String message) {
        return new ApiException(HttpStatus.UNAUTHORIZED, "invalid_request_error", "invalid_api_key", message);
    }

    public static ApiException modelNotFound(String model) {
        return new ApiException(HttpStatus.NOT_FOUND, "invalid_request_error", "model_not_found",
                "The model '" + model + "' does not exist or is not available to this tenant.");
    }

    public static ApiException quotaExceeded(String message) {
        return new ApiException(HttpStatus.TOO_MANY_REQUESTS, "rate_limit_error", "rate_limit_exceeded", message);
    }
}
