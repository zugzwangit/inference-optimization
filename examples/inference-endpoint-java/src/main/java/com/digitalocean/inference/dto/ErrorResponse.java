package com.digitalocean.inference.dto;

/**
 * OpenAI-style error envelope: {@code {"error": {"message", "type", "param", "code"}}}.
 */
public record ErrorResponse(ErrorBody error) {

    public record ErrorBody(
            String message,
            String type,
            String param,
            String code
    ) {
    }

    public static ErrorResponse of(String message, String type, String code) {
        return new ErrorResponse(new ErrorBody(message, type, null, code));
    }
}
