package com.digitalocean.inference.dto;

import java.util.List;

/**
 * Response body for a non-streaming {@code POST /v1/chat/completions} call.
 *
 * @param id      unique completion id, e.g. "chatcmpl-..."
 * @param object  always "chat.completion"
 * @param created unix timestamp (seconds)
 * @param model   the model that served the request
 * @param choices the generated choices
 * @param usage   token accounting
 */
public record ChatCompletionResponse(
        String id,
        String object,
        long created,
        String model,
        List<Choice> choices,
        Usage usage
) {
    public static final String OBJECT_TYPE = "chat.completion";
}
