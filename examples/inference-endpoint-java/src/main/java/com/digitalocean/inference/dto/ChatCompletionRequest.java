package com.digitalocean.inference.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;

import java.util.List;

/**
 * Request body for {@code POST /v1/chat/completions}, mirroring the OpenAI chat completions schema.
 *
 * <p>Only the commonly used fields are modeled here; this is a structural template.
 *
 * @param model       the requested model id (validated against the control plane)
 * @param messages    the conversation so far
 * @param temperature sampling temperature (optional)
 * @param topP        nucleus sampling (optional)
 * @param maxTokens   max tokens to generate (optional)
 * @param stream      when true, the endpoint streams {@code chat.completion.chunk} SSE events
 * @param user        optional end-user identifier supplied by the caller
 */
public record ChatCompletionRequest(
        @NotNull String model,
        @NotEmpty @Valid List<ChatMessage> messages,
        Double temperature,
        @JsonProperty("top_p") Double topP,
        @JsonProperty("max_tokens") Integer maxTokens,
        Boolean stream,
        String user
) {
}
