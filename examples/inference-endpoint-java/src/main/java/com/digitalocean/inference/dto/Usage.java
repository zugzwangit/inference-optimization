package com.digitalocean.inference.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Token accounting returned with a completion, mirroring the OpenAI schema.
 */
public record Usage(
        @JsonProperty("prompt_tokens") int promptTokens,
        @JsonProperty("completion_tokens") int completionTokens,
        @JsonProperty("total_tokens") int totalTokens
) {
    public static Usage of(int promptTokens, int completionTokens) {
        return new Usage(promptTokens, completionTokens, promptTokens + completionTokens);
    }
}
