package com.digitalocean.inference.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * One choice in a non-streaming completion response.
 *
 * @param index        position of this choice
 * @param message      the assistant message
 * @param finishReason why generation stopped, e.g. "stop" or "length"
 */
public record Choice(
        int index,
        ChatMessage message,
        @JsonProperty("finish_reason") String finishReason
) {
}
