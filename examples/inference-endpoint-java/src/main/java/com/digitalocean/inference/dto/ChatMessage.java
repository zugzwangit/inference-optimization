package com.digitalocean.inference.dto;

import jakarta.validation.constraints.NotBlank;

/**
 * A single message in the conversation, mirroring the OpenAI chat schema.
 *
 * @param role    one of "system", "user", "assistant", or "tool"
 * @param content the message text
 */
public record ChatMessage(
        @NotBlank String role,
        String content
) {
}
