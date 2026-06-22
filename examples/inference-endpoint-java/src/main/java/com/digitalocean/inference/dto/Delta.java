package com.digitalocean.inference.dto;

/**
 * The incremental content carried by a streaming chunk choice.
 *
 * <p>The first chunk typically carries the {@code role}; subsequent chunks carry {@code content}
 * fragments.
 */
public record Delta(
        String role,
        String content
) {
}
