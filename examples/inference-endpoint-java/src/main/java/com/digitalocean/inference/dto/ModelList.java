package com.digitalocean.inference.dto;

import java.util.List;

/**
 * Response for {@code GET /v1/models}, mirroring the OpenAI list envelope.
 *
 * @param object always "list"
 * @param data   the available models
 */
public record ModelList(
        String object,
        List<ModelObject> data
) {
    public static ModelList of(List<ModelObject> data) {
        return new ModelList("list", data);
    }
}
