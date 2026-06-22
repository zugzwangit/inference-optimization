package com.digitalocean.inference.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Describes a servable model, mirroring the OpenAI {@code model} object.
 *
 * @param id      the model id used in requests
 * @param object  always "model"
 * @param created unix timestamp (seconds)
 * @param ownedBy owning organization / tenant scope
 */
public record ModelObject(
        String id,
        String object,
        long created,
        @JsonProperty("owned_by") String ownedBy
) {
    public static final String OBJECT_TYPE = "model";
}
