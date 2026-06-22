package com.digitalocean.inference.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * A streamed chunk for {@code POST /v1/chat/completions} with {@code stream=true}, mirroring the
 * OpenAI {@code chat.completion.chunk} schema. Each chunk is sent as one SSE {@code data:} event; the
 * stream terminates with a literal {@code data: [DONE]} event (see the controller).
 *
 * @param id      completion id, stable across the chunks of one response
 * @param object  always "chat.completion.chunk"
 * @param created unix timestamp (seconds)
 * @param model   the model serving the request
 * @param choices the incremental choices for this chunk
 */
public record ChatCompletionChunk(
        String id,
        String object,
        long created,
        String model,
        List<ChunkChoice> choices
) {
    public static final String OBJECT_TYPE = "chat.completion.chunk";

    /**
     * One choice within a streaming chunk.
     *
     * @param index        position of this choice
     * @param delta        the incremental content
     * @param finishReason set on the final chunk (e.g. "stop"), otherwise null
     */
    public record ChunkChoice(
            int index,
            Delta delta,
            @JsonProperty("finish_reason") String finishReason
    ) {
    }
}
