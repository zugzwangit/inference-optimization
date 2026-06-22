package com.digitalocean.inference.dataplane;

import com.digitalocean.inference.dto.ChatCompletionChunk;
import com.digitalocean.inference.dto.ChatCompletionRequest;
import com.digitalocean.inference.dto.ChatCompletionResponse;
import com.digitalocean.inference.dto.ChatMessage;
import com.digitalocean.inference.dto.Choice;
import com.digitalocean.inference.dto.Delta;
import com.digitalocean.inference.dto.Usage;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.util.List;
import java.util.UUID;
import java.util.function.Consumer;

/**
 * Stub implementation that returns canned tokens so the endpoint structure and request/stream flow
 * can be exercised end to end without a real serving backend.
 *
 * <p>TODO: replace with a real client that calls the data plane (apply routing/cascade, continuous
 * batching, disaggregated prefill/decode, distributed KV cache) per DESIGN.md.
 */
@Component
public class StubDataPlaneClient implements DataPlaneClient {

    private static final String CANNED_REPLY =
            "This is a stubbed response from the inference endpoint template.";

    @Override
    public ChatCompletionResponse complete(String tenantId, ChatCompletionRequest request) {
        String id = newId();
        long created = Instant.now().getEpochSecond();
        ChatMessage message = new ChatMessage("assistant", CANNED_REPLY);
        Choice choice = new Choice(0, message, "stop");
        // TODO: real token counts from the data plane.
        Usage usage = Usage.of(estimatePromptTokens(request), countTokens(CANNED_REPLY));
        return new ChatCompletionResponse(
                id, ChatCompletionResponse.OBJECT_TYPE, created, request.model(), List.of(choice), usage);
    }

    @Override
    public void stream(String tenantId,
                       ChatCompletionRequest request,
                       Consumer<ChatCompletionChunk> onChunk,
                       Runnable onComplete,
                       Consumer<Throwable> onError) {
        try {
            String id = newId();
            long created = Instant.now().getEpochSecond();

            // First chunk carries the role, mirroring OpenAI streaming behavior.
            onChunk.accept(chunk(id, created, request.model(), new Delta("assistant", null), null));

            // Subsequent chunks carry content fragments. We split on spaces to emulate token streaming.
            for (String word : CANNED_REPLY.split(" ")) {
                onChunk.accept(chunk(id, created, request.model(), new Delta(null, word + " "), null));
                // TODO: real implementations stream as the decode pool produces tokens.
            }

            // Final chunk carries the finish reason and an empty delta.
            onChunk.accept(chunk(id, created, request.model(), new Delta(null, ""), "stop"));
            onComplete.run();
        } catch (RuntimeException ex) {
            onError.accept(ex);
        }
    }

    private static ChatCompletionChunk chunk(String id, long created, String model, Delta delta, String finish) {
        return new ChatCompletionChunk(
                id,
                ChatCompletionChunk.OBJECT_TYPE,
                created,
                model,
                List.of(new ChatCompletionChunk.ChunkChoice(0, delta, finish)));
    }

    private static String newId() {
        return "chatcmpl-" + UUID.randomUUID().toString().replace("-", "");
    }

    private static int estimatePromptTokens(ChatCompletionRequest request) {
        // Placeholder: the data plane reports real prompt token counts.
        return request.messages().stream()
                .map(ChatMessage::content)
                .filter(c -> c != null)
                .mapToInt(StubDataPlaneClient::countTokens)
                .sum();
    }

    private static int countTokens(String text) {
        if (text == null || text.isBlank()) {
            return 0;
        }
        return text.trim().split("\\s+").length;
    }
}
