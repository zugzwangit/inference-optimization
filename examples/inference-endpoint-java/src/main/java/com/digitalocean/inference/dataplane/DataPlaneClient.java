package com.digitalocean.inference.dataplane;

import com.digitalocean.inference.dto.ChatCompletionChunk;
import com.digitalocean.inference.dto.ChatCompletionRequest;
import com.digitalocean.inference.dto.ChatCompletionResponse;

import java.util.function.Consumer;

/**
 * BLACK BOX boundary to the data plane (the serving fabric: router, prefill/decode pools, KV cache,
 * GPU fleet). The endpoint never runs inference itself; it forwards an authenticated, tenant-scoped
 * request here and relays the result.
 *
 * <p>Implementations would typically call the data plane over the network (gRPC/HTTP) and apply the
 * routing/cascade and batching described in DESIGN.md. Here it is an interface so the endpoint
 * structure is independent of the serving implementation.
 */
public interface DataPlaneClient {

    /**
     * Run a non-streaming completion.
     *
     * @param tenantId the authenticated tenant (used for isolation, routing, accounting)
     * @param request  the validated request
     * @return the full completion
     */
    ChatCompletionResponse complete(String tenantId, ChatCompletionRequest request);

    /**
     * Run a streaming completion, delivering chunks as they are produced.
     *
     * @param tenantId  the authenticated tenant
     * @param request   the validated request
     * @param onChunk   invoked for each {@link ChatCompletionChunk} (the controller forwards these as
     *                  SSE events)
     * @param onComplete invoked once after the final chunk, for the controller to close the stream
     * @param onError   invoked if generation fails
     */
    void stream(String tenantId,
                ChatCompletionRequest request,
                Consumer<ChatCompletionChunk> onChunk,
                Runnable onComplete,
                Consumer<Throwable> onError);
}
