package com.digitalocean.inference.web;

import com.digitalocean.inference.controlplane.ControlPlaneConfig;
import com.digitalocean.inference.dataplane.DataPlaneClient;
import com.digitalocean.inference.dto.ChatCompletionChunk;
import com.digitalocean.inference.dto.ChatCompletionRequest;
import com.digitalocean.inference.error.ApiException;
import com.digitalocean.inference.quota.QuotaService;
import com.digitalocean.inference.security.TenantContext;
import jakarta.annotation.PreDestroy;
import jakarta.validation.Valid;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * OpenAI-compatible chat completions endpoint.
 *
 * <p>This controller embodies DESIGN.md Section 7: it does identity + contract + streaming only. For
 * each request it (1) requires an authenticated tenant, (2) enforces quota, (3) validates the model
 * against the control plane, then (4) delegates to the data plane and relays the result. It contains
 * no inference logic.
 */
@RestController
@RequestMapping("/v1")
public class ChatCompletionsController {

    /** Sentinel that terminates an OpenAI-style SSE stream. */
    private static final String DONE = "[DONE]";

    private final DataPlaneClient dataPlane;
    private final ControlPlaneConfig controlPlane;
    private final QuotaService quota;

    // Streaming runs off the request thread so the SseEmitter is returned to the container promptly.
    // TODO: replace with a managed task executor sized for expected concurrency.
    private final ExecutorService streamExecutor = Executors.newCachedThreadPool();

    public ChatCompletionsController(DataPlaneClient dataPlane,
                                     ControlPlaneConfig controlPlane,
                                     QuotaService quota) {
        this.dataPlane = dataPlane;
        this.controlPlane = controlPlane;
        this.quota = quota;
    }

    @PostMapping(value = "/chat/completions", produces = {
            MediaType.APPLICATION_JSON_VALUE, MediaType.TEXT_EVENT_STREAM_VALUE})
    public Object createChatCompletion(@Valid @RequestBody ChatCompletionRequest request) {
        String tenantId = TenantContext.requireTenant();
        quota.check(tenantId, request.model());
        if (!controlPlane.isModelLive(request.model())) {
            throw ApiException.modelNotFound(request.model());
        }

        if (Boolean.TRUE.equals(request.stream())) {
            return streamCompletion(tenantId, request);
        }
        return dataPlane.complete(tenantId, request);
    }

    private SseEmitter streamCompletion(String tenantId, ChatCompletionRequest request) {
        SseEmitter emitter = new SseEmitter(0L); // no timeout for the template; configure in production
        streamExecutor.execute(() -> dataPlane.stream(
                tenantId,
                request,
                chunk -> sendChunk(emitter, chunk),
                () -> completeStream(emitter),
                emitter::completeWithError));
        return emitter;
    }

    private void sendChunk(SseEmitter emitter, ChatCompletionChunk chunk) {
        try {
            emitter.send(SseEmitter.event().data(chunk, MediaType.APPLICATION_JSON));
        } catch (IOException e) {
            // Surfaced to the data-plane onError callback, which fails the emitter.
            throw new UncheckedIOException(e);
        }
    }

    private void completeStream(SseEmitter emitter) {
        try {
            emitter.send(SseEmitter.event().data(DONE));
            emitter.complete();
        } catch (IOException e) {
            emitter.completeWithError(e);
        }
    }

    @PreDestroy
    void shutdown() {
        streamExecutor.shutdown();
    }
}
