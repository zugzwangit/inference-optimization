package com.digitalocean.inference.controlplane;

import com.digitalocean.inference.dto.ModelObject;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.util.List;
import java.util.Map;

/**
 * Stub control-plane config with a small canned model registry, illustrating the three workload tiers
 * from DESIGN.md (giant MoE, dense long-context, small model).
 *
 * <p>TODO: replace with a client that polls/subscribes to the real control plane for live model
 * versions and SLO tiers.
 */
@Component
public class StubControlPlaneConfig implements ControlPlaneConfig {

    private static final long CREATED = Instant.parse("2024-01-01T00:00:00Z").getEpochSecond();

    // model id -> SLO tier
    private static final Map<String, String> MODELS = Map.of(
            "moe-200b", "interactive",
            "dense-longctx", "interactive",
            "small-fast", "interactive"
    );

    @Override
    public List<ModelObject> listModels() {
        return MODELS.keySet().stream()
                .sorted()
                .map(id -> new ModelObject(id, ModelObject.OBJECT_TYPE, CREATED, "digitalocean"))
                .toList();
    }

    @Override
    public boolean isModelLive(String modelId) {
        return modelId != null && MODELS.containsKey(modelId);
    }

    @Override
    public String sloTier(String modelId) {
        return MODELS.get(modelId);
    }
}
