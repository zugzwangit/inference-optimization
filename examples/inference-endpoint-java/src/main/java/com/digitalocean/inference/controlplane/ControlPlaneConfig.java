package com.digitalocean.inference.controlplane;

import com.digitalocean.inference.dto.ModelObject;

import java.util.List;

/**
 * BLACK BOX boundary to the control plane. The control plane owns the model configuration lifecycle
 * (see SYSTEM_DESIGN.md); the endpoint only needs to know which model versions are currently live and
 * at what SLO tier, so it can validate requests and advertise available models.
 */
public interface ControlPlaneConfig {

    /** @return the models currently live and servable. */
    List<ModelObject> listModels();

    /** @return true if the given model id is live and servable. */
    boolean isModelLive(String modelId);

    /**
     * @return the SLO tier configured for a model (e.g. "interactive", "batch"); used for routing and
     * timeouts. Null/unknown models are the caller's responsibility to reject.
     */
    String sloTier(String modelId);
}
