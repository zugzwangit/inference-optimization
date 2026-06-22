package com.digitalocean.inference.web;

import com.digitalocean.inference.controlplane.ControlPlaneConfig;
import com.digitalocean.inference.dto.ModelList;
import com.digitalocean.inference.dto.ModelObject;
import com.digitalocean.inference.error.ApiException;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * OpenAI-compatible models endpoint. The list of servable models comes from the control plane.
 */
@RestController
@RequestMapping("/v1/models")
public class ModelsController {

    private final ControlPlaneConfig controlPlane;

    public ModelsController(ControlPlaneConfig controlPlane) {
        this.controlPlane = controlPlane;
    }

    @GetMapping
    public ModelList listModels() {
        return ModelList.of(controlPlane.listModels());
    }

    @GetMapping("/{modelId}")
    public ModelObject getModel(@PathVariable String modelId) {
        return controlPlane.listModels().stream()
                .filter(m -> m.id().equals(modelId))
                .findFirst()
                .orElseThrow(() -> ApiException.modelNotFound(modelId));
    }
}
