package com.digitalocean.inference;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Entry point for the inference service endpoint.
 *
 * <p>This module is a structural template for the "thin front door" described in DESIGN.md
 * Section 7. The endpoint handles identity, tenancy, the API contract, and streaming; it performs no
 * inference itself and delegates to a {@code DataPlaneClient} (black box) using configuration from a
 * {@code ControlPlaneConfig} (black box). All business logic is stubbed with TODOs.
 */
@SpringBootApplication
public class InferenceEndpointApplication {

    public static void main(String[] args) {
        SpringApplication.run(InferenceEndpointApplication.class, args);
    }
}
