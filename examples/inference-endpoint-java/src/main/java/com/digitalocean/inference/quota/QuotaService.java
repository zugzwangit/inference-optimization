package com.digitalocean.inference.quota;

/**
 * Enforces per-tenant rate limits and quotas before a request is admitted to the data plane. This is
 * the endpoint's admission-control responsibility; it protects tenants from each other (noisy
 * neighbor) and the fleet from overload.
 */
public interface QuotaService {

    /**
     * Check that the tenant may issue a request for the given model right now.
     *
     * @param tenantId the authenticated tenant
     * @param model    the requested model id
     * @throws com.digitalocean.inference.error.ApiException with status 429 if the quota is exceeded
     */
    void check(String tenantId, String model);
}
