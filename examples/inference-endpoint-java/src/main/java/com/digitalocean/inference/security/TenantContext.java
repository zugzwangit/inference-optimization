package com.digitalocean.inference.security;

import com.digitalocean.inference.error.ApiException;

/**
 * Holds the authenticated tenant id for the duration of a request on the serving thread.
 *
 * <p>The tenant identity is established by {@link ApiKeyAuthFilter} from the authenticated
 * credential - never from user-controllable request content - and is propagated to the data plane so
 * isolation, routing, and accounting are all tenant-scoped (DESIGN.md Section 5).
 */
public final class TenantContext {

    private static final ThreadLocal<String> CURRENT = new ThreadLocal<>();

    private TenantContext() {
    }

    public static void set(String tenantId) {
        CURRENT.set(tenantId);
    }

    public static String get() {
        return CURRENT.get();
    }

    public static String requireTenant() {
        String tenant = CURRENT.get();
        if (tenant == null) {
            throw ApiException.unauthorized("Missing authenticated tenant context.");
        }
        return tenant;
    }

    public static void clear() {
        CURRENT.remove();
    }
}
