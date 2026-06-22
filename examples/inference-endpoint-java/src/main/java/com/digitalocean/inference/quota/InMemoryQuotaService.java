package com.digitalocean.inference.quota;

import org.springframework.stereotype.Component;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Stub quota service: a trivial per-tenant fixed-window counter, purely to show where admission
 * control lives in the request path.
 *
 * <p>TODO: replace with a real distributed rate limiter (e.g. token bucket in Redis) keyed by tenant
 * and SLO tier, with quotas sourced from the control plane.
 */
@Component
public class InMemoryQuotaService implements QuotaService {

    // Demo only: requests allowed per tenant within the current process lifetime window.
    private static final int MAX_REQUESTS_PER_TENANT = 10_000;

    private final ConcurrentHashMap<String, AtomicInteger> counters = new ConcurrentHashMap<>();

    @Override
    public void check(String tenantId, String model) {
        AtomicInteger counter = counters.computeIfAbsent(tenantId, t -> new AtomicInteger());
        int count = counter.incrementAndGet();
        if (count > MAX_REQUESTS_PER_TENANT) {
            throw com.digitalocean.inference.error.ApiException.quotaExceeded(
                    "Quota exceeded for tenant '" + tenantId + "'. Try again later.");
        }
        // TODO: account by tokens and SLO tier, not just request count.
    }
}
