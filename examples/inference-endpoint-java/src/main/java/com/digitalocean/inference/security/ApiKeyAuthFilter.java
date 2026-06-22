package com.digitalocean.inference.security;

import com.digitalocean.inference.dto.ErrorResponse;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.core.annotation.Order;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.Map;

/**
 * Authenticates each {@code /v1/**} request by its bearer API key and resolves the owning tenant,
 * which it stores in {@link TenantContext} for the duration of the request.
 *
 * <p>This is the endpoint's "identity" responsibility from DESIGN.md Section 7. The tenant is derived
 * solely from the authenticated key, never from request content.
 *
 * <p>TODO: replace the in-memory demo key map with a real credential store (hashed keys, rotation,
 * scopes) and richer principal/tenant resolution.
 */
@Component
@Order(1)
public class ApiKeyAuthFilter extends OncePerRequestFilter {

    private static final String BEARER_PREFIX = "Bearer ";

    // Demo keys only. Do not use in production.
    private static final Map<String, String> DEMO_KEYS = Map.of(
            "sk-demo-tenant-a", "tenant-a",
            "sk-demo-tenant-b", "tenant-b"
    );

    private final ObjectMapper objectMapper;

    public ApiKeyAuthFilter(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        return !request.getRequestURI().startsWith("/v1/");
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                    HttpServletResponse response,
                                    FilterChain filterChain) throws ServletException, IOException {
        String header = request.getHeader(HttpHeaders.AUTHORIZATION);
        String tenant = resolveTenant(header);
        if (tenant == null) {
            writeUnauthorized(response);
            return;
        }
        try {
            TenantContext.set(tenant);
            filterChain.doFilter(request, response);
        } finally {
            TenantContext.clear();
        }
    }

    private String resolveTenant(String authorizationHeader) {
        if (authorizationHeader == null || !authorizationHeader.startsWith(BEARER_PREFIX)) {
            return null;
        }
        String key = authorizationHeader.substring(BEARER_PREFIX.length()).trim();
        return DEMO_KEYS.get(key);
    }

    private void writeUnauthorized(HttpServletResponse response) throws IOException {
        response.setStatus(HttpStatus.UNAUTHORIZED.value());
        response.setContentType(MediaType.APPLICATION_JSON_VALUE);
        ErrorResponse body = ErrorResponse.of(
                "Invalid or missing API key. Provide 'Authorization: Bearer <key>'.",
                "invalid_request_error",
                "invalid_api_key");
        objectMapper.writeValue(response.getWriter(), body);
    }
}
