# Inference Endpoint - Java Template

A Spring Boot skeleton showing the **structure** of the inference service endpoint described in
[`../../DESIGN.md`](../../DESIGN.md) Section 7. It is the "thin front door": it handles identity,
tenancy, the OpenAI-compatible API contract, and token streaming, and delegates everything else to a
data plane (black box) using configuration from a control plane (black box).

This is illustrative structure, not a working inference engine - the serving logic is stubbed and
returns canned tokens so the request/stream flow can be run end to end.

## Run it

Requires JDK 17+ and Maven.

```bash
cd examples/inference-endpoint-java
mvn spring-boot:run
```

The service listens on `http://localhost:8080`.

## Try it

Demo API keys map to demo tenants (see `ApiKeyAuthFilter`): `sk-demo-tenant-a` -> `tenant-a`,
`sk-demo-tenant-b` -> `tenant-b`.

List models:

```bash
curl http://localhost:8080/v1/models \
  -H "Authorization: Bearer sk-demo-tenant-a"
```

Non-streaming chat completion:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-tenant-a" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "small-fast",
        "messages": [{"role": "user", "content": "Hello!"}]
      }'
```

Streaming (Server-Sent Events, ends with `data: [DONE]`):

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-tenant-a" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "small-fast",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": true
      }'
```

Missing/invalid key returns a `401` with an OpenAI-style error envelope.

## How the packages map to the design

| Package | Role | DESIGN.md link |
|---|---|---|
| `web` | API surface: `/v1/chat/completions` (sync + SSE), `/v1/models`, error envelope | Section 7 (endpoint = contract + streaming) |
| `dto` | OpenAI-compatible request/response/chunk/model/error schemas | Section 7 (API contract) |
| `security` | API-key auth + per-request tenant identity (`TenantContext`) | Section 5 (tenant isolation), Section 7 (identity) |
| `quota` | Per-tenant admission control (rate limit / quota) | Section 7 (quotas), Section 5 (noisy-neighbor) |
| `dataplane` | BLACK BOX boundary to the serving fabric | Sections 3-4 (serving) |
| `controlplane` | BLACK BOX boundary for live model versions / SLO tiers | [`../../SYSTEM_DESIGN.md`](../../SYSTEM_DESIGN.md) |

## What is stubbed (and where the real work would go)

- **`StubDataPlaneClient`** - returns canned tokens. Real impl calls the data plane and applies
  routing/cascade, continuous batching, disaggregated prefill/decode, and the distributed KV cache.
- **`StubControlPlaneConfig`** - a fixed three-model registry. Real impl tracks live model versions
  and SLO tiers from the control plane.
- **`ApiKeyAuthFilter`** - in-memory demo keys. Real impl uses a credential store (hashed keys,
  rotation, scopes).
- **`InMemoryQuotaService`** - a trivial counter. Real impl is a distributed, token-and-tier-aware
  rate limiter.

Each stub is annotated with `TODO` comments pointing at the production responsibility.
