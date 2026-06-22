# Architecture Diagrams

Full-size Mermaid diagrams for the inference optimization design. These render natively on GitHub.
A condensed subset is inlined in [`../DESIGN.md`](../DESIGN.md).

---

## 1. System Architecture

End-to-end view: ingress and routing, disaggregated prefill/decode pools, the distributed KV and
prefix caches, the hierarchical model-weight cache, and the heterogeneous GPU fleet underneath.

```mermaid
flowchart TB
    subgraph edge [Ingress and Control Plane]
        LB["API Gateway / Load Balancer"]
        Router["Scheduler / Router<br/>tenant-aware admission,<br/>continuous batching,<br/>session affinity, SLO routing"]
        PrefixCache["Prefix / Prompt Cache<br/>(per-tenant namespaced)"]
    end

    subgraph serving [Disaggregated Serving]
        PrefillPool["PREFILL Pool<br/>compute-bound<br/>optimizes TTFT"]
        DecodePool["DECODE Pool<br/>memory-bound<br/>optimizes TPOT / ITL"]
        KVStore["Distributed Paged KV Cache<br/>(tiered: HBM -> CPU RAM -> NVMe)"]
    end

    subgraph models [Model Weight Supply]
        WeightCache["Hierarchical Weight Cache<br/>NVMe -> node RAM -> object store"]
    end

    subgraph fleet [Heterogeneous GPU Fleet]
        NV["NVIDIA H200 / B300<br/>(CUDA), 1x and 8x"]
        AMD["AMD MI300X / MI325X / MI350X<br/>(ROCm), 1x and 8x"]
    end

    subgraph obs [Observability]
        Telemetry["Telemetry & Profiling<br/>TTFT/TPOT/ITL, tokens/sec/$,<br/>Nsight / rocprof, roofline"]
    end

    LB --> Router
    Router <--> PrefixCache
    Router --> PrefillPool
    PrefillPool -->|"KV handoff<br/>(intra-node / RDMA)"| DecodePool
    PrefillPool <--> KVStore
    DecodePool <--> KVStore
    WeightCache --> PrefillPool
    WeightCache --> DecodePool
    PrefillPool -.runs on.-> fleet
    DecodePool -.runs on.-> fleet
    Telemetry -.observes.-> Router
    Telemetry -.observes.-> PrefillPool
    Telemetry -.observes.-> DecodePool
```

---

## 2. Request Lifecycle - Multi-Turn Agentic Workflow

Shows where time goes (TTFT vs TPOT), how a prefix-cache hit and KV reuse short-circuit work, and
why disaggregation lets the two phases scale independently.

```mermaid
sequenceDiagram
    participant U as Agent / Client
    participant R as Router (tenant-aware)
    participant PC as Prefix Cache
    participant P as Prefill Pool
    participant D as Decode Pool
    participant KV as Distributed KV Cache

    U->>R: Turn N request (tenant T, session S)
    R->>PC: Lookup shared prefix (scoped to T)
    alt Prefix hit
        PC-->>R: Reuse cached prefix KV (skip recompute)
    else Prefix miss
        R->>P: Run prefill (compute-bound)
        P->>KV: Write prefix + prompt KV
        P-->>R: First token (defines TTFT)
    end
    R->>D: Stream decode (session affinity to warm KV)
    loop Each output token
        D->>KV: Read/append KV (memory-bound)
        D-->>U: Token (gap = ITL, avg = TPOT)
    end
    Note over R,KV: KV for session S retained for Turn N+1<br/>to avoid latency compounding
```

---

## 3. Parallelism Layout - 200B+ MoE Model

How a Mixture-of-Experts model is mapped onto hardware: Tensor + Expert Parallelism kept *inside*
a node over the fast interconnect, with Pipeline Parallelism used only to cross node boundaries.
Data Parallelism replicates the whole unit for throughput.

```mermaid
flowchart TB
    subgraph dp [Data Parallel Replicas - throughput scaling]
        direction TB
        subgraph nodeA [Node A - 8x GPU, NVLink/xGMI intra-node]
            direction LR
            A0["GPU0<br/>TP shard + Experts 0-k"]
            A1["GPU1<br/>TP shard + Experts ..."]
            A7["GPU7<br/>TP shard + Experts ..."]
            A0 <-->|"all-to-all<br/>(EP routing)"| A1
            A1 <--> A7
        end
        subgraph nodeB [Node B - 8x GPU, NVLink/xGMI intra-node]
            direction LR
            B0["GPU0<br/>later layers"]
            B7["GPU7<br/>later layers"]
            B0 <--> B7
        end
        nodeA -->|"Pipeline Parallel stage handoff<br/>(small tensors over 25Gbps VPC)"| nodeB
    end

    Note["TP + EP: intra-node only (chatty, needs NVLink/xGMI)<br/>PP: only across nodes (small, latency-tolerant transfers)<br/>DP: replicate the whole pipeline for more concurrent users"]
```

---

## 4. Cold-Start Mitigation - Hierarchical Weight Loading

The fallback path when a 100GB+ model must be made ready, fastest tier first.

```mermaid
flowchart LR
    Req["Model needed on node"] --> L1{"Warm in GPU HBM?"}
    L1 -->|yes| Serve["Serve immediately"]
    L1 -->|no| L2{"In node RAM /<br/>page cache?"}
    L2 -->|yes| LoadRAM["Stream RAM -> HBM<br/>(seconds)"] --> Serve
    L2 -->|no| L3{"On local NVMe?"}
    L3 -->|yes| LoadNVMe["mmap / lazy load<br/>NVMe -> HBM"] --> Serve
    L3 -->|no| L4["Stream from object store<br/>(slowest; pre-warm to avoid)"] --> LoadNVMe
```
