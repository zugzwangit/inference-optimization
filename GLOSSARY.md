# Glossary - Plain-Language Definitions

Every term used in [`DESIGN.md`](DESIGN.md), explained in one or two simple sentences.
Grouped so you can scan quickly. If you read only one section, read **Core Flow** and **KPIs** - they unlock the rest.

---

## Core Flow (how a model actually answers)

- **Inference** - Using an already-trained model to answer requests in production. (Training is the one-time teaching phase; inference is the everyday "use" phase.)
- **Token** - The unit of text a model reads and writes. Roughly 3/4 of a word. Models generate text one token at a time.
- **Context** - Everything the model is currently "looking at" for a request: the prompt plus the conversation so far. **Long context** means a very large prompt (e.g., a whole document), which is harder and more memory-hungry to process.
- **Prefill** - Phase 1 of answering: the model reads the entire prompt at once and produces the first output token. It is **compute-bound** (limited by raw math speed). This phase determines **TTFT**.
- **Decode** - Phase 2: the model generates the rest of the answer one token at a time, each token depending on all previous ones. It is **memory-bound** (limited by how fast data moves, not math). This phase determines **TPOT**.

## KPIs (the scoreboard - how we measure "good")

- **TTFT (Time To First Token)** - How long until the first word appears. Set mostly by the prefill phase. Lower = feels snappier.
- **TPOT (Time Per Output Token)** - Average time to produce each token after the first. Set by the decode phase. Lower = text streams out faster.
- **ITL (Inter-Token Latency)** - The gap between two consecutive tokens. Closely related to TPOT (TPOT is the average; ITL is the per-gap measurement, useful for spotting stutter).
- **Tokens/sec/dollar** - The business metric: how much useful output you get per dollar of hardware. The real goal is to make serving fast *and* cheap, not just fast.
- **SLO (Service Level Objective)** - A target you promise to hit, e.g., "TTFT under 500 ms for 99% of requests."

## Hardware

- **GPU** - The chip that does the heavy math. Our fleet has two vendors.
- **NVIDIA (H200 / B300)** - One GPU vendor. B300 ("Blackwell") is the newer generation. Programmed with **CUDA** software.
- **AMD (MI300X / MI325X / MI350X)** - The other GPU vendor. Programmed with **ROCm** software. Code written for CUDA does not automatically run on ROCm - this is a real portability cost.
- **Node** - One physical server. Holds either **1 GPU ("1x slug")** or **8 GPUs ("8x slug")**.
- **HBM (High Bandwidth Memory)** - The GPU's large main memory (~100+ GB). Big but relatively slow. Model weights and the KV cache live here.
- **SRAM** - Tiny, ultra-fast memory right next to the compute cores. The performance game in decode is doing as much as possible in SRAM and minimizing trips to slow HBM.
- **NVLink / xGMI** - Ultra-fast links connecting GPUs **inside** one node (NVLink = NVIDIA, xGMI = AMD). Great for chatty GPU-to-GPU traffic.
- **VPC networking (25 Gbps)** - The much slower network connecting **separate** nodes. Rule of thumb: keep chatty communication inside a node; crossing nodes is expensive.

## Kernels & Precision

- **Kernel** - A small, highly optimized program that runs on the GPU to do one operation (like a matrix multiply).
- **Custom kernel** - A hand-tuned kernel you write yourself to go faster than the generic, off-the-shelf one.
- **Triton / CUTLASS** - Tools/languages for writing fast NVIDIA kernels. **Composable Kernel** is the AMD/ROCm equivalent.
- **FlashAttention** - A famous custom kernel that makes the "attention" step fast and memory-efficient by being clever about SRAM vs HBM.
- **Attention** - The core mechanism by which a model relates every token to every other token to understand meaning. Gets expensive for long context.
- **Quantization** - Storing the model's numbers using fewer bits to save memory and go faster. The trade-off: too aggressive and the model gets less accurate.
- **FP8 / INT8 / FP4 / MXFP4** - Specific low-bit number formats (8-bit and experimental 4-bit). Fewer bits = smaller, faster, less memory traffic, but more accuracy risk. MXFP4 is a 4-bit format with a shared scaling factor to reduce that risk.
- **Quality floor / accuracy regression** - The minimum acceptable answer quality. An accuracy regression test checks that an optimization didn't make the model noticeably dumber.

## Model Types

- **Dense model** - The "normal" kind: the entire model runs for every token. Our long-context reasoning model is dense.
- **MoE (Mixture of Experts)** - A model built from many sub-networks called **experts**. Huge in total size (e.g., 200B+ parameters) but only a few experts run per token, so it's efficient for its size.
- **Expert** - One of the interchangeable sub-networks inside an MoE model.
- **Router** - The small part of an MoE that decides which experts handle each token. It's accuracy-sensitive, so we often keep it in higher precision.
- **Draft model** - A small, fast model used to speed up a big one (see speculative decoding).
- **Speculative decoding** - The small draft model guesses several tokens ahead; the big model verifies them in one pass. Net effect: faster output with identical quality.

## Distributing a Model Across GPUs

- **Tensor Parallelism (TP)** - Split a single layer's math across several GPUs. Very chatty, so keep it inside one node (over NVLink/xGMI).
- **Pipeline Parallelism (PP)** - Put early layers on one node and later layers on another, like an assembly line. Used to fit models too big for one node.
- **Data Parallelism (DP)** - Run multiple full copies of the model, each handling different requests. Adds throughput (more users), not size.
- **Expert Parallelism (EP)** - For MoE: place different experts on different GPUs so they run in parallel.
- **All-to-all communication** - The traffic pattern where every GPU must exchange data with every other GPU (common in MoE routing). It's sensitive to slow inter-node links, so we try to contain it within a node.

## Serving Primitives (how we run it efficiently)

- **KV cache (Key-Value cache)** - During decode, the model saves the intermediate results for tokens it already processed, so it doesn't recompute the whole conversation for each new token. It's large and lives in HBM.
- **Paged KV** - Managing the KV cache in small fixed-size "pages" (like an operating system manages RAM) to avoid wasted memory. (Popularized by vLLM.)
- **Prefix / prompt caching** - If many requests share the same beginning (e.g., a long shared system prompt), compute it once and reuse it - a big TTFT win. Must be isolated per tenant for security.
- **Continuous batching** - Continuously slotting new requests onto the GPU as old ones finish, instead of waiting to assemble a fixed-size batch. Keeps the GPU busy = more tokens/sec/dollar.
- **Disaggregated prefill/decode** - Running the compute-bound prefill phase and the memory-bound decode phase on **separate** pools of machines, each tuned for its job. Requires shipping the KV cache from a prefill machine to a decode machine.
- **Session affinity** - Routing a user's follow-up requests to the machine that already holds their cached state, to avoid recomputing it.
- **Latency compounding (agentic)** - Agents make many back-to-back model calls; small delays per call add up into a big total. Caching and session affinity fight this.
- **Request routing / tiered routing** - The router deciding *which model tier* should handle a request (cheap small model vs. expensive big model) based on estimated difficulty, to save cost on easy requests.
- **Model cascade / escalation** - Let the small model answer first; if its confidence is low, "escalate" the request to a bigger model. Saves cost on the easy majority while protecting quality on the hard cases.
- **Standalone tier** - Serving real client traffic directly from the small model (its own product tier), as opposed to using it only as a speculator.

## Infrastructure & Operations

- **Cold start** - The delay when a model must be loaded from storage into GPU memory before it can serve anything. For 100GB+ models over a network, this can be painfully slow.
- **Hierarchical caching** - Keeping model weights at the fastest available storage tier: local fast disk (NVMe) -> node RAM -> remote object store, falling back as needed.
- **Memory pooling** - Sharing/reusing memory across replicas so models can stay "warm" and ready instead of reloading.
- **Snapshot / restore** - Saving a ready-to-serve GPU memory state and quickly restoring it, instead of loading from scratch.
- **Predictive autoscaling** - Spinning machines up *before* demand arrives, using forecasts, so users don't hit a cold start.
- **Telemetry** - The metrics and traces we continuously collect to know how the system is performing.
- **Nsight / rocprof** - Low-level profilers (NVIDIA / AMD) that show exactly what the GPU is doing, microsecond by microsecond.
- **Roofline** - A simple chart that tells you whether a workload is limited by compute or by memory bandwidth - so you know what to optimize.
- **Regression gate** - An automated check in CI that blocks a change if it makes latency or accuracy worse.

## Security (a constraint that runs through everything)

- **Multi-tenant** - Many different customers share the same hardware.
- **Tenant isolation** - Hard guarantee that one customer can never see another's data or cached state.
- **No cross-tenant cache reuse** - Specifically, never reuse one tenant's prefix/KV cache for another tenant, even if the text looks identical. This protects privacy at the cost of some cache efficiency.
