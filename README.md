# Agentic Inference Cloud - Inference Optimization Architecture

A production-grade performance architecture and optimization strategy for a multi-tenant,
multi-model LLM inference endpoint running on a heterogeneous NVIDIA + AMD GPU fleet.

This repository is the deliverable for the "Kernel Optimization for AI Inference" exercise.

## What's here

| File | Purpose |
|------|---------|
| [`DESIGN.md`](DESIGN.md) | The main architecture specification (the graded deliverable). Covers kernel & precision engineering, distributed inference, resiliency, and observability, with trade-off tables and inline diagrams. |
| [`GLOSSARY.md`](GLOSSARY.md) | Plain-language definitions of every term used in the spec. Read this first if any jargon is unfamiliar. |
| [`docs/diagrams.md`](docs/diagrams.md) | All architecture diagrams (Mermaid source, renders natively on GitHub). |

## How to read it

1. Skim [`GLOSSARY.md`](GLOSSARY.md) (or the condensed key-terms box at the top of `DESIGN.md`).
2. Read [`DESIGN.md`](DESIGN.md) top to bottom - it is organized to match the four evaluation pillars.
3. Refer to [`docs/diagrams.md`](docs/diagrams.md) for the full-size system, request-lifecycle, and parallelism diagrams.

## Publishing to your personal GitHub

This repo is initialized locally with commits but is intentionally **not** pushed anywhere.
To publish it to your own account:

```bash
cd /workspaces/inference-optimization

# Option A: with the GitHub CLI (if authenticated as you)
gh repo create inference-optimization --private --source=. --remote=origin --push

# Option B: manually, after creating an empty repo in the GitHub UI
git remote add origin git@github.com:<your-username>/inference-optimization.git
git branch -M main
git push -u origin main
```

After pushing, remember to sign out of GitHub, Cursor, and the browser before handing back the workstation.
