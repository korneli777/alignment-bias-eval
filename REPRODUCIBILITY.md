# Reproducing the paper

This document separates verification from a full experimental rerun. The first
path is fast and uses the released results. The second path starts from public
model checkpoints and requires substantial GPU time.

## 1. Verify the released results on CPU

Use Python 3.10–3.12. The exact package versions used by the CPU workflow are
in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
make verify
```

This runs three steps:

1. `ruff check src scripts tests`
2. `pytest tests/`
3. a rebuild of paper Figures 3–5

The tests exercise the reusable statistics, aggregation, benchmark, and
regression code. The analysis scripts listed in the README separately
recompute the paper's numerical summaries from the released aggregates. None
of these CPU steps reruns a language model. CI runs the software tests on
Python 3.10, 3.11, and 3.12.

## 2. Understand the paired comparison

Each entry in `configs/models.yaml` defines one base–instruct pair. The paper
uses the following three scores:

- **Base:** base checkpoint, plain prompt (`variant=base`, `prompt_mode=raw`).
- **Instruct without template:** instruct checkpoint, the same plain prompt
  (`variant=instruct`, `prompt_mode=raw`).
- **Instruct with template:** instruct checkpoint, native chat template
  (`variant=instruct`, `prompt_mode=instruct`).

The difference between the last two alignment deltas is the estimated chat-
template contribution. The runner also permits chat-formatted diagnostics on
base checkpoints, but those cells are not used for the paper's comparison.

## 3. Rerun from model checkpoints

### Requirements

- Linux with a CUDA-capable NVIDIA GPU
- Python 3.12
- enough memory for the selected checkpoint. The full registry includes a 24B
  model and was run on 96 GB GPUs
- accepted Hugging Face licenses for the gated Llama and Gemma checkpoints
- `HF_TOKEN` exported in the environment

Install the GPU dependencies:

```bash
python -m pip install -e ".[gpu]"
```

Use Python 3.12 with PyTorch 2.11.0 (CUDA 12.8) and Transformers 5.12.1.
Scoring is a deterministic likelihood comparison in bfloat16. No text is
sampled. Every result file records the versions that produced it in its
`runtime` block.

### Stage 1: benchmark scoring

```bash
python scripts/run_benchmarks.py --no-wandb
```

This runs CrowS-Pairs, StereoSet, BBQ, and IAT for every registry entry. Use
`--family`, `--models`, `--benchmarks`, or `--limit` for a smaller diagnostic
run. `--limit` is for debugging only and must not be used for paper results.

### Stage 2: activation extraction and probing

```bash
python scripts/run_probing.py --no-wandb
```

The default probing attribute is gender. Activations are pooled at the last
token and probes use five-fold cross-validation with seed 42.

### Stage 3: concept-erasure interventions

```bash
python scripts/run_intervention.py --no-wandb
```

INLP and LEACE are applied at five normalized depths. A result is retained only
when post-projection probe accuracy is at most 0.55 and the held-out perplexity
ratio is at most 1.5.

All stages write one result file per cell and skip completed cells on restart.
Use a new `--results-root` when you need to preserve an earlier run.

## 4. Recoverability ablation

The CrowS-Pairs ablation uses all 1,508 pairs. The BBQ ablation uses the same
stratified 6,001-item set for every available condition.

```bash
python scripts/run_benchmarks.py --variant instruct \
  --benchmarks crows_pairs bbq \
  --prompt-modes jb_persona jb_roleplay jb_historical \
                 jb_refusal jb_academic jb_fluency
```

The fluency control is skipped for BBQ because it has no multiple-choice form.
Each applicable BBQ framing covers all 27 instruct checkpoints and uses the
same 6,001-item set.

## 5. Expected outputs

- `data/raw_logit_scores/`: per-model benchmark JSONs
- `data/activations/`: residual-stream activations
- `data/probe_results/`: per-model, per-layer probe summaries
- `data/intervention_projections/`: fitted INLP and LEACE projections
- `data/intervention_results/`: intervention summaries
- `data/aggregated/`: compact tables used by the analyses and figures
- `figures/`: rebuilt paper figures

The omitted raw artifacts are large: roughly 5.8 GB of per-example benchmark
JSON, 709 MB of activations, and 10 GB of projection matrices.

## 6. Release checklist

Before tagging a release:

- [ ] `make verify` passes in a clean environment.
- [ ] All three GPU entry points return useful `--help` output on a CPU install.
- [ ] The model registry still contains 27 unique, matched pairs.
- [ ] Every paper benchmark has complete base/raw, instruct/raw, and
  instruct/instruct coverage.
- [ ] Every applicable BBQ framing covers all 27 instruct checkpoints on the
  shared 6,001-item set.
- [ ] The rebuilt Figures 3–5 match the submitted figures visually.
- [ ] `CITATION.cff`, the README BibTeX, and the paper title/authors agree.
- [ ] The ACL Anthology URL, DOI, pages, and canonical citation replace the
  “to appear” citation once available.
- [ ] Code and data licenses are visible from the repository root.
- [ ] A fresh clone can complete the CPU path without private files or secrets.

## Compute accounting

The paper reports about 295 GPU-hours on Google Cloud G4 instances with NVIDIA
RTX PRO 6000 Blackwell GPUs (96 GB), measured from Weights & Biases runtime
logs. That number includes iteration and exploratory analyses. The final
three-stage rerun is estimated at roughly 200 GPU-hours. Framing jobs also used
an NVIDIA A100 40 GB. Seed 42 is used throughout. INLP iteration `k` uses
`42 + k`.
