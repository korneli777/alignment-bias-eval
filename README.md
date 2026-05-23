# biaseval

A library for measuring stereotypical bias in causal language models. Implements
four logit-based benchmarks (CrowS-Pairs, StereoSet, BBQ, IAT), linear probing
on residual-stream activations, and INLP/LEACE concept-erasure interventions.

Each component works on any HuggingFace causal-LM checkpoint. Benchmarks support
both raw next-token scoring and chat-template-conditional scoring, so base and
instruction-tuned models can be compared on equal footing.

## Install

Python 3.10–3.12.

```bash
pip install -e .          # CPU: analysis, figures, tests
pip install -e .[gpu]     # add torch/transformers for scoring + probing
```

## Usage

Score one model on CrowS-Pairs:

```python
from biaseval.model_loader import ModelSpec, load_model
from biaseval.benchmarks import crows_pairs

spec = ModelSpec(model_id="meta-llama/Llama-3.1-8B-Instruct", ...)
model, tok = load_model(spec)
result = crows_pairs.run(model, tok, spec, prompt_mode="instruct")
print(result.summary["overall"])      # % of pairs where stereotype wins
```

Train a per-layer probe on any binary attribute:

```python
from biaseval.probing import datasets, linear_probe

ds = datasets.build_probe_dataset("gender")   # or pass your own labelled sentences
results = linear_probe.train_probes_all_layers(
    activation_dir, ds.labels, num_layers=32, attribute_name="gender",
)
```

Fit and apply an erasure projection:

```python
from biaseval.intervention import inlp, hooks

P = inlp.fit_inlp(X, y).projection
with hooks.ProjectionHook(model, P, layer_idx=16):
    # any forward pass on `model` is now intervened
    score_again(model, tok)
```

The driver scripts (`scripts/run_*.py`) wrap these components for batch use
across the model registry in `configs/models.yaml`, with W&B tracking and
resume-safe execution.

## Reproducing the paper

This repository accompanies the EMNLP submission *Does Alignment Debias or Just
Suppress?*, which uses the toolkit to evaluate 27 base/instruct pairs from
four open-weight families.

```bash
make figures    # rebuild paper figures from cached aggregates (~30 s, CPU)
make test       # verify every paper number against cached data (~1 min, CPU)
```

The integration test `tests/test_paper_numbers.py` recomputes every number
cited in §3 from the parquets in `data/aggregated/` and asserts it matches
the value in the paper. If anything has drifted, the test fails and prints
which claim.

Re-running the GPU pipeline from scratch (≈200 GPU-hours on one NVIDIA RTX
PRO 6000 Blackwell, 96 GB) is documented in `notebooks/01_gpu_pipeline.ipynb`.
The three stages — `run_benchmarks.py`, `run_probing.py`, `run_intervention.py`
— are independent and each resumes from cached output. Seed 42 throughout
(`configs/benchmarks.yaml`); INLP iteration `k` uses `seed + k` so each
nullspace step is reproducible in isolation.

## Layout

```
src/biaseval/        importable library
  benchmarks/        CrowS-Pairs, StereoSet, BBQ, IAT
  probing/           dataset construction, activation extraction, probes
  intervention/      INLP, LEACE, forward-hook wrapper, sanity checks
  analysis/          bootstrap, regression, statistics helpers
scripts/             entry-point CLIs and paper-specific replication scripts
configs/             27-pair model registry, benchmark config
data/                cached parquets, per-cell intervention JSONs
notebooks/           01: GPU pipeline. 02: paper walkthrough on CPU.
tests/               unit tests + paper-number integration test
```

## Data availability

The aggregated parquets, probe accuracy JSONs, and per-cell intervention
result JSONs in `data/` are sufficient to reproduce every number and figure
in the paper. Larger artifacts — raw per-example benchmark JSONs (≈5.8 GB),
residual-stream activations (≈709 MB), and INLP/LEACE projection matrices
(≈10 GB) — will be deposited on Zenodo with a DOI on de-anonymization.

## Citation

```bibtex
@inproceedings{anonymous2026alignment,
  title  = {Does Alignment Debias or Just Suppress?},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review at EMNLP 2026}
}
```

`CITATIONS.bib` lists the benchmarks, datasets, and erasure methods this
toolkit builds on; please cite them as well when using the corresponding
components.

## License

MIT (see `LICENSE`). Aggregated data in `data/` is released under CC-BY-4.0.
