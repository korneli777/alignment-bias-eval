# Released data

The files in this directory are the compact results behind the paper. They are
included so the statistical analysis and figures can be checked without
downloading model weights or running a GPU sweep.

> **Content note:** These outputs measure benchmark-defined stereotypical
> associations. Category names and predictions may reflect offensive content
> from the source benchmarks, although benchmark questions and answer text are
> not redistributed in the aggregate files.

## Aggregated files

- `aggregated/logit.parquet` contains one row per model, benchmark, prompt
  condition, and metric. It includes the four paper benchmarks plus a small
  number of exploratory benchmark rows that are not used in the headline
  analysis.
- `aggregated/bbq_items.parquet` contains per-item prediction labels for the
  fixed BBQ framing sample. It records item identifiers, categories, question
  polarity, deferral, and stereotype alignment, but not the source question or
  answer text.
- `aggregated/probe.parquet` contains layer-wise probe accuracy for the eight
  probed base–instruct pairs.
- `aggregated/intervention.parquet` contains INLP/LEACE outcomes and both sanity
  gates for each intervention cell.

The CSV files under `tables/` hold the gender-direction cosines, the random-
pair cosine baseline, and the layer-wise probe values used in Section 3.3.
`probe_results/` and `intervention_results/` retain the smaller per-cell JSON
summaries from which the aggregate tables were built.
Each of these JSONs includes the recorded Python, PyTorch, Transformers,
platform, and GPU metadata for its run.

## Coverage

The paper's headline comparison uses 27 base–instruct pairs across CrowS-Pairs,
StereoSet, BBQ, and IAT. For each benchmark it compares base/plain,
instruct/plain, and instruct/chat-template scores.

The BBQ framing sample contains 6,001 items per model-condition cell. Each
applicable condition covers all 27 instruct checkpoints; the base/plain and
instruct/chat baselines cover all 27 pairs. No values are imputed.

## Provenance

The benchmark runners load CrowS-Pairs, StereoSet, BBQ, and the IAT stimulus
release from the public sources cited in `CITATIONS.bib`. The aggregate files
contain model outputs and derived statistics, not copies of the original
benchmark texts. Model identifiers and pair definitions are preserved in
`configs/models.yaml`.

To rebuild the aggregates from per-cell JSONs, run:

```bash
python scripts/aggregate.py
python scripts/analysis/build_bbq_items.py
python scripts/analysis/build_crows_effect_sizes.py
```

The CrowS-Pairs exporter writes paired outcome counts and exact per-pair
Cohen's d for both scoring conditions to
`tables/crows_pair_effect_sizes.csv`. It does not copy benchmark sentences or
model log probabilities into the released table.

`scripts/aggregate.py` includes every prompt condition found under
`data/raw_logit_scores/`, including recoverability framings. A full rerun
therefore uses the same aggregation path for baseline and framing results.

`build_bbq_items.py` requires the raw BBQ result JSONs, which are not included
in the compact release. It verifies that conditions use the same selected item
set and refuses to overwrite the aggregate when a cell is missing;
`--allow-incomplete` is available only for diagnostic work.

## License

The authors' original aggregate outputs in this directory are released under
[CC BY 4.0](../LICENSE-DATA.md). This does not relicense third-party benchmark
data, model weights, or other upstream artifacts; their original licenses and
access conditions still apply.
