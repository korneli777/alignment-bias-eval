"""Stage 1 — score every model on each bias benchmark.

Iterates over the registry × benchmarks. Each model is loaded once, all
enabled benchmarks run, then unloaded. Per-cell JSONs go to
`data/raw_logit_scores/{benchmark}/{model_short_name}__{prompt_mode}.json`.

Resume-safe: cells whose output JSON already exists are skipped, so an
interrupted run can be restarted with the same command.

Usage:
    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --family llama
    python scripts/run_benchmarks.py --models meta-llama/Llama-2-7b-hf
    python scripts/run_benchmarks.py --smoke      # tiny model, --limit 8
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

# Heavy imports (torch, transformers, biaseval.benchmarks) are deferred to
# main() so that `--help` works without a GPU install.

PROMPT_MODES = ("raw", "instruct", "jailbreak")

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",       default="configs/models.yaml")
    p.add_argument("--bench-config", default="configs/benchmarks.yaml")
    p.add_argument("--results-root", default="data")
    p.add_argument("--family",  default=None,
                   help="Limit to one family (llama, qwen, mistral, gemma).")
    p.add_argument("--models",  nargs="+", default=None,
                   help="Limit to specific HF model IDs (overrides --family).")
    p.add_argument("--variant", default=None, choices=["base", "instruct"])
    p.add_argument("--benchmarks", nargs="+", default=None,
                   help="Subset of benchmarks (default: all enabled in benchmarks.yaml).")
    p.add_argument("--prompt-modes", nargs="+", default=list(PROMPT_MODES),
                   choices=list(PROMPT_MODES),
                   help="Prompt conditions to evaluate.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap dataset size per benchmark (debugging only).")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--smoke",    action="store_true",
                   help="Run only the smoke_test_model with --limit 8.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Heavy imports happen here so `--help` works on a CPU-only install.
    import torch
    import yaml
    from dotenv import load_dotenv
    from transformers import set_seed

    from biaseval.benchmarks import bbq, crows_pairs, iat, stereoset
    from biaseval.io import (
        is_completed,
        logit_result_path,
        write_benchmark_result,
    )
    from biaseval.model_loader import ModelSpec, load_model, unload_model
    from biaseval.registry import filter_specs, load_registry
    from biaseval.tracking import init_run

    runners = {
        "crows_pairs": crows_pairs.run,
        "stereoset":   stereoset.run,
        "bbq":         bbq.run,
        "iat":         iat.run,
    }

    load_dotenv()
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.bench_config) as f:
        bench_cfg = yaml.safe_load(f)
    enabled = [b for b, c in bench_cfg["benchmarks"].items() if c.get("enabled", True)]
    if args.benchmarks:
        enabled = [b for b in enabled if b in args.benchmarks]
    logger.info("Enabled benchmarks: %s", enabled)

    if args.smoke:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        model_id = cfg.get("smoke_test_model", "sshleifer/tiny-gpt2")
        specs = [ModelSpec(
            model_id=model_id, family="smoke", generation="smoke", size="tiny",
            variant="base", num_params=1, num_layers=2, hidden_size=64,
            dtype="float32", model_class="AutoModelForCausalLM",
        )]
        limit = args.limit or 8
    else:
        specs = list(filter_specs(
            load_registry(args.config),
            family=args.family, variant=args.variant,
            only_ids=set(args.models) if args.models else None,
        ))
        limit = args.limit
    logger.info("Will evaluate %d models × %d benchmarks", len(specs), len(enabled))

    results_root = Path(args.results_root)
    prompt_modes: list[str] = args.prompt_modes
    n_done, n_skipped, n_errors = 0, 0, 0

    for spec in specs:
        # Jailbreak only makes sense on instruct variants — base models have
        # no safety filter to bypass and may lack a chat template.
        all_cells = [(b, pm) for b in enabled for pm in prompt_modes
                     if not (pm == "jailbreak" and spec.variant != "instruct")]
        pending = [(b, pm) for b, pm in all_cells
                   if not is_completed(logit_result_path(results_root, b, spec, pm))]
        if not pending:
            logger.info("[skip] %s — all cells done", spec.model_id)
            n_skipped += len(all_cells)
            continue

        logger.info("[load] %s (pending: %d cells)", spec.model_id, len(pending))
        try:
            model, tokenizer = load_model(spec)
        except Exception:
            logger.error("[error] failed to load %s\n%s",
                         spec.model_id, traceback.format_exc())
            n_errors += len(pending)
            continue

        for bench_name, pm in pending:
            run_ctx = init_run(
                name=f"{bench_name}/{pm}/{spec.short_name}",
                job_type=bench_name,
                tags=[spec.family, spec.variant, bench_name, f"prompt:{pm}"],
                config={
                    "model_id": spec.model_id, "family": spec.family,
                    "variant": spec.variant, "size": spec.size,
                    "benchmark": bench_name, "prompt_mode": pm,
                    "seed": args.seed, "limit": limit,
                },
                enabled=not args.no_wandb,
            )
            with run_ctx as tracker:
                try:
                    runner = runners[bench_name]
                    kwargs: dict = {"prompt_mode": pm}
                    if limit is not None and bench_name != "iat":
                        kwargs["limit"] = limit
                    result = runner(model, tokenizer, spec, **kwargs)
                    write_benchmark_result(results_root, result, spec)
                    tracker.summary_update(result.summary)
                    n_done += 1
                except Exception:
                    logger.error("[error] %s/%s on %s\n%s",
                                 bench_name, pm, spec.model_id, traceback.format_exc())
                    n_errors += 1

        unload_model(model)
        del tokenizer

    logger.info("Done. completed=%d skipped=%d errors=%d", n_done, n_skipped, n_errors)
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
