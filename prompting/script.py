"""Command-line entry point for CEFR descriptor prompting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Keep `python prompting/script.py` and `python -m prompting` usable.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompting.data import (check_destination, load_records, prepare_samples,
                            save_predictions, save_preview, write_json)
from prompting.scoring import extract_label, compute_metrics


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="local JSON, JSONL, or CSV records")
    source.add_argument("--dataset-id", "--dataset_id", help="Hugging Face dataset ID")
    parser.add_argument("--split", default="train", help="dataset split (original default: train)")
    parser.add_argument("--dataset-revision", help="Hugging Face dataset commit or revision")
    parser.add_argument("--prompt-template", "--prompt_template", required=True, type=Path)
    parser.add_argument("--lang-prompts", "--lang_prompts", action="store_true",
                        help="use <template stem>_<lang>.txt for each selected record")
    parser.add_argument("--model-id", "--model_id", help="supported model ID or local model path")
    parser.add_argument("--model-family", choices=("gemma3", "eurollm", "gemma", "aya", "llama", "bloomz"),
                        help="override model-family detection, useful for local paths")
    parser.add_argument("--revision", help="model/tokenizer commit or revision")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face cache directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto",
                        help="auto retains bfloat16 on CUDA and uses float32 on CPU")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="allow custom model/tokenizer code from the chosen repository")
    parser.add_argument("--local-files-only", action="store_true",
                        help="use cached model and dataset resources without network access")
    parser.add_argument("--seed", type=int, help="optional random seed; the original run had no fixed seed")
    parser.add_argument("--limit", type=positive_int, help="maximum selected records after excluding plus levels")
    parser.add_argument("--dry-run", action="store_true", help="write rendered prompts without loading a model")
    parser.add_argument("--output", type=Path, help="default: predictions.csv or prompts.jsonl in dry-run mode")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args, predictor=None):
    """Validate inputs, build prompts, predict, and keep each result with its source."""
    output = args.output or Path("prompts.jsonl" if args.dry_run else "predictions.csv")
    metric_path = Path(str(output) + ".metrics.json")
    destinations = [output] if args.dry_run else [output, metric_path]
    source_path = args.input
    if source_path is None and args.dataset_id and Path(args.dataset_id).is_file():
        source_path = Path(args.dataset_id)
    for path in destinations:
        check_destination(path, overwrite=args.overwrite)
        if source_path is not None and path.resolve() == source_path.resolve():
            raise ValueError("input and output paths must differ")
        if path.resolve() == args.prompt_template.resolve():
            raise ValueError("output must not replace the prompt template")
    if not args.dry_run and not args.model_id and predictor is None:
        raise ValueError("--model-id is required unless --dry-run is used")
    records = load_records(args.input, args.dataset_id, split=args.split,
                           revision=args.dataset_revision, cache_dir=args.cache_dir,
                           local_files_only=args.local_files_only)
    samples, skipped = prepare_samples(records, args.prompt_template,
                                       language_specific=args.lang_prompts, limit=args.limit)
    for sample in samples:
        if any(path.resolve() == sample.template_path.resolve() for path in destinations):
            raise ValueError("output must not replace a selected prompt template")
    if args.dry_run:
        save_preview(samples, output, overwrite=args.overwrite)
        print(f"Saved {len(samples)} prompts to {output}; skipped {skipped} plus-level records.")
        return None
    if predictor is None:
        from prompting.models import load_predictor

        predictor = load_predictor(args)
    results = []
    for sample in samples:
        try:
            response = predictor(sample.prompt)
            prediction = extract_label(response)
        except Exception as exc:
            detail = str(exc) or type(exc).__name__
            raise RuntimeError(f"record {sample.source_record}: generation failed: {detail}") from exc
        # Pair retained records directly with predictions after exclusions.
        results.append({"text": sample.record["text"], "lang": sample.record["lang"],
                        "format": sample.record["format"], "gold_label": sample.record["cefr_level"],
                        "prediction": prediction})
        print(f"Processed {len(results)}/{len(samples)} records", file=sys.stderr)
    metrics = compute_metrics(results)
    metrics.update(skipped_plus_levels=skipped, model_id=args.model_id,
                   model_revision=args.revision, dataset_revision=args.dataset_revision, seed=args.seed)
    save_predictions(results, output, overwrite=args.overwrite)
    write_json(metrics, metric_path, overwrite=args.overwrite)
    print(json.dumps(metrics, indent=2, allow_nan=False))
    return results


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
