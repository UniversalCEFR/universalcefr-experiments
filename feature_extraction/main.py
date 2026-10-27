"""Extract UniversalCEFR features from JSON, JSONL, or CSV records."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
from importlib import metadata as package_metadata
import json
import math
from numbers import Real
import os
from pathlib import Path
import sys
import tempfile

from . import constants


FORMATS = ("sentence-level", "paragraph-level", "document-level", "dialogue-level")
PROCESSORS = "tokenize,pos,lemma,depparse"


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON, JSONL, or CSV input")
    parser.add_argument("--output", required=True, type=Path, help="CSV output")
    parser.add_argument("--languages", nargs="+", choices=tuple(constants.lang_dict),
                        help="extract only these languages (default: all input languages)")
    parser.add_argument("--limit", type=positive_int, help="maximum number of selected rows")
    parser.add_argument("--feature-set", choices=("paper", "all"), default="paper",
                        help="paper: 100 classifier features; all: 146 extracted features (default: paper)")
    parser.add_argument("--model-dir", type=Path, default=Path.home() / "stanza_resources",
                        help="Stanza model directory (default: ~/stanza_resources)")
    parser.add_argument("--download-models", action="store_true",
                        help="download the required Stanza models before extraction")
    parser.add_argument("--use-gpu", action="store_true", help="use a CUDA GPU (default: CPU)")
    parser.add_argument("--megahr-dir", type=Path, help="directory containing megahr.<lang> lexicons")
    parser.add_argument("--fasttext-dir", type=Path, help="directory containing cc.<lang>.300.vec files")
    parser.add_argument("--skip-imageability", action="store_true",
                        help="omit imageability and concreteness features")
    parser.add_argument("--skip-tseval", action="store_true", help="omit all tseval features")
    parser.add_argument("--skip-frequency", action="store_true",
                        help="omit the three tseval features requiring fastText vocabulary ranks")
    parser.add_argument("--skip-syllables", action="store_true",
                        help="omit three syllable features and two readability scores")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    return parser


def read_data(file_path):
    """Read records while retaining JSON metadata and CSV strings."""
    path = Path(file_path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() == ".csv":
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError("input CSV has no header")
            if len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError("input CSV has duplicate column names")
            records = list(reader)
            for number, row in enumerate(records, 1):
                if None in row:
                    raise ValueError(f"row {number}: more CSV fields than header columns")
        elif path.suffix.lower() in (".jsonl", ".ndjson"):
            records = []
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL line {number}: {exc.msg}") from exc
        elif path.suffix.lower() == ".json":
            records = json.load(stream)
            if not isinstance(records, list):
                raise ValueError("input JSON must contain an array of record objects")
        else:
            raise ValueError("input extension must be .json, .jsonl, .ndjson, or .csv")
    if not records:
        raise ValueError("input contains no records")
    return records


def validate_records(records):
    """Validate every row before resource loading; return the original records."""
    for number, row in enumerate(records, 1):
        prefix = f"row {number}"
        if not isinstance(row, dict):
            raise ValueError(f"{prefix}: expected a record object")
        if any(not isinstance(key, str) for key in row):
            raise ValueError(f"{prefix}: all column names must be strings")
        missing = {"text", "lang", "format"} - row.keys()
        if missing:
            raise ValueError(f"{prefix}: missing fields {', '.join(sorted(missing))}")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError(f"{prefix}: text must be a nonempty string")
        if not isinstance(row["lang"], str) or row["lang"] not in constants.lang_dict:
            raise ValueError(f"{prefix}: unsupported language {row['lang']!r}")
        if row["format"] not in FORMATS:
            raise ValueError(f"{prefix}: unsupported format {row['format']!r}")
        collisions = [key for key in row if key.startswith("F_")]
        if collisions:
            raise ValueError(f"{prefix}: input contains reserved feature columns {', '.join(collisions)}")
    return records


def load_language_specific_data(lang, model_name, *, model_dir, use_gpu=False,
                                download_models=False):
    """Load one language pipeline; network access requires download_models=True."""
    import spacy_stanza
    import stanza

    if use_gpu:
        import torch

        if not torch.cuda.is_available():
            raise ValueError("--use-gpu requires an available CUDA GPU; omit it to use CPU")
    if download_models:
        stanza.download(lang, model_dir=str(model_dir), package=model_name,
                        processors=PROCESSORS, verbose=False)
    # A string remains serializable through spaCy's config. Stanza resolves
    # "none" to DownloadMethod.NONE, including local-only transformer loading.
    return spacy_stanza.load_pipeline(
        "xx" if lang == "cy" else lang,
        lang=lang, dir=str(model_dir), package=model_name, processors=PROCESSORS,
        use_gpu=use_gpu, download_method="none", verbose=False,
    )


def _validate_features(features):
    for name, value in features.items():
        if not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"feature {name} has a non-finite or non-numeric value: {value!r}")
    return features


def parse_sentence(sent, imagebility_data, lang, *, skip_imageability=False,
                   skip_tseval=False, skip_frequency=False, skip_syllables=False):
    """Retain the original sentence-level definitions and F_* column names."""
    from . import feature_extraction as fe

    if len(sent) == 0:
        raise ValueError("the parser returned no tokens")
    features = {}
    for function in (fe.feature_morphology, fe.feature_dependency_tree_distance,
                     fe.features_verb_noun_ratio, fe.feature_negations):
        features.update(function(sent))
    if not skip_imageability:
        features.update(fe.features_imageability_familiartiy(sent, imagebility_data))
    if not skip_tseval:
        features.update(fe.features_tseval(sent, lang, skip_frequency=skip_frequency,
                                          skip_syllables=skip_syllables))
    features["F_doc_num_sents"] = len(list(sent.sents))
    features["F_doc_num_tokens"] = len(sent)
    return _validate_features(features)


def parse_document(doc, imagebility_data, lang, **options):
    """Average sentence features equally and retain whole-document counts."""
    sentences = list(doc.sents)
    if not sentences or len(doc) == 0:
        raise ValueError("the parser returned no sentences or tokens")
    rows = [parse_sentence(sent, imagebility_data, lang, **options) for sent in sentences]
    features = {key: round(sum(row[key] for row in rows) / len(rows), 6) for key in rows[0]}
    features["F_doc_num_sents"] = len(sentences)
    features["F_doc_num_tokens"] = len(doc)
    return _validate_features(features)


def select_features(features, feature_set="paper"):
    """Apply the feature selection used by the original classifier pipeline."""
    if feature_set == "all":
        return features
    if feature_set != "paper":
        raise ValueError(f"unsupported feature set: {feature_set}")
    return {name: value for name, value in features.items()
            if not name.startswith("F_num_")
            and name not in ("F_ratio_negated_words", "F_ratio_negations")}


def metadata_path(result_path):
    return Path(str(result_path) + ".metadata.json")


def create_metadata(records, args, languages):
    """Describe settings and dependencies without recording source texts or paths."""
    versions = {}
    for package in ("spacy", "spacy-stanza", "stanza", "torch", "pandas", "numpy", "tseval", "pyphen"):
        try:
            versions[package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            versions[package] = None
    names = [name for name in records[0] if name.startswith("F_")]
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "package_versions": versions,
        "feature_set": args.feature_set,
        "feature_count": len(names),
        "feature_names": names,
        "partial_feature_set": bool(args.skip_imageability or args.skip_tseval
                                    or args.skip_frequency or args.skip_syllables),
        "skip_imageability": args.skip_imageability,
        "skip_tseval": args.skip_tseval,
        "skip_frequency": args.skip_frequency,
        "skip_syllables": args.skip_syllables,
        "use_gpu": args.use_gpu,
        "requested_model_packages": {lang: constants.lang_dict[lang] for lang in languages},
        "processors": PROCESSORS.split(","),
        "row_count": len(records),
        "rows_per_language": {lang: sum(row["lang"] == lang for row in records) for lang in languages},
    }


def save_output(records, result_path, *, overwrite=False, metadata=None):
    """Write complete UTF-8 files atomically; preserve existing files by default."""
    path = Path(result_path)
    targets = [path] + ([metadata_path(path)] if metadata is not None else [])
    for target in targets:
        if target.exists() and not target.is_file():
            raise ValueError(f"output target must be a file: {target}")
        if target.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {target}; use --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in records for key in row))
    temporaries = []
    created = []
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporaries.append(Path(stream.name))
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in records:
                # Nested JSON metadata can be recovered by applying json.loads
                # to the corresponding CSV cell.
                writer.writerow({key: json.dumps(value, ensure_ascii=False)
                                 if isinstance(value, (dict, list)) else value
                                 for key, value in row.items()})
        if metadata is not None:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                             dir=path.parent, prefix=f".{path.name}.metadata.",
                                             suffix=".tmp", delete=False) as stream:
                temporaries.append(Path(stream.name))
                json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        for temporary, target in zip(temporaries, targets):
            if overwrite:
                os.replace(temporary, target)
            elif os.name == "nt":
                # Windows rename refuses an existing destination and works on
                # filesystems and synced folders that do not allow hard links.
                os.rename(temporary, target)
                created.append(target)
            else:
                # link() creates the destination only if absent, including when
                # a competing process creates it after the preflight check.
                os.link(temporary, target)
                created.append(target)
    except Exception:
        # Only remove destinations created by this call. Existing files are
        # never removed when a collision prevents the second file being saved.
        for target in created:
            target.unlink(missing_ok=True)
        raise
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)


def run(args):
    """Extract selected records and return the feature-augmented records."""
    for target in (args.output, metadata_path(args.output)):
        if args.input.resolve() == target.resolve():
            raise ValueError("input and output paths must differ")
        if target.exists() and not target.is_file():
            raise ValueError(f"output target must be a file: {target}")
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"output already exists: {target}; use --overwrite to replace it")
    rows = validate_records(read_data(args.input))
    selected = [(number, row) for number, row in enumerate(rows, 1)
                if args.languages is None or row["lang"] in args.languages]
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError("no records remain after filtering")
    languages = list(dict.fromkeys(row["lang"] for _, row in selected))
    if not args.skip_imageability:
        if args.megahr_dir is None:
            raise ValueError("provide --megahr-dir or explicitly use --skip-imageability")
        for lang in languages:
            path = args.megahr_dir / f"megahr.{lang}"
            if not path.is_file():
                raise FileNotFoundError(f"missing lexicon: {path}")
    if not args.skip_tseval and not args.skip_frequency:
        if args.fasttext_dir is None:
            raise ValueError("provide --fasttext-dir or explicitly use --skip-frequency")
        for lang in languages:
            path = args.fasttext_dir / f"cc.{lang}.300.vec"
            if not path.is_file():
                raise FileNotFoundError(f"missing fastText vocabulary: {path}")
    if not args.skip_tseval and not args.skip_syllables:
        import pyphen

        unsupported = [lang for lang in languages if pyphen.language_fallback(lang) is None]
        if unsupported:
            raise ValueError(
                "Pyphen has no syllable dictionary for " + ", ".join(unsupported)
                + "; explicitly use --skip-syllables to omit these five features"
            )

    from . import feature_extraction as fe

    if not args.skip_tseval and not args.skip_frequency:
        fe.configure_fasttext(args.fasttext_dir)
    output = [None] * len(selected)
    options = {name: getattr(args, name) for name in
               ("skip_imageability", "skip_tseval", "skip_frequency", "skip_syllables")}
    for lang in languages:
        print(f"Loading {lang} ({constants.lang_dict[lang]})", file=sys.stderr)
        nlp = load_language_specific_data(
            lang, constants.lang_dict[lang], model_dir=args.model_dir,
            use_gpu=args.use_gpu, download_models=args.download_models,
        )
        lexicon = None if args.skip_imageability else fe.read_imagebility_data(
            args.megahr_dir / f"megahr.{lang}", lang,
        )
        for position, (number, row) in enumerate(selected):
            if row["lang"] != lang:
                continue
            try:
                doc = nlp(row["text"])
                parse = parse_sentence if row["format"] == "sentence-level" else parse_document
                features = parse(doc, lexicon, lang, **options)
                features = select_features(features, args.feature_set)
            except Exception as exc:
                detail = str(exc) or type(exc).__name__
                raise RuntimeError(f"row {number} ({lang}, {row['format']}): {detail}") from exc
            output[position] = {**row, **features}
        del nlp, lexicon
        gc.collect()
    metadata = create_metadata(output, args, languages)
    save_output(output, args.output, overwrite=args.overwrite, metadata=metadata)
    print(f"Saved {len(output)} rows to {args.output}", file=sys.stderr)
    return output


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
