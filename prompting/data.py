"""Input records, cached prompt templates, and explicit output paths."""

from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile

from .scoring import LABELS


FORMATS = ("sentence-level", "paragraph-level", "document-level", "dialogue-level")
PREDICTION_COLUMNS = ("text", "lang", "format", "gold_label", "prediction")


def load_records(input_path=None, dataset_id=None, *, split="train", revision=None,
                 cache_dir=None, local_files_only=False):
    """Read local records without optional dependencies, or load a Hub dataset."""
    if input_path is None and dataset_id and Path(dataset_id).is_file():
        input_path = Path(dataset_id)
    if input_path is not None:
        path = Path(input_path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            suffix = path.suffix.lower()
            if suffix == ".json":
                records = json.load(stream)
                if not isinstance(records, list):
                    raise ValueError("JSON input must be an array of records")
            elif suffix in (".jsonl", ".ndjson"):
                records = []
                for number, line in enumerate(stream, 1):
                    if line.strip():
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"JSONL line {number}: {exc.msg}") from exc
            elif suffix == ".csv":
                reader = csv.DictReader(stream)
                if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                    raise ValueError("CSV input needs a header with unique column names")
                records = list(reader)
                if any(None in record for record in records):
                    raise ValueError("CSV input contains more fields than header columns")
            else:
                raise ValueError("local input extension must be .json, .jsonl, .ndjson, or .csv")
    else:
        if local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
        from datasets import DownloadConfig, load_dataset

        records = load_dataset(dataset_id, split=split, revision=revision,
                               cache_dir=str(cache_dir) if cache_dir else None,
                               download_config=DownloadConfig(local_files_only=local_files_only))
    if len(records) == 0:
        raise ValueError("input contains no records")
    return records


@dataclass(frozen=True)
class PromptSample:
    source_record: int
    record: dict
    template_path: Path
    template: str

    @property
    def prompt(self):
        return self.template.replace("<<TEXT>>", self.record["text"])


def skip_plus_level(label):
    """Retain the original script's rule for excluding plus-level labels."""
    return "+" in label and ("1" not in label or "2" not in label)


def prepare_samples(records, template_path, *, language_specific=False, limit=None):
    """Validate selected records and read each required template once."""
    base = Path(template_path)
    templates = {}
    samples = []
    skipped = 0
    for number, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"record {number}: expected an object")
        label = record.get("cefr_level")
        if not isinstance(label, str):
            raise ValueError(f"record {number}: cefr_level must be a string")
        if skip_plus_level(label):
            skipped += 1
            continue
        if label not in LABELS:
            raise ValueError(f"record {number}: unsupported CEFR level {label!r}")
        if not isinstance(record.get("text"), str) or not record["text"].strip():
            raise ValueError(f"record {number}: text must be a nonempty string")
        lang = record.get("lang")
        if not isinstance(lang, str) or not re.fullmatch(r"[a-z]{2,3}", lang):
            raise ValueError(f"record {number}: lang must be a lowercase language code")
        if record.get("format") not in FORMATS:
            raise ValueError(f"record {number}: format must be one of {', '.join(FORMATS)}")
        path = base.with_name(f"{base.stem}_{lang}{base.suffix}") if language_specific else base
        if path not in templates:
            template = path.read_text(encoding="utf-8")
            if "<<TEXT>>" not in template:
                raise ValueError(f"prompt template has no <<TEXT>> placeholder: {path}")
            templates[path] = template
        samples.append(PromptSample(number, record, path, templates[path]))
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise ValueError("no records remain after excluding plus-level labels")
    return samples, skipped


def check_destination(path, *, overwrite=False):
    path = Path(path)
    if path.exists() and not path.is_file():
        raise ValueError(f"output target must be a file: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}; use --overwrite to replace it")


@contextmanager
def atomic_output(path, *, overwrite=False):
    """Publish one complete UTF-8 file, refusing existing files by default."""
    path = Path(path)
    check_destination(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            yield stream
        if overwrite:
            os.replace(temporary, path)
        elif os.name == "nt":
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_predictions(records, path, *, overwrite=False):
    with atomic_output(path, overwrite=overwrite) as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def save_preview(samples, path, *, overwrite=False):
    with atomic_output(path, overwrite=overwrite) as stream:
        for sample in samples:
            row = {"source_record": sample.source_record, "lang": sample.record["lang"],
                   "format": sample.record["format"], "gold_label": sample.record["cefr_level"],
                   "prompt": sample.prompt}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(value, path, *, overwrite=False):
    with atomic_output(path, overwrite=overwrite) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
