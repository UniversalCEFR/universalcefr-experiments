"""Shared record schema, table loading, and safe JSON output for converters."""

import argparse
import json
import os
from pathlib import Path
import tempfile


def make_record(*, title, lang, source_name, text_format, category,
                cefr_level, license_label, text):
    """Build the eight fields used by the original UniversalCEFR converters."""
    return {
        "title": title,
        "lang": lang,
        "source_name": source_name,
        "format": text_format,
        "category": category,
        "cefr_level": cefr_level,
        "license": license_label,
        "text": text,
    }


def read_table(input_path, columns, *, excel=False, sheet_name=0):
    """Read required columns with the original pandas missing-value semantics."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("install preprocessing/requirements.txt to read CSV or Excel") from exc
    if excel:
        table = pd.read_excel(input_path, sheet_name=sheet_name)
    else:
        table = pd.read_csv(input_path, encoding="utf-8-sig")
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError("missing input columns: " + ", ".join(sorted(missing)))
    if table.empty:
        raise ValueError("input table contains no rows")
    return table[list(columns)].to_dict(orient="records")


def clean_cell(value, *, default="", uppercase=False):
    """Strip a scalar table cell, handling missing and numeric cells explicitly."""
    import pandas as pd

    text = default if pd.isna(value) else str(value).strip()
    return text.upper() if uppercase else text


def check_output(output_path, input_paths=(), *, overwrite=False):
    """Check destinations before conversion and prohibit overwriting input."""
    output = Path(output_path)
    if any(output.resolve() == Path(path).resolve() for path in input_paths):
        raise ValueError("input and output paths must differ")
    if output.exists():
        if not output.is_file():
            raise ValueError(f"output must be a file: {output}")
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}; use --overwrite to replace it")


def write_records(records, output_path, *, overwrite=False):
    """Publish complete UTF-8 JSON, preserving existing files unless requested."""
    output = Path(output_path)
    check_output(output, overwrite=overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(records, stream, ensure_ascii=False, indent=4, allow_nan=False)
            stream.write("\n")
        if overwrite:
            os.replace(temporary, output)
        elif os.name == "nt":
            os.rename(temporary, output)
        else:
            os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_parser(description, *, input_help):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", type=Path, required=True, help=input_help)
    parser.add_argument("--output", type=Path, required=True, help="UniversalCEFR JSON output")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    return parser


def run_conversion(parser, args, converter, **options):
    """Run a converter and show concise errors without publishing partial data."""
    try:
        check_output(args.output, [args.input], overwrite=args.overwrite)
        records = converter(args.input, **options)
        write_records(records, args.output, overwrite=args.overwrite)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Wrote {len(records)} records to {args.output}")
    return 0
