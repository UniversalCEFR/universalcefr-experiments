"""Concatenate JSON record arrays, preserving file order and record order."""

import argparse
import json
from pathlib import Path

if __package__:
    from .common import check_output, write_records
else:
    from common import check_output, write_records


def read_records(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        data = json.load(stream)
    if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
        raise ValueError(f"{path}: expected a JSON array of record objects")
    return data


def merge_json_files(json_files, output_file, *, overwrite=False):
    """Merge complete inputs; never silently skip missing or malformed files."""
    paths = list(json_files)
    if not paths:
        raise ValueError("provide at least one input file")
    check_output(output_file, paths, overwrite=overwrite)
    records = []
    for path in paths:
        try:
            records.extend(read_records(path))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot merge {path}: {exc}") from exc
    write_records(records, output_file, overwrite=overwrite)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True,
                        help="JSON arrays in the order they should be merged")
    parser.add_argument("--output", type=Path, required=True, help="merged JSON output")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    args = parser.parse_args(argv)
    try:
        records = merge_json_files(args.inputs, args.output, overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Wrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
