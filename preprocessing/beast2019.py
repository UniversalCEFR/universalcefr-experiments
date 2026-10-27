"""Convert BEA 2019 Write & Improve JSONL or JSON arrays to UniversalCEFR JSON."""

import json
from pathlib import Path

if __package__:
    from .common import build_parser, make_record, run_conversion
else:
    from common import build_parser, make_record, run_conversion


def read_entries(input_path):
    """Accept the original line-delimited .json files and ordinary JSON arrays."""
    content = Path(input_path).read_text(encoding="utf-8-sig").strip()
    if not content:
        raise ValueError("input contains no records")
    if content.startswith("["):
        entries = json.loads(content)
    else:
        entries = []
        for line_number, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON ({exc.msg})") from exc
    if not entries:
        raise ValueError("input contains no records")
    return entries


def transform_entry(entry, *, row_number=1):
    """Preserve the historical first-two-character CEFR mapping and raw text."""
    if not isinstance(entry, dict):
        raise ValueError(f"record {row_number}: expected a JSON object")
    level = entry.get("cefr")
    if level is None or level == "":
        level = "Unknown"
    elif not isinstance(level, str):
        raise ValueError(f"record {row_number}: cefr must be a string or null")
    elif level != "Unknown":
        level = level[:2]
    text = entry.get("text", "")
    if not isinstance(text, str):
        raise ValueError(f"record {row_number}: text must be a string")
    return make_record(
        title="na", lang="en", source_name="bea2019st-write-improve",
        text_format="sentence-level", category="learner", cefr_level=level,
        license_label="CC BY-SA-NC 4.0", text=text,
    )


def convert_file(input_path):
    return [transform_entry(entry, row_number=number)
            for number, entry in enumerate(read_entries(input_path), 1)]


def main(argv=None):
    parser = build_parser(__doc__, input_help="BEA JSONL file or JSON array")
    return run_conversion(parser, parser.parse_args(argv), convert_file)


if __name__ == "__main__":
    raise SystemExit(main())
