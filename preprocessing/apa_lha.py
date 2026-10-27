"""Convert a directory of APA-LHA .simpde files to UniversalCEFR JSON."""

from pathlib import Path

if __package__:
    from .common import build_parser, check_output, make_record, run_conversion
else:
    from common import build_parser, check_output, make_record, run_conversion


def extract_text_from_file(txt_file):
    """Read the entire text, retaining the original Latin-1 fallback."""
    path = Path(txt_file)
    for encoding in ("utf-8", "latin-1"):
        try:
            with path.open(encoding=encoding) as stream:
                lines = stream.readlines()
            return "".join(lines).strip(), lines
        except UnicodeDecodeError:
            if encoding == "latin-1":
                raise


def convert_directory(input_path, *, cefr_level="A2"):
    """Convert matching files in filename order, without traversing subfolders."""
    directory = Path(input_path)
    if not directory.is_dir():
        raise ValueError(f"input must be a directory: {directory}")
    paths = sorted(path for path in directory.iterdir()
                   if path.is_file() and path.suffix == ".simpde")
    if not paths:
        raise ValueError(f"no .simpde files found in {directory}")
    records = []
    for path in paths:
        text, _ = extract_text_from_file(path)
        records.append(make_record(
            title=path.name, lang="de", source_name="apa-lha",
            text_format="document-level", category="reference",
            cefr_level=cefr_level, license_label="Unknown", text=text,
        ))
    return records


def main(argv=None):
    parser = build_parser(__doc__, input_help="directory containing .simpde files")
    parser.add_argument("--cefr-level", default="A2",
                        choices=("A1", "A1+", "A2", "A2+", "B1", "B1+",
                                 "B2", "B2+", "C1", "C1+", "C2", "C2+"),
                        help="label applied to every file in this directory (default: A2)")
    args = parser.parse_args(argv)
    try:
        check_output(args.output, args.input.glob("*.simpde"), overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return run_conversion(parser, args, convert_directory, cefr_level=args.cefr_level)


if __name__ == "__main__":
    raise SystemExit(main())
