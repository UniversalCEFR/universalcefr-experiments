"""Convert DEPlain-APA simplified sentence CSV rows to UniversalCEFR JSON."""

if __package__:
    from .common import build_parser, clean_cell, make_record, read_table, run_conversion
else:
    from common import build_parser, clean_cell, make_record, read_table, run_conversion


COLUMNS = ("simplification", "language_level_simple", "license", "simple_title")


def transform_row(row):
    return make_record(
        title=clean_cell(row["simple_title"]), lang="de",
        source_name="deplain-apa-sent", text_format="sentence-level",
        category="reference",
        cefr_level=clean_cell(row["language_level_simple"], default="Unknown", uppercase=True),
        license_label=clean_cell(row["license"]), text=clean_cell(row["simplification"]),
    )


def convert_file(input_path):
    return [transform_row(row) for row in read_table(input_path, COLUMNS)]


def main(argv=None):
    parser = build_parser(__doc__, input_help="DEPlain-APA simplified-sentence CSV")
    return run_conversion(parser, parser.parse_args(argv), convert_file)


if __name__ == "__main__":
    raise SystemExit(main())
