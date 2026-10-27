"""Convert a cleaned EFCAMDAT Excel worksheet to UniversalCEFR JSON."""

if __package__:
    from .common import build_parser, clean_cell, make_record, read_table, run_conversion
else:
    from common import build_parser, clean_cell, make_record, read_table, run_conversion


COLUMNS = ("cefr", "text_corrected")


def transform_row(row):
    return make_record(
        title="na", lang="en", source_name="efcamdat-cleaned",
        text_format="paragraph-level", category="learner",
        cefr_level=clean_cell(row["cefr"], default="Unknown"),
        license_label="Cambridge", text=clean_cell(row["text_corrected"]),
    )


def convert_file(input_path, *, sheet_name=0):
    rows = read_table(input_path, COLUMNS, excel=True, sheet_name=sheet_name)
    return [transform_row(row) for row in rows]


def main(argv=None):
    parser = build_parser(__doc__, input_help="cleaned EFCAMDAT .xlsx workbook")
    parser.add_argument("--sheet", default=None, help="worksheet name (default: first worksheet)")
    args = parser.parse_args(argv)
    return run_conversion(parser, args, convert_file,
                          sheet_name=0 if args.sheet is None else args.sheet)


if __name__ == "__main__":
    raise SystemExit(main())
