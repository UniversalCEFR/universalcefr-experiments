# Preprocessing

Convert locally obtained source datasets into the eight-field UniversalCEFR JSON
format. Each script has a command-line interface and reusable Python functions.
Importing a module performs no data processing or file writes.

The scripts preserve the original dataset-specific text transformations, source
names, categories, formats, and license labels. The output labels are inherited
metadata. They do not establish or verify the source dataset's legal terms.
Obtain the datasets and their current usage conditions from their providers.
No source corpus records are included in this folder.

## Setup

Use Python 3.11 or later. The APA-LHA converter, BEA converter, and JSON merger
use the Python standard library. The DEPlain CSV and EFCAMDAT Excel converters
also require pandas and openpyxl.

```sh
python -m pip install -r preprocessing/requirements.txt
```

Run the examples below from the repository root. Replace the example input paths
with paths to your copies of the source data. `--help` works without installing
the optional table-reading dependencies.

## Dataset converters

| Script | Input | Source fields or files | Output source / format |
| --- | --- | --- | --- |
| `apa_lha.py` | Directory | `.simpde` files | `apa-lha` / `document-level` |
| `beast2019.py` | JSONL or JSON array | `cefr`, `text` | `bea2019st-write-improve` / `sentence-level` |
| `deplain_apa_doc.py` | CSV | `original`, `complex_level`, `license`, `complex_title` | `deplain-apa-doc` / `document-level` |
| `deplain_apa_sent.py` | CSV | `simplification`, `language_level_simple`, `license`, `simple_title` | `deplain-apa-sent` / `sentence-level` |
| `efcamdat_cleaned.py` | Excel workbook | `cefr`, `text_corrected` | `efcamdat-cleaned` / `paragraph-level` |

### APA-LHA

```sh
python -m preprocessing.apa_lha --input data/A2-OR --output preprocessing/outputs/apa-a2.json --cefr-level A2
```

The selected CEFR label applies to every file in the directory. Its default is
`A2`, matching the original script. Files are processed in filename order,
without recursion. The complete file content, including its first line, is
retained with outer whitespace removed. Reading tries UTF-8, then Latin-1.
The output is German reference text with license label `Unknown`.

### BEA 2019 Write & Improve

```sh
python -m preprocessing.beast2019 --input data/B.dev.json --output preprocessing/outputs/bea.json
```

The original `.json` files contain one JSON object per line. Both that format
and ordinary JSON arrays are accepted. Empty lines are ignored. The original
first-two-character CEFR rule is retained, so `B1+` becomes `B1`. Text is kept
verbatim, including whitespace. Missing or null CEFR labels become `Unknown`;
a missing `text` field becomes an empty string. Non-string text is rejected.
The original `sentence-level` format and `CC BY-SA-NC 4.0` license label are
preserved verbatim. The license label's historical spelling is not normalized.

### DEPlain-APA

```sh
python -m preprocessing.deplain_apa_doc --input data/documents.csv --output preprocessing/outputs/deplain-doc.json
python -m preprocessing.deplain_apa_sent --input data/sentences.csv --output preprocessing/outputs/deplain-sent.json
```

The document converter selects the original complex text. The sentence converter
selects its simplified text. Both preserve row order, strip surrounding whitespace,
and uppercase the supplied CEFR label. Missing labels become `UNKNOWN`, matching
the original scripts. Missing titles, text, and license values become empty
strings. CSV missing-value interpretation follows pandas defaults.

### EFCAMDAT

```sh
python -m preprocessing.efcamdat_cleaned --input data/database.xlsx --output preprocessing/outputs/efcamdat.json
```

The first worksheet is used by default. Add `--sheet "Sheet name"` to select a
worksheet by name. CEFR labels and corrected text have surrounding whitespace
removed. Label case is preserved. Missing labels become `Unknown`, and missing
text becomes an empty string. Output records are English learner paragraphs,
with the inherited license label `Cambridge`.

## Merge converted files

```sh
python -m preprocessing.merge_json --inputs preprocessing/outputs/apa-a2.json preprocessing/outputs/apa-b1.json --output preprocessing/outputs/apa.json
```

Each input must be a JSON array of objects. The merger concatenates files in
the specified order and preserves record order within each file. It performs
no deduplication, filtering, label changes, or metadata changes. A missing or
malformed input stops the merge before an output is written.

## Output and compatibility

Each converter writes a UTF-8 JSON array. Every converted record contains
`title`, `lang`, `source_name`, `format`, `category`, `cefr_level`, `license`,
and `text`. A synthetic example is shown below.

```json
{
  "title": "na",
  "lang": "en",
  "source_name": "bea2019st-write-improve",
  "format": "sentence-level",
  "category": "learner",
  "cefr_level": "B1",
  "license": "CC BY-SA-NC 4.0",
  "text": "The student visited the library."
}
```

Existing output files require `--overwrite`. Inputs cannot be overwritten.
Directories are created as needed, and complete results are written through
a temporary file. Invalid JSON, missing required table columns, and empty
converter inputs produce explicit errors. Records with missing text are retained
as empty strings to preserve the original record counts. The feature extraction
pipeline requires nonempty text, so review those records before using its output.

The existing filenames remain executable directly, for example
`python preprocessing/apa_lha.py --help`. Module invocation with `python -m`
also works. The empty `script` placeholder was removed.

Changes from the original scripts are limited to configurable paths, reusable
functions, validation, deterministic APA-LHA ordering, and output protection.
BEA's accidental missing-label value `Un` is now `Unknown`. Malformed BEA lines
and invalid merger inputs raise errors instead of silently producing incomplete
outputs. Numeric table cells are converted to strings instead of failing during
whitespace stripping.

## Tests

```sh
python -m unittest discover -s preprocessing/tests -v
```

Tests use synthetic data only and cover mappings, missing values, file encoding,
CSV and Excel reading, merge order, validation, and output protection. They do
not download or process a research corpus.
