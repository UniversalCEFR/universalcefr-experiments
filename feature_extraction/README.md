# UniversalCEFR feature extraction

Extract linguistic features from UniversalCEFR-style records. This is a public
adaptation of the extraction code accompanying
[UniversalCEFR (Imperial et al., EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.491/).

The default `paper` configuration exports the paper's 100-feature selection.
Use `--feature-set all` to export all 146 columns from the original extractor.
The [feature guide](FEATURES.md) explains the selection, formulas, historical
column spellings, and known limitations. This folder covers extraction only.

## Install

Use Python 3.11 or later and Git. Run the following from the repository root
inside a virtual environment.

```sh
python -m venv .venv
# Activate the environment using your shell's activation command.
python -m pip install --upgrade pip
python -m pip install -r feature_extraction/requirements.txt
```

The requirements pin the multilingual TSEval fork to a commit. Its package name
is `tseval`. The Facebook implementation has a different API. The extraction
dependencies are separate from classifier training dependencies.

The default device is CPU. `--use-gpu` requires a compatible CUDA-enabled PyTorch
installation. Install the appropriate build using the
[PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Quick smoke run

The included examples are synthetic. This command downloads the required English
Stanza models and extracts a **partial 95-feature set**, omitting two lexicon
scores and three frequency features.

```sh
python -m feature_extraction --input feature_extraction/examples/input.json --output feature_extraction/outputs/example.csv --download-models --skip-imageability --skip-frequency
```

Model downloads occur only with `--download-models`. Models are cached in
`~/stanza_resources`; use `--model-dir PATH` to choose another directory. Omit
`--download-models` for subsequent runs using existing resources. Only languages
present in the selected input are loaded.

## Full feature extraction

Prepare these resources separately.

1. Download the `megahr.<lang>` files from
   [MEGAHR crossling](https://github.com/clarinsi/megahr-crossling), or clone that
   repository. Put the unmodified tab-separated files in one directory.
2. Download and decompress the `.vec.gz` files for the required languages from
   [FastText Common Crawl vectors](https://fasttext.cc/docs/en/crawl-vectors.html).
   Put files named `cc.<lang>.300.vec` in one directory. Retain their header and
   vocabulary order. These files are large. The extractor reads their first
   50,000 lines to compute vocabulary ranks and does not load vector matrices.

```sh
python -m feature_extraction --input data/input.json --output feature_extraction/outputs/features.csv --megahr-dir resources/megahr-crossling --fasttext-dir resources/fasttext --download-models
```

This exports 100 features when all feature resources are available. Add
`--feature-set all` for the original 146-column export.

**Historical MEGAHR labels are preserved.** With upstream lexicons,
`F_imagebility` contains concreteness values and `F_concreteness` contains
imageability values because the original code reversed the column labels.
See [FEATURES.md](FEATURES.md) before interpreting these scores.

Pyphen 0.17.2 has no dictionary for Arabic, Hindi, or Welsh. Extraction with
TSEval stops with a clear error for these languages unless `--skip-syllables`
is provided. That option omits three syllable features and two readability
scores, giving 95 features in `paper` mode with the other resources available.
Omitted features remain absent from the output.

## Input and output

Inputs may be a JSON array of records, JSONL/NDJSON, or a CSV with a header.
Files use UTF-8. Each record needs the following fields.

| Field | Values |
| --- | --- |
| `text` | A nonempty text string |
| `lang` | One of the language codes below |
| `format` | `sentence-level`, `paragraph-level`, `document-level`, or `dialogue-level` |

```json
{"id": "example-1", "text": "The students read a book.", "lang": "en", "format": "sentence-level"}
```

Other metadata fields, including labels if present, are retained. Labels are
optional. CSV metadata remains text; nested JSON metadata is serialized as JSON
inside its CSV cell. Input order is preserved. Columns beginning with `F_` are
reserved for extracted features and rejected in input.

Output is one CSV plus `<output>.metadata.json`, which records feature names,
selection flags, row counts, language packages, and installed software versions.
Existing files require `--overwrite`. A parsing or feature error stops the run
with its source record number before publishing a partial CSV. Input data is
processed locally. Model downloads contact the model provider only when enabled.

| Language | Code | Stanza package |
| --- | --- | --- |
| Arabic | `ar` | `padt` |
| Czech | `cs` | `pdt` |
| Welsh | `cy` | `ccg` |
| German | `de` | `hdt` |
| English | `en` | `gum` |
| Spanish | `es` | `ancora` |
| Estonian | `et` | `edt` |
| French | `fr` | `sequoia` |
| Hindi | `hi` | `hdtb` |
| Italian | `it` | `isdt` |
| Dutch | `nl` | `alpino` |
| Portuguese | `pt` | `bosque` |
| Russian | `ru` | `syntagrus` |

These are the original language/package mappings. They do not imply equal
annotation coverage or completed validation for every language.

## Options

Run `python -m feature_extraction --help` for all options.

| Option | Effect |
| --- | --- |
| `--languages en de` | Select input languages |
| `--limit 10` | Process the first ten selected records |
| `--feature-set paper` | Default 100-feature selection |
| `--feature-set all` | All 146 original features |
| `--skip-imageability` | Omit both MEGAHR scores |
| `--skip-frequency` | Omit three FastText-dependent features |
| `--skip-syllables` | Omit three syllable and two readability features |
| `--skip-tseval` | Omit all 47 TSEval features |
| `--model-dir PATH` | Set the Stanza cache location |
| `--download-models` | Download the required parsing resources |
| `--use-gpu` | Request CUDA instead of CPU |
| `--overwrite` | Replace output files |

Skip options produce partial feature sets. Their columns differ from the full
paper configuration and must be matched to any downstream model.

## Validation and reproducibility

```sh
python -m unittest discover -s feature_extraction/tests -v
```

The tests use synthetic inputs and require no model downloads. They cover
feature denominators, document aggregation, lexicon fallback, input validation,
row order, and output protection. See [VALIDATION.md](VALIDATION.md) for the
release checks and their limits.

The original experiment environment was not fully locked. Current model and
library versions may change annotations. The sidecar records software versions;
archive the exact model, lexicon, and vocabulary files with your experiment to
make a run reproducible.

## Citation and source attribution

Please cite [Imperial et al. (2025)](https://aclanthology.org/2025.emnlp-main.491/)
when using the UniversalCEFR pipeline. [SOURCES.md](SOURCES.md) records the
original source lineage and external resource citations and licenses.
