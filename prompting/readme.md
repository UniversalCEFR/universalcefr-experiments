# CEFR descriptor prompting

Run the descriptor-prompting experiments accompanying
[UniversalCEFR (Imperial et al., EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.491/).
The existing prompt templates and original model-specific generation controls
are retained. The code now separates input handling, model execution, and scoring.

## Quick preview without a model

Run commands from the repository root. Python 3.11 or later is recommended.
The local preview and tests need only the Python standard library.

```sh
python -m prompting --input prompting/examples/input.json --prompt-template prompting/prompt_files/lang_read/prompt_with_specs.txt --lang-prompts --dry-run --output prompting/outputs/prompts.jsonl
```

This writes the rendered English and German prompts from synthetic examples.
Their CEFR labels are illustrative test values. No model is loaded. With local
`--input`, preview does not contact Hugging Face or require model dependencies.
A preview using `--dataset-id` may download that dataset unless offline mode is set.

## Install model dependencies

Use a virtual environment and install the separate prompting requirements.

```sh
python -m pip install -r prompting/requirements.txt
```

For CUDA, install the appropriate PyTorch build for your system using the
[official instructions](https://pytorch.org/get-started/locally/). Gated models
and datasets may require access under your own Hugging Face account. Configure
`HF_TOKEN` in your environment or use an existing Hub login. Tokens are never
written into this code. Model execution is local; Hugging Face may download
weights, tokenizer files, and the requested dataset.

## Run classification

Use English reading descriptors for every record:

```sh
python -m prompting --input data/test.json --prompt-template prompting/prompt_files/lang_read/prompt_with_specs_en.txt --model-id google/gemma-7b-it --output prompting/outputs/gemma-en-reading.csv
```

Use descriptors in each record's language:

```sh
python -m prompting --input data/test.json --prompt-template prompting/prompt_files/lang_read/prompt_with_specs.txt --lang-prompts --model-id google/gemma-7b-it --output prompting/outputs/gemma-language-reading.csv
```

Replace `lang_read` with `lang_write` to use written-production descriptors.
When `--lang-prompts` is set, a base filename such as `prompt_with_specs.txt`
resolves to `prompt_with_specs_<lang>.txt`. Use the unsuffixed base filename for
that option. All 28 existing descriptor files are unchanged.

To load a dataset from the Hub, replace `--input data/test.json` with
`--dataset-id UniversalCEFR/universalcefr_test`. The original script selected the
`train` split, and this remains the default. Set `--split` explicitly when your
dataset uses a different split. A dataset name containing `test` does not change
the split argument.

The previous `python prompting/script.py` entry point and underscore options
`--prompt_template`, `--dataset_id`, `--model_id`, and `--lang_prompts` remain
available. Local JSON/JSONL/CSV file paths also work with legacy `--dataset_id`.

## Input and output

Local input may be a JSON array, JSONL/NDJSON, or CSV with a header, encoded as
UTF-8. Each selected record requires the following fields.

| Field | Meaning |
| --- | --- |
| `text` | Nonempty input text |
| `lang` | Lowercase two- or three-letter language code |
| `format` | `sentence-level`, `paragraph-level`, `document-level`, or `dialogue-level` |
| `cefr_level` | Gold label `A1`, `A2`, `B1`, `B2`, `C1`, or `C2` |

The original rule excluding plus-level labels is retained, so usual labels such
as `A2+` and `B1+` are skipped. Other unsupported labels fail validation.
`--limit` counts selected records after these exclusions. A run with no selected
records stops before model loading.

The output CSV retains the original five columns in this order:

```text
text,lang,format,gold_label,prediction
```

Only selected records are written, in source order, with their own prediction.
The original script paired predictions with the unfiltered dataset and could
misalign records after a skipped plus-level label. This release corrects that.
Other input metadata is omitted, matching the original CSV schema.

A companion `<output>.metrics.json` records scores, recognized/unknown counts,
coverage, skipped plus levels, and supplied model/dataset revisions. Existing
outputs require `--overwrite`. Validation and generation failures occur before
a partial predictions CSV is published. CSV and metrics are separate files,
each written atomically.

## Model families and original generation settings

| Family | Automatic ID match | Input and generation |
| --- | --- | --- |
| Gemma 3 | `gemma-3-12b-it` | Structured chat, padding to a multiple of 8, 10 new tokens, sampling disabled |
| EuroLLM | `EuroLLM-9B-Instruct` | Chat template, 10 new tokens |
| Gemma | `gemma-7b-it` | Raw prompt, 10 new tokens |
| Aya | `aya-101` | Raw prompt truncated to **128 input tokens**, sampling enabled, output length 1–100 tokens |
| Llama | `Llama-3.1-8B` | Chat template, 10 new tokens, sampling disabled, EOS and end-of-turn terminators |
| BLOOMZ | `bloomz` | Raw prompt, 128 new tokens |

Where sampling is not specified above, checkpoint generation defaults remain
in effect, as in the original code. Aya's inherited 128-token input limit can
truncate descriptors and the supplied text. It is retained for compatibility.
A Llama checkpoint must provide the chat template used by this branch.
Use `--model-family` when loading a compatible model from a local directory
whose name does not identify its family.

The default `--device auto` uses CUDA when available and CPU otherwise. CUDA
uses automatic model placement. Inputs follow the model's embedding device,
which also fixes the original EuroLLM CPU-input/device mismatch. The default
dtype is bfloat16 on CUDA, except Aya retains its original model-default dtype.
CPU uses float32. `--device` and `--dtype` allow explicit choices. CPU inference
with these large models may require substantial memory and time.

Custom repository code is disabled by default. If a selected model requires it,
pass `--trust-remote-code` after reviewing the model repository. `--cache-dir`
sets resource storage; `--local-files-only` uses cached resources in offline mode.
The script leaves GPU visibility and cache settings unchanged unless an explicit
option selects offline mode. Set `CUDA_VISIBLE_DEVICES` in your environment if
needed for your hardware.

## Scoring and reproducibility

Prediction extraction preserves the original substring policy. It searches the
case-sensitive labels in the order A1, A2, B1, B2, C1, C2 and takes the first
matching label in that order. For example, `B2 or A1` produces `A1`, `A10`
produces `A1`, and lowercase `a2` produces `UNKNOWN`. This behavior is preserved
so the refactor does not silently change existing prediction interpretation.

The original prompting script's scoring convention excludes `UNKNOWN`
predictions and macro-averages precision, recall, and F1 over all six CEFR
labels, with zero for an undefined per-label score. The new coverage field is
the fraction with a recognized prediction. If all predictions are unknown,
scores are `null` and coverage is zero. These aggregate metrics do not cover
every per-language evaluation reported elsewhere in the experiments.

Use `--revision`, `--dataset-revision`, and optionally `--seed` to record a
repeatable configuration. Aya samples by default and the original code set no
seed. Dependency ranges target the Transformers 4.x API and are not a recovered
lockfile from the 2025 experiments. Model/tokenizer changes, hardware, and
sampling can change outputs.

## Code layout and tests

| File | Responsibility |
| --- | --- |
| `script.py` | Arguments and experiment orchestration |
| `data.py` | Input validation, cached templates, and file output |
| `models.py` | Lazy model loading and family-specific generation |
| `scoring.py` | CEFR extraction and aggregate metrics |
| `prompt_files/` | Original multilingual descriptor templates |
| `examples/` | Synthetic input for local previews |

```sh
python -m unittest discover -s prompting/tests -v
```

The tests use synthetic records and mocked model/tokenizer objects. They cover
all six generation paths, output slicing and device placement, the skipped-row
alignment fix, template selection, legacy label parsing, metrics, dry-run mode,
and output protection. Actual model inference, downloads, and a fresh model
dependency installation were not run during this refactor.
