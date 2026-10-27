# UniversalCEFR Experiments

Code accompanying [UniversalCEFR: Enabling Open Multilingual Research on
Language Proficiency Assessment](https://aclanthology.org/2025.emnlp-main.491/)
(Imperial et al., EMNLP 2025).

## Repository guide

| Folder | Purpose |
| --- | --- |
| [preprocessing](preprocessing/) | Convert APA-LHA, BEA 2019 Write & Improve, DEPlain, and EFCAMDAT source files into the UniversalCEFR record format. |
| [feature_extraction](feature_extraction/) | Extract the paper's 100 linguistic features, with an option to retain all 146 original columns. |
| [prompting](prompting/) | Run CEFR classification with the original reading or writing prompt templates, or preview prompts offline. |

Each folder contains its own usage instructions, dependency requirements, and
tests. Run commands from the repository root unless a folder's instructions
specify otherwise. Use separate virtual environments when running different
experiment pipelines.

## Data and resources

Obtain source datasets from their providers and follow their access and reuse
terms. Restricted corpus records and derived datasets are not included here.
Provider links are available on the [UniversalCEFR website](https://universalcefr.github.io/).
The [UniversalCEFR Hugging Face organisation](https://huggingface.co/UniversalCEFR)
hosts the released datasets.

Examples and tests use synthetic text. Models, embeddings, and lexicons are
obtained separately according to each pipeline's instructions. The public
refactors retain the documented experiment conventions, with specific fixes
and reproducibility limits described in the folder READMEs.

Please cite [Imperial et al. (2025)](https://aclanthology.org/2025.emnlp-main.491/)
when using this work. External datasets and software retain their own licenses
and citation requirements.
