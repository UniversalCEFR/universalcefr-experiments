# Sources and external resources

This release adapts the original UniversalCEFR `features/feature_extraction`
Python sources. The original `features/classifier_cefr.py` supplies the selection
rule for the 100-feature configuration. Classifier training scripts, corpus
records, model weights, and downloaded lexicons are distributed separately.

The original source README credits the following work.

- [UniversalCEFR, Imperial et al. (2025)](https://aclanthology.org/2025.emnlp-main.491/).
- [Reference-less Quality Estimation of Text Simplification Systems,
  Martin et al. (2018)](https://aclanthology.org/W18-7005/).
- [A Multi-Lingual and Cross-Domain Analysis of Features for Text
  Simplification, Stodden and Kallmeyer (2020)](https://aclanthology.org/2020.readi-1.12/).
- The HHUplexity feature collection attributed to Arps et al. (2022) in the
  original README. Its cited
  [notebook revision](https://github.com/Vipitis/HHUplexity/blob/b0fec3efb9fdc5535ae39416d196099c017bb3dc/Feature_collection.ipynb)
  documents that lineage. The notebook itself is not included here.
- [Predicting Concreteness and Imageability of Words Within and Across Languages
  via Word Embeddings, Ljubešić, Fišer, and Peti-Stantić (2018)](https://aclanthology.org/W18-3028/).

## Dependency and resource terms

The repository has an [MIT license](../LICENSE). External software, lexicons,
language models, and embeddings retain their own terms.

| Component | Source and terms |
| --- | --- |
| Multilingual TSEval | [Pinned revision](https://github.com/rstodden/text-simplification-evaluation/tree/12acc7b547cd4525218f742091b88b3fb21480bc), [CC BY-NC 4.0 license](https://github.com/rstodden/text-simplification-evaluation/blob/12acc7b547cd4525218f742091b88b3fb21480bc/LICENSE). Upstream also identifies EASSE-derived code in its source. Installed as an external dependency. |
| MEGAHR crossling | [Repository and citation instructions](https://github.com/clarinsi/megahr-crossling). Obtain the resource separately and retain its provenance and applicable terms. |
| FastText Common Crawl vectors | [Downloads, attribution, and CC BY-SA 3.0 terms](https://fasttext.cc/docs/en/crawl-vectors.html). Obtain vectors separately. |
| Stanza models | [Model documentation](https://stanfordnlp.github.io/stanza/available_models.html). Model resources and their underlying datasets can have separate licenses. |

Installing TSEval does not change its noncommercial license. The repository's
MIT license does not replace the licenses of external components.

## Changes for this release

- Added a package entry point, explicit paths, CPU support, input validation,
  output protection, run metadata, examples, and regression tests.
- Replaced repeated MEGAHR table scans with a per-language lookup while
  preserving first-duplicate, lemma, and fallback behavior.
- Made FastText file reads UTF-8 and avoided loading unused embedding matrices
  for the maximum rank feature. Original vocabulary rank calculations remain.
- Preserved historical feature spellings and the original MEGAHR label mapping.
- Corrected sentence-relative dependency indices for non-projectivity so
  multi-sentence records can be processed.
- Added explicit options for partial feature sets when resources are unavailable.

See [FEATURES.md](FEATURES.md) for calculation conventions and limitations.
