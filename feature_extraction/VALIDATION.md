# Release validation

Checked on 27 October 2026 on Windows with CPU inference.

## Completed checks

- All 18 automated tests passed without model downloads.
- The 97 morphology, distance, verb/noun, negation, and lexicon features matched
  the supplied original code on synthetic annotated tokens, including duplicate
  lexicon entries, lemma fallback, and unknown words.
- Both example records ran through the installed English Stanza `gum` pipeline.
  One is a sentence and the other contains two sentences. Runs used an existing
  MEGAHR English lexicon and a small synthetic FastText-format vocabulary.
- The default export contained exactly 100 finite feature values per record.
  The `all` export contained 146. Shared column values matched between exports.
  Source record order, document sentence counts, and metadata were checked.
- Four focused regression tests cover the sentence-indexing correction for
  non-projectivity, including crossing dependencies in a later sentence.
- Package compilation and `--help` with third-party imports disabled succeeded.

## Environment

| Component | Version |
| --- | --- |
| Python | 3.14.4 |
| spaCy | 3.8.13 |
| Stanza | 1.11.1 |
| spacy-stanza | 1.0.4 |
| PyTorch | 2.11.0+cpu |
| pandas | 3.0.2 |
| NumPy | 2.4.4 |
| Pyphen | 0.17.2 |
| spacy-udpipe | 1.0.0 |
| TSEval | Multilingual fork at `12acc7b547cd4525218f742091b88b3fb21480bc` |

## Scope

The English runs validate execution and feature schemas. The synthetic rank
vocabulary does not establish agreement with scores from the full FastText
resource. A fresh dependency installation, model downloads, CUDA execution,
and full extraction across all 13 languages were not exercised. The original
experiment environment was not available as a complete version lock.

No research corpus records or downloaded model/lexicon resources are included
in this release. The example sentences and test inputs are synthetic.
