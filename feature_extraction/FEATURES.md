# Feature definitions and compatibility

This release adapts the original UniversalCEFR extraction code accompanying
[Imperial et al. (2025)](https://aclanthology.org/2025.emnlp-main.491/).
Section 4 and Appendix E describe a 100-feature configuration. The original
extractor also produces intermediate morphology counts and German negation
features, giving 146 columns before selection.

## Feature sets

The default `paper` feature set follows the `all_wo_ratio` selection rule in the
original `features/classifier_cefr.py`. That historical name refers to a rule
that retains morphology ratios and excludes all 44 `F_num_*` columns plus
`F_ratio_negated_words` and `F_ratio_negations`. The `all` feature set retains
all 146 columns.

| Extraction group | `all` | `paper` | Column names |
| --- | ---: | ---: | --- |
| Noun and verb morphology | 88 | 44 | `F_num_*`, `F_ratio_*` |
| Dependency distances | 4 | 4 | `F_*_distance_betweeen_*` |
| Verb-to-noun ratio | 1 | 1 | `F_verb_noun_ratio` |
| German negation heuristics | 2 | 0 | `F_ratio_negated_words`, `F_ratio_negations` |
| Lexicon scores | 2 | 2 | `F_imagebility`, `F_concreteness` |
| TSEval features | 47 | 47 | `F_tseval_*` |
| Whole-text counts | 2 | 2 | `F_doc_num_sents`, `F_doc_num_tokens` |
| **Total** | **146** | **100** | |

The paper groups these 100 columns by linguistic function into morphosyntactic
(62), syntactic (18), length-based (11), lexical (4), readability (2),
psycholinguistic (2), and discourse (1) features. These categories differ from
the implementation groups above.

Feature names retain the original spelling, including `imagebility` and
`betweeen`, so existing column selections continue to work. Explicitly skipping
resource-dependent groups reduces the number of output columns.

## Calculation conventions

- Morphology covers 28 noun categories and 16 verb categories. Each category has
  a count and a ratio. Ratios use the number of `NOUN` or `VERB` tokens as their
  denominator. A missing category contributes zero. A zero denominator yields
  zero. Multi-valued morphology labels match the original exact-label rule.
- Sentence-level records are processed as one parsed text. Other supported
  formats are processed sentence by sentence. Their feature values are
  unweighted sentence means rounded to six decimal places. The two `F_doc_*`
  columns are then set to whole-text sentence and token counts. Consequently,
  sentence count features from TSEval and other sentence counts can differ
  from `F_doc_num_sents`.
- Token counts include punctuation. Most token ratios therefore include
  punctuation in their denominators. `F_verb_noun_ratio` uses `VERB / NOUN` and
  returns zero when there are no nouns. Proper nouns and auxiliaries are
  separate POS categories.
- Average dependency distance is the mean absolute difference between token
  and head positions. Maximum dependency distance is divided by token count.
  Both verb-particle distance measures are divided by token count and use the
  German `PTKVZ` tag.
- MEGAHR lookup tries exact token text, then exact lemma. Missing words use
  the mean of the corresponding lexicon column, rounded to six decimals.
  The first matching entry is used for duplicate terms.
- FastText features use vocabulary order as a frequency-rank proxy.
  The original TSEval reader includes the vector-file header at rank zero and
  reads the first 50,000 lines; this behavior is retained. Maximum and average
  rank features use `log(1 + rank)`. Lexical complexity uses the third quartile
  of these values among known words after stopword and punctuation removal.

## Interpretation and known limitations

**MEGAHR column labels are inherited from the original extractor.** The
[upstream lexicon documentation](https://github.com/clarinsi/megahr-crossling)
defines its columns as word, concreteness, imageability. The original extractor
assigns them to term, imageability, concreteness. This release preserves that
mapping for compatibility. With unmodified upstream files, `F_imagebility`
contains concreteness values and `F_concreteness` contains imageability values.
Account for this reversal when interpreting or relabelling exported columns.

The two negation features use German words and affixes. Their values in other
languages do not provide a validated negation measure. They are excluded from
the `paper` feature set. The parsing pipeline does not run named entity
recognition, so its named-entity ratio does not measure independently detected
entities. Language-specific morphology and dependency annotation coverage can
also produce zeros.

[TSEval's multilingual implementation](https://github.com/rstodden/text-simplification-evaluation/tree/12acc7b547cd4525218f742091b88b3fb21480bc)
supplies syllable, readability, lexical, POS, and syntactic features. Syllable
counts depend on Pyphen dictionary availability. Flesch Reading Ease and
Flesch-Kincaid Grade Level use formulas developed for English; interpreting
them across languages requires care. Sentence-level binary features become
proportions after document averaging. This release corrects the upstream
non-projectivity function to use indices relative to each sentence. Absolute
document indices caused failures on later sentences. A multi-sentence input is
non-projective if any sentence has a non-projective dependency tree.

Apart from that indexing correction, the release preserves the supplied
extraction and selection conventions.
The original experiment's complete dependency and model version lock was
unavailable, so identical historical outputs are not guaranteed. Record the
software versions, model resources, lexicons, and selected feature set with
each new experiment.
