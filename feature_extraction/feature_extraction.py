"""Linguistic features from the UniversalCEFR research extractor.

Feature names and formulas retain the original behavior, including historical
spellings. See the accompanying documentation for methodological limitations
and source attribution.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path

import pandas as pd

from . import constants


FREQUENCY_FEATURES = frozenset({
    "max_pos_in_freq_table", "average_pos_in_freq_table",
    "get_lexical_complexity_score",
})
SYLLABLE_FEATURES = frozenset({
    "count_syllables_in_sentence", "count_syllables_per_sentence",
    "count_syllables_per_word", "sentence_fre", "sentence_fkgl",
})


def _require_tokens(tokens):
    if len(tokens) == 0:
        raise ValueError("Feature extraction requires at least one token.")


def check_ufeats(tokens, dict_morph_feats: dict, pos: str):
    """Count selected morphological annotations and divide by the POS count."""
    _require_tokens(tokens)
    num_pos = 0
    for token in tokens:
        if token.pos_ == pos:
            num_pos += 1
            for morph_type, morph_value in token.morph.to_dict().items():
                if (morph_type in dict_morph_feats
                        and morph_value in dict_morph_feats[morph_type]):
                    dict_morph_feats[morph_type][morph_value] += 1

    output_dict = {}
    for category, values in dict_morph_feats.items():
        for value, count in values.items():
            output_dict[f"F_num_{category}_{value}"] = count
            output_dict[f"F_ratio_{category}_{value}"] = (
                count / num_pos if num_pos else 0
            )
    return output_dict


def feature_morphology(tokens):
    """Return the original 88 noun and verb morphology features."""
    noun_values = check_ufeats(
        tokens, copy.deepcopy(constants.morph_dict_nouns), "NOUN"
    )
    verb_values = check_ufeats(
        tokens, copy.deepcopy(constants.morph_dict_verbs), "VERB"
    )
    return {**noun_values, **verb_values}


def distance_between_words(tokens):
    """Return mean head distance and maximum head distance divided by length."""
    _require_tokens(tokens)
    distances = [abs(token.i - token.head.i) for token in tokens]
    return (round(sum(distances) / len(distances), 6),
            round(max(distances) / len(tokens), 6))


def distance_between_verb_particles(tokens):
    """Return length-normalized mean and maximum distances for PTKVZ tokens."""
    _require_tokens(tokens)
    distances = [abs(token.i - token.head.i) for token in tokens
                 if token.tag_ == "PTKVZ"]
    if not distances:
        return 0, 0
    return (round((sum(distances) / len(distances)) / len(tokens), 6),
            round(max(distances) / len(tokens), 6))


def feature_dependency_tree_distance(tokens):
    avg_distance, max_distance = distance_between_words(tokens)
    avg_verb, max_verb = distance_between_verb_particles(tokens)
    return {
        "F_avg_distance_betweeen_words": avg_distance,
        "F_max_distance_betweeen_words": max_distance,
        "F_avg_distance_betweeen_verb_particle": avg_verb,
        "F_max_distance_betweeen_verb_particles": max_verb,
    }


def verb_noun_ratio(tokens):
    """Return VERB/NOUN, retaining zero when the noun count is zero."""
    _require_tokens(tokens)
    n_nouns = sum(token.pos_ == "NOUN" for token in tokens)
    n_verbs = sum(token.pos_ == "VERB" for token in tokens)
    return round(n_verbs / n_nouns, 6) if n_nouns else 0


def features_verb_noun_ratio(tokens):
    return {"F_verb_noun_ratio": verb_noun_ratio(tokens)}


def get_negations_ratio(tokens):
    """Apply the original German lexical heuristics for every input language."""
    _require_tokens(tokens)
    negations = {"kein", "nein", "nicht", "nie", "niemals"}
    num_real_negations = 0
    num_negated_words = 0
    for token in tokens:
        if token.lemma_ in negations:
            num_real_negations += 1
        elif (token.lemma_ != "los" and token.lemma_.endswith("los")
              and token.pos_ == "ADJ"):
            num_negated_words += 1
        elif (token.lemma_.lower().startswith("un") and token.lemma_ != "und"
              and token.pos_ == "ADJ"):
            num_negated_words += 1
        elif token.lemma_.lower().startswith("des") and token.pos_ == "ADJ":
            num_negated_words += 1
        elif token.lemma_.lower().startswith("irr") and token.pos_ == "ADJ":
            num_negated_words += 1
    return (round(num_negated_words / len(tokens), 6),
            round(num_real_negations / len(tokens), 6))


def feature_negations(tokens):
    negated_words, negations = get_negations_ratio(tokens)
    return {
        "F_ratio_negated_words": negated_words,
        "F_ratio_negations": negations,
    }


@dataclass(frozen=True)
class ImageabilityLexicon:
    """Indexed MEGAHR values with the original rounded fallback means."""

    scores: dict
    mean_imageability: float
    mean_concreteness: float

    @classmethod
    def from_dataframe(cls, data):
        required = {"term", "imageability", "concreteness"}
        if not required.issubset(data.columns) or data.empty:
            raise ValueError("The MEGAHR lexicon must contain terms and two scores.")
        data = data.copy()
        for column in ("imageability", "concreteness"):
            try:
                data[column] = pd.to_numeric(data[column], errors="raise")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"MEGAHR {column} scores must be numeric.") from exc
            if not all(math.isfinite(value) for value in data[column]):
                raise ValueError(f"MEGAHR {column} scores must all be finite.")
        scores = {}
        for term, imageability, concreteness in data[
            ["term", "imageability", "concreteness"]
        ].itertuples(index=False, name=None):
            # Original lookup takes the first duplicate term. All rows still
            # contribute to the fallback means.
            scores.setdefault(term, (imageability, concreteness))
        return cls(
            scores=scores,
            mean_imageability=round(data["imageability"].mean(), 6),
            mean_concreteness=round(data["concreteness"].mean(), 6),
        )


def read_imagebility_data(file_path, lang=None):
    """Read a three-column, headerless UTF-8 MEGAHR TSV once per language."""
    data = pd.read_csv(file_path, sep="\t", index_col=False, header=None,
                       encoding="utf-8")
    if len(data.columns) != 3:
        raise ValueError("The MEGAHR lexicon must have exactly three TSV columns.")
    # MEGAHR documents its columns as term, concreteness, imageability. The
    # original extractor labels these two scores in the opposite order. Keep
    # that historical mapping so existing feature values remain comparable.
    data.columns = ["term", "imageability", "concreteness"]
    return ImageabilityLexicon.from_dataframe(data)


def calculate_imagebility_concreteness(tokens, imageability_concreteness_data):
    """Match exact text, then lemma, then the rounded mean of the full lexicon."""
    _require_tokens(tokens)
    lexicon = imageability_concreteness_data
    if isinstance(lexicon, pd.DataFrame):
        lexicon = ImageabilityLexicon.from_dataframe(lexicon)
    if not isinstance(lexicon, ImageabilityLexicon):
        raise TypeError("Pass an ImageabilityLexicon or a MEGAHR DataFrame.")
    imageability = []
    concreteness = []
    fallback = (lexicon.mean_imageability, lexicon.mean_concreteness)
    for token in tokens:
        values = lexicon.scores.get(token.text)
        if values is None:
            values = lexicon.scores.get(token.lemma_, fallback)
        imageability.append(values[0])
        concreteness.append(values[1])
    return {
        "F_imagebility": sum(imageability) / len(imageability),
        "F_concreteness": sum(concreteness) / len(concreteness),
    }


def features_imageability_familiartiy(tokens, imageability_data):
    return calculate_imagebility_concreteness(tokens, imageability_data)


@lru_cache(maxsize=1)
def _load_tseval():
    try:
        import tseval.feature_extraction
    except ImportError as exc:
        raise ImportError(
            "Install feature_extraction/requirements.txt, including the pinned "
            "multilingual tseval fork, before extracting tseval features."
        ) from exc
    return tseval.feature_extraction


def _yield_utf8_lines(filepath):
    with Path(filepath).open("r", encoding="utf-8") as stream:
        for line in stream:
            yield line.rstrip("\n")


def configure_fasttext(directory):
    """Use local cc.LANG.300.vec files, preserving upstream rank semantics.

    The upstream rank reader includes the header in its line numbering. This
    behavior is retained. UTF-8 reads are explicit for Windows compatibility.
    """
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"FastText directory does not exist: {directory}")
    module = _load_tseval()
    module.FASTTEXT_EMBEDDINGS_PATH = directory
    module.yield_lines = _yield_utf8_lines
    module.get_word2rank.cache_clear()
    module.get_word2frequency.cache_clear()


def _is_non_projective(tokens):
    """Check complete sentences using the local head indices spaCy requires."""
    from spacy.pipeline._parser_internals.nonproj import is_nonproj_tree

    return int(any(
        is_nonproj_tree([token.head.i - sentence.start for token in sentence])
        for sentence in tokens.sents
    ))


def features_tseval(tokens, lang, skip_frequency=False, skip_syllables=False):
    """Calculate tseval features with explicit optional feature omissions."""
    _require_tokens(tokens)
    module = _load_tseval()
    if not skip_syllables:
        import pyphen

        if pyphen.language_fallback(lang) is None:
            raise ValueError(
                f"Language {lang!r} has no Pyphen hyphenation dictionary. "
                "Use --skip-syllables to omit its five syllable/readability "
                "features explicitly."
            )
    methods = (module.get_sentence_simplification_feature_extractors()
               + module.get_sentence_feature_extractors())
    output_dict = {}
    for method in methods:
        name = method.__name__
        if skip_frequency and name in FREQUENCY_FEATURES:
            continue
        if skip_syllables and name in SYLLABLE_FEATURES:
            continue
        if name == "max_pos_in_freq_table":
            # Upstream loads unused embedding vectors before this calculation.
            # The rank-based numerical result is unchanged without that load.
            value = max(module.get_frequency_table_ranks(tokens, lang))
        elif name == "is_non_projective":
            # Upstream passes document-global head indices for sentence spans.
            # spaCy requires indices within each sentence and can otherwise
            # raise AssertionError on every sentence after the first.
            value = _is_non_projective(tokens)
        else:
            value = method(tokens, lang)
        if not math.isfinite(value):
            raise ValueError(f"Feature F_tseval_{name} is not finite for {lang}.")
        output_dict["F_tseval_" + name] = round(value, 6)
    return output_dict


def features_POS_tags():
    """Retained compatibility entry point; POS features are provided by tseval."""
    return {}
