"""Regression tests for sentence-local dependency head indexing."""

import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from feature_extraction import feature_extraction as fe


@unittest.skipUnless(importlib.util.find_spec("spacy"), "spaCy is required")
class NonprojectivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import spacy

        cls.vocab = spacy.blank("en").vocab

    def make_document(self, crossing=False):
        from spacy.tokens import Doc

        if crossing:
            # In the second sentence, arcs 3->5 and 4->6 cross. The first
            # sentence remains projective.
            return Doc(
                self.vocab,
                words=["A", "runs", ".", "a", "b", "c", "d"],
                heads=[1, 1, 1, 5, 6, 5, 5],
                deps=["nsubj", "ROOT", "punct", "nsubj", "obj", "ROOT", "dep"],
                sent_starts=[True, False, False, True, False, False, False],
            )
        return Doc(
            self.vocab,
            words=["A", "runs", ".", "B", "walks", "."],
            heads=[1, 1, 1, 4, 4, 4],
            deps=["nsubj", "ROOT", "punct", "nsubj", "ROOT", "punct"],
            sent_starts=[True, False, False, True, False, False],
        )

    def test_projective_sentence_after_first(self):
        first, second = list(self.make_document().sents)
        self.assertEqual(fe._is_non_projective(first), 0)
        self.assertEqual(fe._is_non_projective(second), 0)

    def test_crossing_sentence_after_first(self):
        first, second = list(self.make_document(crossing=True).sents)
        self.assertEqual(fe._is_non_projective(first), 0)
        self.assertEqual(fe._is_non_projective(second), 1)

    def test_multiple_sentences(self):
        self.assertEqual(fe._is_non_projective(self.make_document()), 0)
        self.assertEqual(fe._is_non_projective(self.make_document(crossing=True)), 1)

    def test_tseval_dispatch_uses_local_indices(self):
        def is_non_projective(tokens, lang):
            raise AssertionError("the upstream absolute-index function was called")

        backend = SimpleNamespace(
            get_sentence_simplification_feature_extractors=lambda: [is_non_projective],
            get_sentence_feature_extractors=lambda: [],
        )
        with patch.object(fe, "_load_tseval", return_value=backend):
            result = fe.features_tseval(
                self.make_document(crossing=True), "en",
                skip_frequency=True, skip_syllables=True,
            )
        self.assertEqual(result, {"F_tseval_is_non_projective": 1})


if __name__ == "__main__":
    unittest.main()
