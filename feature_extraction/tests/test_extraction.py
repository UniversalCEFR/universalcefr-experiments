"""Regression checks using synthetic input; no model downloads are needed."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from feature_extraction import feature_extraction as fe
from feature_extraction import main


class Token:
    def __init__(self, text, pos="NOUN", lemma=None, index=0, head=0, morph=None, tag=""):
        self.text = text
        self.pos_ = pos
        self.lemma_ = text if lemma is None else lemma
        self.i = index
        self.head = SimpleNamespace(i=head)
        self.tag_ = tag
        self.morph = SimpleNamespace(to_dict=lambda: morph or {})


class Sentence(list):
    @property
    def sents(self):
        return iter([self])


class Document(list):
    def __init__(self, sentences):
        super().__init__(token for sentence in sentences for token in sentence)
        self.sentences = sentences

    @property
    def sents(self):
        return iter(self.sentences)


def record(text="A sentence.", lang="en", format="sentence-level", **metadata):
    return dict(text=text, lang=lang, format=format, **metadata)


class FeatureTests(unittest.TestCase):
    def test_morphology_uses_pos_denominator_and_does_not_accumulate(self):
        tokens = Sentence([Token("dogs", morph={"Number": "Plur"}),
                           Token("cats"), Token("run", "VERB", morph={"Tense": "Pres"})])
        features = fe.feature_morphology(tokens)
        self.assertEqual(len(features), 88)
        self.assertEqual(features["F_num_Number_Plur"], 1)
        self.assertEqual(features["F_ratio_Number_Plur"], 0.5)
        self.assertEqual(features["F_ratio_Tense_Pres"], 1)
        self.assertEqual(fe.feature_morphology(tokens), features)

    def test_distance_preserves_distinct_normalizations(self):
        tokens = Sentence([Token("A", index=5, head=6), Token("runs", index=6, head=6),
                           Token("off", index=7, head=6, tag="PTKVZ")])
        self.assertEqual(fe.distance_between_words(tokens), (0.666667, 0.333333))
        self.assertEqual(fe.distance_between_verb_particles(tokens), (0.333333, 0.333333))

    def test_zero_nouns_and_empty_input(self):
        self.assertEqual(fe.verb_noun_ratio([Token("run", "VERB")]), 0)
        with self.assertRaises(ValueError):
            fe.feature_morphology([])

    def test_lexicon_exact_lemma_duplicate_and_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "megahr.en"
            path.write_text("cat\t1\t2\ncat\t3\t4\ndog\t5\t6\n", encoding="utf-8")
            lexicon = fe.read_imagebility_data(path, "en")
        features = fe.calculate_imagebility_concreteness(
            [Token("cat"), Token("Dogs", lemma="dog"), Token("unknown")], lexicon)
        self.assertEqual(features, {"F_imagebility": 3, "F_concreteness": 4})

    def test_bad_lexicon_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "megahr.en"
            for value in ("cat\tNaN\t2\n", "cat\tinf\t2\n", "cat\tbad\t2\n"):
                path.write_text(value, encoding="utf-8")
                with self.assertRaises(ValueError):
                    fe.read_imagebility_data(path)

    def test_document_means_are_unweighted_and_counts_are_totals(self):
        short = Sentence([Token("run", "VERB")])
        long = Sentence([Token("cat"), Token("dog"), Token("run", "VERB")])
        result = main.parse_document(Document([short, long]), None, "en",
                                     skip_imageability=True, skip_tseval=True)
        self.assertEqual(result["F_verb_noun_ratio"], 0.25)
        self.assertEqual(result["F_doc_num_tokens"], 4)
        self.assertEqual(result["F_doc_num_sents"], 2)

    def test_nonfinite_features_are_rejected(self):
        with self.assertRaises(ValueError):
            main._validate_features({"F_bad": float("nan")})


class InputOutputTests(unittest.TestCase):
    def test_json_csv_and_jsonl_unicode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [record("Über den Fluss.", "de", id="001")]
            for suffix in (".json", ".jsonl", ".csv"):
                path = root / ("input" + suffix)
                if suffix == ".csv":
                    main.save_output(rows, path)
                else:
                    data = rows if suffix == ".json" else rows[0]
                    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8-sig")
                self.assertEqual(main.validate_records(main.read_data(path)), rows)

    def test_invalid_rows_are_rejected_before_loading_models(self):
        invalid = [record(text=" "), record(lang="zz"), record(format="essay"),
                   record(F_existing=0), {"text": "Hello"}, None]
        for row in invalid:
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "row 1"):
                main.validate_records([row])

    def test_invalid_input_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for value in ({"text": "hello"}, []):
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    main.read_data(path)
            path = Path(directory) / "input.csv"
            path.write_text("text,text,lang\na,b,en\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                main.read_data(path)

    def test_write_does_not_replace_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.csv"
            path.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                main.save_output([record()], path)
            self.assertEqual(path.read_text(), "keep")

    def test_sidecar_directory_cannot_cause_partial_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.csv"
            path.write_text("keep", encoding="utf-8")
            main.metadata_path(path).mkdir()
            with self.assertRaises(ValueError):
                main.save_output([record()], path, overwrite=True, metadata={})
            self.assertEqual(path.read_text(), "keep")

    def test_pipeline_preserves_order_and_loads_only_present_languages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            rows = [record(id=1, lang="en"), record(id=2, lang="de"), record(id=3, lang="en")]
            source.write_text(json.dumps(rows), encoding="utf-8")
            output = root / "output.csv"
            args = main.build_parser().parse_args([
                "--input", str(source), "--output", str(output),
                "--skip-imageability", "--skip-tseval",
            ])
            with patch.object(main, "load_language_specific_data",
                              return_value=lambda text: Sentence([Token("run", "VERB")])) as load:
                result = main.run(args)
            self.assertEqual([row["id"] for row in result], [1, 2, 3])
            self.assertEqual([call.args[0] for call in load.call_args_list], ["en", "de"])
            self.assertFalse(any(key.startswith("F_num_") for key in result[0]))
            self.assertNotIn("F_ratio_negations", result[0])

    def test_failed_row_does_not_publish_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            source.write_text(json.dumps([record()]), encoding="utf-8")
            output = root / "output.csv"
            args = main.build_parser().parse_args([
                "--input", str(source), "--output", str(output),
                "--skip-imageability", "--skip-tseval",
            ])
            with patch.object(main, "load_language_specific_data", return_value=lambda text: Sentence()):
                with self.assertRaisesRegex(RuntimeError, "row 1"):
                    main.run(args)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
