"""Synthetic CLI and mocked model checks. No downloads or inference are performed."""

from contextlib import nullcontext, redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from prompting import data, models, scoring, script


def record(text="A short example.", label="A2", lang="en"):
    return {"text": text, "cefr_level": label, "lang": lang, "format": "sentence-level"}


class InputAndScoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.template = self.root / "prompt.txt"
        self.template.write_text("Classify <<TEXT>>", encoding="utf-8")

    def test_local_json_csv_jsonl_preserve_unicode(self):
        rows = [record("Über Bücher.", lang="de")]
        for suffix in ("json", "jsonl", "csv"):
            path = self.root / ("input." + suffix)
            if suffix == "csv":
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            else:
                value = rows if suffix == "json" else rows[0]
                path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8-sig")
            self.assertEqual(data.load_records(path), rows)

    def test_templates_use_language_suffix_and_are_cached(self):
        path = self.root / "prompt_de.txt"
        path.write_text("Deutsch <<TEXT>>", encoding="utf-8")
        samples, _ = data.prepare_samples([record(lang="de"), record(lang="de")], self.template,
                                           language_specific=True)
        self.assertEqual(samples[0].prompt, "Deutsch A short example.")
        self.assertIs(samples[0].template, samples[1].template)

    def test_template_without_placeholder_is_rejected(self):
        self.template.write_text("No placeholder", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "placeholder"):
            data.prepare_samples([record()], self.template)

    def test_selected_record_validation_and_empty_selection(self):
        for row in [record(text=" "), record(label="Pre-A1"), record(lang="../en"), {"text": "Hello"}]:
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "record 1"):
                data.prepare_samples([row], self.template)
        with self.assertRaisesRegex(ValueError, "no records remain"):
            data.prepare_samples([record(label="A2+")], self.template)

    def test_limit_applies_after_plus_level_exclusion(self):
        samples, skipped = data.prepare_samples([record(label="A2+"), record("keep"), record("later")],
                                                self.template, limit=1)
        self.assertEqual(samples[0].record["text"], "keep")
        self.assertEqual(samples[0].source_record, 2)
        self.assertEqual(skipped, 1)

    def test_label_extraction_retains_legacy_precedence_and_case(self):
        for text, expected in [("B2 or A1", "A1"), ("Level B1.", "B1"),
                               ("a2", "UNKNOWN"), ("A10", "A1"), ("none", "UNKNOWN")]:
            self.assertEqual(scoring.extract_label(text), expected)

    def test_metrics_exclude_unknown_and_average_six_labels(self):
        result = scoring.compute_metrics([
            {"gold_label": "A1", "prediction": "A1"},
            {"gold_label": "B1", "prediction": "UNKNOWN"},
        ])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["coverage"], 0.5)
        self.assertAlmostEqual(result["f1_macro"], 1 / 6)

    def test_all_unknown_metrics_are_json_safe(self):
        result = scoring.compute_metrics([{"gold_label": "A1", "prediction": "UNKNOWN"}])
        self.assertIsNone(result["accuracy"])
        self.assertEqual(result["coverage"], 0)
        json.dumps(result, allow_nan=False)

    def args(self, rows, *options):
        path = self.root / "input.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        return script.build_parser().parse_args([
            "--input", str(path), "--prompt_template", str(self.template),
            "--output", str(self.root / "output.csv"), *options,
        ])

    def test_predictions_stay_aligned_after_skipped_record(self):
        args = self.args([record("skip", "A2+"), record("first", "B1"), record("second", "C1")])
        responses = iter(["B1", "C1"])
        with redirect_stdout(io.StringIO()):
            script.run(args, predictor=lambda prompt: next(responses))
        with args.output.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["text"] for row in rows], ["first", "second"])
        self.assertEqual([row["prediction"] for row in rows], ["B1", "C1"])
        self.assertEqual(list(rows[0]), list(data.PREDICTION_COLUMNS))

    def test_dry_run_never_loads_model(self):
        args = self.args([record()], "--dry-run")
        with patch("prompting.models.load_predictor") as load, redirect_stdout(io.StringIO()):
            script.run(args)
        load.assert_not_called()
        preview = json.loads(args.output.read_text(encoding="utf-8"))
        self.assertEqual(preview["prompt"], "Classify A short example.")

    def test_existing_output_blocks_prediction(self):
        args = self.args([record()])
        args.output.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            script.run(args, predictor=lambda prompt: self.fail("prediction should not run"))
        self.assertEqual(args.output.read_text(), "keep")

    def test_generation_failure_does_not_write_csv(self):
        args = self.args([record()])
        def fail(prompt):
            raise RuntimeError("test failure")
        with self.assertRaisesRegex(RuntimeError, "record 1"):
            script.run(args, predictor=fail)
        self.assertFalse(args.output.exists())


class Tensor:
    """Minimal tensor stand-in that records device movement and slicing."""
    def __init__(self, values):
        self.values = values
        self.device = "cpu"
        self.shape = (len(values), len(values[0])) if values and isinstance(values[0], list) else (len(values),)

    def __getitem__(self, item):
        return Tensor(self.values[item])

    def to(self, device):
        self.device = device
        return self

    def contiguous(self):
        return self


class Batch(dict):
    def to(self, device):
        for value in self.values():
            value.to(device)
        return self


class Tokenizer:
    eos_token_id = 2
    def __init__(self):
        self.chat_kwargs = None
        self.tokenize_kwargs = None
        self.decoded = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.chat_kwargs = messages, kwargs
        values = Tensor([[10, 20, 30]])
        return Batch(input_ids=values) if kwargs.get("return_dict") else values

    def __call__(self, prompt, **kwargs):
        self.tokenize_kwargs = kwargs
        return Batch(input_ids=Tensor([[10, 20, 30]]), attention_mask=Tensor([[1, 1, 1]]))

    def convert_tokens_to_ids(self, token):
        return 99

    def decode(self, output, **kwargs):
        self.decoded = output.values
        return " B1 "


class Model:
    device = "cuda:1"
    def __init__(self, family):
        self.family = family

    def get_input_embeddings(self):
        return SimpleNamespace(weight=SimpleNamespace(device=self.device))

    def generate(self, *args, **kwargs):
        self.options = kwargs
        self.input = args[0] if args else kwargs["input_ids"]
        return Tensor([[101, 102]] if self.family == "aya" else [[10, 20, 30, 101, 102]])


class GenerationTests(unittest.TestCase):
    def test_unresolved_offload_device_fails_before_generation(self):
        model = Model("gemma")
        model.device = "meta"
        predictor = models.Predictor(model, Tokenizer(), "gemma", SimpleNamespace(inference_mode=nullcontext))
        with self.assertRaisesRegex(ValueError, "input device is unresolved"):
            predictor("Classify this.")

    def test_all_families_preserve_generation_and_decode_paths(self):
        expected = {
            "gemma3": {"max_new_tokens": 10, "do_sample": False},
            "llama": {"max_new_tokens": 10, "do_sample": False, "eos_token_id": [2, 99]},
            "eurollm": {"max_new_tokens": 10}, "gemma": {"max_new_tokens": 10},
            "aya": {"do_sample": True, "min_length": 1, "max_length": 100},
            "bloomz": {"max_new_tokens": 128},
        }
        for family, options in expected.items():
            with self.subTest(family=family):
                model, tokenizer = Model(family), Tokenizer()
                predictor = models.Predictor(model, tokenizer, family, SimpleNamespace(inference_mode=nullcontext))
                self.assertEqual(predictor("Classify this."), "B1")
                self.assertEqual(tokenizer.decoded, [101, 102])
                self.assertEqual(model.input.device, "cuda:1")
                actual = {key: value for key, value in model.options.items()
                          if key not in ("input_ids", "attention_mask")}
                self.assertEqual(actual, options)
                if family == "aya":
                    self.assertEqual(tokenizer.tokenize_kwargs["max_length"], 128)
                    self.assertTrue(tokenizer.tokenize_kwargs["truncation"])
                if family == "gemma3":
                    self.assertEqual(tokenizer.chat_kwargs["pad_to_multiple_of"], 8)
                    self.assertEqual(tokenizer.messages[0]["content"][0]["type"], "text")

    def test_model_family_detection(self):
        self.assertEqual(models.model_family("google/gemma-3-12b-it"), "gemma3")
        self.assertEqual(models.model_family("utter-project/EuroLLM-9B-Instruct"), "eurollm")
        self.assertEqual(models.model_family("CohereForAI/aya-101"), "aya")
        with self.assertRaises(ValueError):
            models.model_family("unrelated-model")


if __name__ == "__main__":
    unittest.main()
