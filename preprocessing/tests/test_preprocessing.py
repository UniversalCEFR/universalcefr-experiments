"""Synthetic regression tests for dataset conversion and output protection."""

import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from preprocessing import apa_lha, beast2019, deplain_apa_doc, deplain_apa_sent
from preprocessing import efcamdat_cleaned, merge_json
from preprocessing.common import write_records


ROOT = Path(__file__).resolve().parents[2]
FIELDS = {"title", "lang", "source_name", "format", "category",
          "cefr_level", "license", "text"}


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.temporary_root = ROOT / "preprocessing" / "outputs" / "tests"
        self.temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=self.temporary_root)
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        # Keep recursive fixture cleanup inside this test's generated directory.
        self.directory.resolve().relative_to(self.temporary_root.resolve())
        self.temporary.cleanup()

    def write_json(self, filename, value):
        path = self.directory / filename
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def write_csv(self, filename, fields, rows):
        path = self.directory / filename
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_apa_encoding_complete_text_and_deterministic_order(self):
        (self.directory / "b.simpde").write_bytes(" Zweite Überschrift\nText. ".encode("latin-1"))
        (self.directory / "a.simpde").write_text(" Erste Überschrift\nText. ", encoding="utf-8")
        (self.directory / "ignored.txt").write_text("ignored", encoding="utf-8")
        rows = apa_lha.convert_directory(self.directory, cefr_level="B1")
        self.assertEqual([row["title"] for row in rows], ["a.simpde", "b.simpde"])
        self.assertEqual(rows[1]["text"], "Zweite Überschrift\nText.")
        self.assertEqual(rows[0]["cefr_level"], "B1")
        self.assertEqual(set(rows[0]), FIELDS)
        self.assertEqual(rows[0]["source_name"], "apa-lha")

    def test_apa_cannot_overwrite_a_source_file(self):
        source = self.directory / "source.simpde"
        source.write_text("Keep this.", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            apa_lha.main(["--input", str(self.directory), "--output", str(source), "--overwrite"])
        self.assertEqual(source.read_text(encoding="utf-8"), "Keep this.")

    def test_apa_requires_matching_files(self):
        with self.assertRaisesRegex(ValueError, "no .simpde"):
            apa_lha.convert_directory(self.directory)

    def test_bea_labels_and_text_are_preserved(self):
        rows = [{"cefr": "B1+", "text": "  Original text.\n"}, {"cefr": "C2", "text": "Second."}]
        path = self.directory / "input.json"
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        converted = beast2019.convert_file(path)
        self.assertEqual([row["cefr_level"] for row in converted], ["B1", "C2"])
        self.assertEqual(converted[0]["text"], rows[0]["text"])
        self.assertEqual(converted[0]["format"], "sentence-level")
        self.assertEqual(converted[0]["license"], "CC BY-SA-NC 4.0")
        self.assertEqual(set(converted[0]), FIELDS)

    def test_bea_json_arrays_and_missing_labels(self):
        path = self.write_json("input.json", [{}, {"cefr": None, "text": "Example."}])
        rows = beast2019.convert_file(path)
        self.assertEqual([row["cefr_level"] for row in rows], ["Unknown", "Unknown"])
        self.assertEqual(rows[0]["text"], "")

    def test_bea_invalid_line_fails_without_partial_output(self):
        source = self.directory / "input.jsonl"
        source.write_text('{"cefr":"A2","text":"Example."}\nbroken\n', encoding="utf-8")
        output = self.directory / "output.json"
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            beast2019.main(["--input", str(source), "--output", str(output)])
        self.assertFalse(output.exists())

    def test_bea_rejects_nonobject_or_nonstring_text(self):
        for value in ([], "text", {"text": None}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                beast2019.transform_entry(value)

    def test_deplain_document_mapping_and_missing_values(self):
        source = self.write_csv("doc.csv", deplain_apa_doc.COLUMNS, [
            {"original": " Inhalt, mit Komma. ", "complex_level": " b2 ",
             "license": " Source terms ", "complex_title": " Titel "},
            {"original": "", "complex_level": "", "license": "", "complex_title": ""},
        ])
        rows = deplain_apa_doc.convert_file(source)
        self.assertEqual(rows[0]["text"], "Inhalt, mit Komma.")
        self.assertEqual(rows[0]["cefr_level"], "B2")
        self.assertEqual(rows[0]["title"], "Titel")
        self.assertEqual(rows[0]["license"], "Source terms")
        self.assertEqual(rows[0]["source_name"], "deplain-apa-doc")
        self.assertEqual(rows[1]["cefr_level"], "UNKNOWN")
        self.assertEqual(rows[1]["text"], "")

    def test_deplain_sentence_mapping(self):
        row = deplain_apa_sent.transform_row({"simplification": " Einfach. ",
              "language_level_simple": " a2 ", "license": " Terms ", "simple_title": " Titel "})
        self.assertEqual(row["cefr_level"], "A2")
        self.assertEqual(row["text"], "Einfach.")
        self.assertEqual(row["format"], "sentence-level")
        self.assertEqual(row["source_name"], "deplain-apa-sent")
        self.assertEqual(set(row), FIELDS)

    def test_table_missing_columns_fail(self):
        source = self.write_csv("bad.csv", ["text"], [{"text": "example"}])
        with self.assertRaisesRegex(ValueError, "missing input columns"):
            deplain_apa_doc.convert_file(source)

    def test_efcamdat_excel_and_sheet_selection(self):
        source = self.directory / "input.xlsx"
        with pd.ExcelWriter(source, engine="openpyxl") as writer:
            pd.DataFrame({"cefr": [" A2 "], "text_corrected": [" First. "]}).to_excel(
                writer, index=False, sheet_name="First")
            pd.DataFrame({"cefr": [" b1+ ", None], "text_corrected": [" Corrected. ", None]}).assign(
                retained=[1, 2]).to_excel(writer, index=False, sheet_name="Selected")
        self.assertEqual(efcamdat_cleaned.convert_file(source)[0]["cefr_level"], "A2")
        rows = efcamdat_cleaned.convert_file(source, sheet_name="Selected")
        self.assertEqual(rows[0]["cefr_level"], "b1+")
        self.assertEqual(rows[0]["text"], "Corrected.")
        self.assertEqual(rows[1]["cefr_level"], "Unknown")
        self.assertEqual(rows[0]["license"], "Cambridge")
        self.assertEqual(rows[0]["format"], "paragraph-level")

    def test_merge_order_and_metadata(self):
        first = self.write_json("one.json", [{"id": "a", "nested": {"key": [1, 2]}}])
        second = self.write_json("two.json", [{"id": "b"}, {"id": "a"}])
        output = self.directory / "merged.json"
        rows = merge_json.merge_json_files([second, first], output)
        self.assertEqual([row["id"] for row in rows], ["b", "a", "a"])
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), rows)

    def test_merge_failure_preserves_existing_output(self):
        source = self.write_json("input.json", [{"id": 1}])
        bad = self.write_json("bad.json", {"id": 2})
        output = self.directory / "output.json"
        output.write_text("keep", encoding="utf-8")
        with self.assertRaises(ValueError):
            merge_json.merge_json_files([source, bad], output, overwrite=True)
        self.assertEqual(output.read_text(), "keep")

    def test_merge_never_overwrites_an_input(self):
        source = self.write_json("input.json", [{"id": 1}])
        with self.assertRaisesRegex(ValueError, "paths must differ"):
            merge_json.merge_json_files([source], source, overwrite=True)
        self.assertEqual(json.loads(source.read_text()), [{"id": 1}])

    def test_output_protection(self):
        output = self.write_json("output.json", [{"id": 1}])
        with self.assertRaises(FileExistsError):
            write_records([{"id": 2}], output)
        self.assertEqual(json.loads(output.read_text()), [{"id": 1}])

    def test_imports_and_help_need_no_table_dependencies(self):
        modules = ("apa_lha", "beast2019", "deplain_apa_doc", "deplain_apa_sent",
                   "efcamdat_cleaned", "merge_json")
        imported = subprocess.run(
            [sys.executable, "-S", "-c", ";".join("import preprocessing." + name for name in modules)],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(imported.returncode, 0, imported.stderr)
        for name in modules:
            with self.subTest(module=name):
                result = subprocess.run([sys.executable, "-S", str(ROOT / "preprocessing" / (name + ".py")),
                                         "--help"], cwd=self.directory, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(list(self.directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
