"""Protect manuscript facts and files at the project helper boundary."""
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "paper-writing-shared/scripts/project.py"
spec = importlib.util.spec_from_file_location("paper_writing_project", SCRIPT)
project = importlib.util.module_from_spec(spec)
spec.loader.exec_module(project)


class ProjectIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def manuscript(self, name, body):
        path = self.root / name
        (path / "sections").mkdir(parents=True)
        (path / "main.tex").write_text(r"\documentclass{article}\input{sections/body}" )
        (path / "sections/body.tex").write_text(body)
        return path

    def compare(self, before, after):
        with redirect_stdout(io.StringIO()) as output:
            code = project.compare_projects(argparse.Namespace(before=str(before), after=str(after), report=None))
        return code, json.loads(output.getvalue())

    def test_prose_edit_preserves_formula_citations_and_macro(self):
        protected = r"\newcommand{\vect}[1]{\mathbf{#1}} $x^2\geq0$ \label{eq:claim}\cite{real-key}"
        before = self.manuscript("before", "We now show the claim. " + protected)
        after = self.manuscript("after", "We establish the claim. " + protected)
        code, report = self.compare(before, after)
        self.assertEqual(code, 0)
        self.assertFalse(report["changes"])

    def test_formula_scope_change_is_detected(self):
        before = self.manuscript("before", r"For $x\geq0$, the result holds.")
        after = self.manuscript("after", r"For $x\in\mathbb R$, the result holds.")
        code, report = self.compare(before, after)
        self.assertEqual(code, 1)
        self.assertIn("math_expressions", report["changes"])

    def test_macro_body_change_is_detected_with_same_name(self):
        before = self.manuscript("before", r"\newcommand{\domain}{\mathbb{R}}")
        after = self.manuscript("after", r"\newcommand{\domain}{\mathbb{C}}")
        code, report = self.compare(before, after)
        self.assertEqual(code, 1)
        self.assertIn("macro_definitions", report["changes"])

    def test_plain_tex_macro_body_change_is_detected(self):
        before = self.manuscript("before", r"\def\domain{\mathbb{R}}")
        after = self.manuscript("after", r"\def\domain{\mathbb{C}}")
        self.assertIn("macro_definitions", self.compare(before, after)[1]["changes"])

    def test_custom_entrypoint_compare_checks_the_actual_manuscript(self):
        before = self.manuscript("before", r"For $x\geq0$, the bound holds.")
        after = self.manuscript("after", r"For $x\in\mathbb R$, the bound holds.")
        for path in [before, after]:
            (path / "main.tex").rename(path / "article.tex")
            (path / "project.json").write_text(json.dumps({"entrypoint": "article.tex"}))
        self.assertIn("math_expressions", self.compare(before, after)[1]["changes"])

    def test_missing_citation_is_detected(self):
        before = self.manuscript("before", r"As established in \cite{source-a,source-b}.")
        after = self.manuscript("after", r"As established in \cite{source-a}.")
        self.assertEqual(self.compare(before, after)[0], 1)

    def test_same_citation_key_with_changed_bibliography_is_detected(self):
        body = r"Study~\cite{same-key}. \bibliography{references}"
        before = self.manuscript("before", body)
        after = self.manuscript("after", body)
        (before / "references.bib").write_text("@misc{same-key,title={Original source},year={2024}}")
        (after / "references.bib").write_text("@misc{same-key,title={Different source},year={2024}}")
        code, report = self.compare(before, after)
        self.assertEqual(code, 1)
        self.assertIn("bibliography_sources", report["changes"])

    def test_bibliography_relocation_preserves_reference_contents(self):
        before = self.manuscript("before", r"\cite{source}\bibliography{references}")
        after = self.manuscript("after", r"\cite{source}\bibliography{bib/library}")
        (after / "bib").mkdir()
        bib = "@misc{source,title={Provided reference}}"
        (before / "references.bib").write_text(bib)
        (after / "bib/library.bib").write_text(bib)
        self.assertEqual(self.compare(before, after)[0], 0)

    def test_missing_declared_bibliography_is_not_silently_ignored(self):
        path = self.manuscript("missing", r"\cite{source}\bibliography{not-supplied}")
        with self.assertRaises(FileNotFoundError):
            project.protected_content(path)

    def test_template_identifier_digits_do_not_mimic_research_values(self):
        before = self.manuscript("before", r"\bibliographystyle{icml2026} Measured score 84.0.")
        after = self.manuscript("after", r"\bibliographystyle{IEEEtran} Measured score 84.0.")
        self.assertEqual(self.compare(before, after)[0], 0)

    def test_init_refuses_existing_directory_without_touching_it(self):
        target = self.root / "existing"
        target.mkdir()
        user_file = target / "author-notes.txt"
        user_file.write_text("Author's original notes")
        with self.assertRaises(ValueError):
            project.init_project(argparse.Namespace(venue="icml", version=None, mode="submission", output=str(target)))
        self.assertEqual(user_file.read_text(), "Author's original notes")
        self.assertEqual(list(target.iterdir()), [user_file])

    def test_unknown_template_version_creates_nothing(self):
        target = self.root / "unknown"
        with self.assertRaises(ValueError):
            project.init_project(argparse.Namespace(venue="icml", version="2099", mode="submission", output=str(target)))
        self.assertFalse(target.exists())

    def test_packaged_official_asset_matches_generated_project(self):
        target = self.root / "fresh"
        with redirect_stdout(io.StringIO()):
            project.init_project(argparse.Namespace(venue="tai", version=None, mode="submission", output=str(target)))
        metadata = json.loads((target / "project.json").read_text())
        self.assertIn("IEEEtai.cls", metadata["official_files"])
        for name, expected in metadata["official_files"].items():
            self.assertEqual(hashlib.sha256((target / name).read_bytes()).hexdigest(), expected)
        self.assertTrue((target / "sections/body.tex").is_file())

    def test_custom_entrypoint_is_checked_and_stale_pdf_error_is_rejected(self):
        path = self.root / "custom"
        (path / "build").mkdir(parents=True)
        (path / "article.tex").write_text(r"\documentclass{article}\begin{document}Draft.\end{document}")
        (path / "project.json").write_text(json.dumps({"entrypoint": "article.tex"}))
        (path / "build/article.pdf").write_bytes(b"stale PDF placeholder")
        (path / "build/article.log").write_text("Undefined control sequence. Fatal error occurred")
        with redirect_stdout(io.StringIO()) as output:
            code = project.check_project(argparse.Namespace(project=str(path), main=None))
        report = json.loads(output.getvalue())
        self.assertEqual(code, 1)
        self.assertEqual(report["entrypoint"], "article.tex")
        self.assertTrue(any(i["kind"] == "compile-diagnostic" for i in report["issues"]))

    def test_required_ai_statement_is_not_silently_dropped(self):
        path = self.manuscript("missing-statement", "A draft without the required disclosure.")
        (path / "project.json").write_text(json.dumps({"rules": {"required_sections": ["AI use statement"]}}))
        with redirect_stdout(io.StringIO()) as output:
            project.check_project(argparse.Namespace(project=str(path), main=None))
        self.assertTrue(any(i["kind"] == "required-section-missing" for i in json.loads(output.getvalue())["issues"]))

    def test_macro_title_does_not_pass_as_zero_rendered_words(self):
        path = self.manuscript("macro-title", r"\newcommand{\LongTitle}{An expanded title that must be reviewed}\title{\LongTitle}")
        (path / "project.json").write_text(json.dumps({"rules": {"title_max_words": 15}}))
        with redirect_stdout(io.StringIO()) as output:
            project.check_project(argparse.Namespace(project=str(path), main=None))
        report = json.loads(output.getvalue())
        self.assertTrue(any(w["kind"] == "title-word-count-requires-manual-check" for w in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
