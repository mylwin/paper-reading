#!/usr/bin/env python3
"""Portable official-template assembly, compilation and bounded static audits.

No third-party Python dependency. This script never writes into an existing
destination on init, never edits vendor files, and never sends manuscripts online.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bibliography_paths(project, chunks):
    paths = set()
    for _, text in chunks:
        for group in re.findall(r"\\bibliography\s*\{([^{}]+)\}", text):
            for value in group.split(","):
                value = value.strip()
                if "\\" not in value:
                    paths.add(bounded_path(project, value if value.endswith(".bib") else value + ".bib"))
        for value in re.findall(r"\\addbibresource(?:\[[^]]*\])?\s*\{([^{}]+)\}", text):
            if "\\" not in value:
                paths.add(bounded_path(project, value))
    return sorted(paths)


def project_inputs(project, main):
    """Track reachable sources, literal bibliographies and local TeX recorder inputs.

    External runtime dependencies and generated build files are excluded. This is
    a freshness audit, not arbitrary TeX expansion or a reproducible-build proof.
    """
    project = Path(project).resolve()
    chunks = tex_content(project, main)
    paths = {bounded_path(project, name) for name, _ in chunks}
    paths.update(bibliography_paths(project, chunks))
    recorder = project / "build" / (Path(main).stem + ".fls")
    if recorder.exists():
        for line in recorder.read_text(errors="replace").splitlines():
            if not line.startswith("INPUT "):
                continue
            candidate = Path(line[6:])
            candidate = (project / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
            if candidate.is_relative_to(project) and not candidate.is_relative_to(project / "build"):
                paths.add(candidate)
    return {str(p.relative_to(project)): digest(p) if p.is_file() else None for p in sorted(paths)}


def bounded_path(base, relative):
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f"Path escapes resource directory: {relative}")
    return path


def resolve_spec(venue, version=None):
    registry = read_json(ROOT / "registry.json")
    key = venue.lower().replace("ieee ", "").strip()
    if key not in registry["venues"]:
        raise ValueError(f"Unknown venue {venue}; choose {', '.join(registry['venues'])}")
    entry = registry["venues"][key]
    version = version or entry["latest"]
    if version not in entry["versions"]:
        raise ValueError(f"No verified {key} {version} template; available: {', '.join(entry['versions'])}")
    path = bounded_path(ROOT, entry["versions"][version])
    return path, read_json(path)


def init_project(args):
    spec_path, spec = resolve_spec(args.venue, args.version)
    mode = spec["modes"].get(args.mode)
    if not mode or not mode.get("available"):
        raise ValueError(f"{args.venue} {spec['version']} {args.mode} unavailable: {spec.get('limitations', [])}")
    target = Path(args.output).expanduser().resolve()
    if target.exists():
        raise ValueError(f"Destination already exists; refusing overwrite: {target}")
    source = bounded_path(spec_path.parent, mode["directory"])
    if not (source / "main.tex").is_file():
        raise ValueError("Packaged project entrypoint is missing")
    # Verify the package BEFORE copying any files or creating the destination.
    for relative, expected in spec["official_files"].items():
        path = bounded_path(source, relative)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Official asset missing or changed: {relative}")
    shutil.copytree(source, target, symlinks=False)
    write_json(target / "project.json", {
        "venue": spec["venue"], "version": spec["version"], "mode": args.mode,
        "template_source": spec["template_source"], "verified_on": spec["verified_on"],
        "official_files": spec["official_files"], "constraints": mode,
        "rules": spec.get("rules", {}),
        "entrypoint": "main.tex", "status": "draft-with-author-inputs-pending",
        "scope": "Template is locally verified; future venue policies need rechecking.",
    })
    print(json.dumps({"created": str(target), "venue": spec["venue"], "version": spec["version"], "mode": args.mode}, ensure_ascii=False))


def compile_project(args):
    project = Path(args.project).expanduser().resolve()
    metadata_path = project / "project.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    entrypoint = args.main or metadata.get("entrypoint", "main.tex")
    main = bounded_path(project, entrypoint)
    if not main.is_file():
        raise ValueError(f"Missing entrypoint: {main}")
    latexmk = shutil.which("latexmk")
    if not latexmk:
        raise ValueError("latexmk is unavailable; diagnose the existing TeX runtime; do not auto-install")
    build = project / "build"
    build.mkdir(exist_ok=True)
    selected_engine = args.engine or metadata.get("constraints", {}).get("engine", "pdflatex")
    engine = "-xelatex" if selected_engine == "xelatex" else "-lualatex" if selected_engine == "lualatex" else "-pdf"
    command = [latexmk, "-norc", engine, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "-latexoption=-no-shell-escape", f"-outdir={build}", str(main)]
    proc = subprocess.run(command, cwd=project, capture_output=True, text=True, errors="replace", timeout=240)
    pdf = build / main.with_suffix(".pdf").name
    result = {"command": command, "entrypoint": str(main.relative_to(project)), "checked_at": datetime.now(timezone.utc).isoformat(), "exit_code": proc.returncode, "pdf": str(pdf) if pdf.exists() else None, "log_tail": (proc.stdout + proc.stderr)[-6000:]}
    if proc.returncode == 0 and pdf.exists():
        result["input_sha256"] = project_inputs(project, entrypoint)
        result["pdf_sha256"] = digest(pdf)
    write_json(build / f"{main.stem}-compile.json", result)
    write_json(build / "compile.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "log_tail"}, ensure_ascii=False))
    if proc.returncode:
        print(result["log_tail"], file=sys.stderr)
    return proc.returncode


def tex_content(project, main="main.tex"):
    """Read reachable TeX inputs; not a complete expansion of arbitrary TeX macros."""
    project = Path(project).resolve()
    entry = bounded_path(project, main)
    if not entry.exists():
        raise ValueError(f"Missing entrypoint: {entry}")
    seen, chunks = set(), []
    def visit(path):
        path = path.resolve()
        if path in seen:
            return
        if not path.is_relative_to(project.resolve()):
            raise ValueError(f"Input outside project: {path}")
        if not path.exists():
            raise ValueError(f"Missing input: {path}")
        seen.add(path)
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"(?<!\\)%[^\n]*", "", text)
        chunks.append((str(path.relative_to(project)), text))
        for value in re.findall(r"\\(?:input|include)\s*\{([^{}]+)\}", text):
            if "\\" in value:
                continue
            candidate = project / value
            if not candidate.suffix:
                candidate = candidate.with_suffix(".tex")
            visit(candidate)
    visit(entry)
    return chunks


def check_project(args):
    project = Path(args.project).resolve()
    metadata_path = project / "project.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    issues, warnings = [], []
    for relative, expected in metadata.get("official_files", {}).items():
        path = bounded_path(project, relative)
        if not path.exists() or digest(path) != expected:
            issues.append({"kind": "official-asset-changed", "file": relative})
    entrypoint = args.main or metadata.get("entrypoint", "main.tex")
    chunks = tex_content(project, entrypoint)
    text = "\n".join(value for _, value in chunks)
    if re.search(r"(?:AUTHOR INPUT PENDING|AUTHOR INFORMATION PENDING|DRAFT PLACEHOLDER|\\AuthorPending\b)", text):
        issues.append({"kind": "author-input-pending", "note": "Skeleton is not a finished manuscript"})
    for token in (r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\bgeometry\b[^}]*\}", r"\\(?:textwidth|textheight|oddsidemargin|evensidemargin)\s*=", r"\\(?:addtolength|setlength)\{\\(?:textwidth|textheight|baselineskip)\}"):
        if re.search(token, text):
            issues.append({"kind": "manual-layout-override", "pattern": token})
    build = project / "build"
    stem = Path(entrypoint).stem
    log = build / f"{stem}.log"
    pdf = build / f"{stem}.pdf"
    if not pdf.exists():
        issues.append({"kind": "not-compiled"})
    compile_record = build / f"{stem}-compile.json"
    record = read_json(compile_record) if compile_record.exists() else {}
    if record.get("exit_code"):
        issues.append({"kind": "last-compilation-failed"})
    if pdf.exists():
        current_inputs = project_inputs(project, entrypoint)
        if "input_sha256" in record:
            changed = [name for name in sorted(set(current_inputs) | set(record["input_sha256"])) if current_inputs.get(name) != record["input_sha256"].get(name)]
            if changed:
                issues.append({"kind": "compiled-inputs-changed", "files": changed, "note": "Recompile before reviewing the PDF"})
        else:
            newer = [name for name in current_inputs if (project / name).exists() and (project / name).stat().st_mtime_ns > pdf.stat().st_mtime_ns]
            if newer:
                issues.append({"kind": "source-newer-than-pdf", "files": newer})
            warnings.append({"kind": "compile-freshness-unverified", "note": "Older build has no input fingerprints; compile again to record them"})
        if record.get("pdf_sha256") and digest(pdf) != record["pdf_sha256"]:
            issues.append({"kind": "compiled-pdf-changed"})
    if log.exists():
        log_text = log.read_text(errors="replace")
        for needle in ("undefined references", "undefined citations", "LaTeX Error", "Missing character", "Undefined control sequence", "Fatal error occurred", "Emergency stop"):
            if needle in log_text:
                issues.append({"kind": "compile-diagnostic", "message": needle})
        if "Overfull" in log_text:
            warnings.append({"kind": "overfull", "note": "Inspect actual page rendering"})
    constraints = metadata.get("constraints", {})
    total_pages = None
    pdfinfo = shutil.which("pdfinfo")
    if pdf.exists() and pdfinfo:
        info = subprocess.run([pdfinfo, str(pdf)], capture_output=True, text=True, errors="replace")
        match = re.search(r"Pages:\s*(\d+)", info.stdout)
        if match:
            total_pages = int(match.group(1))
    full_limit = constraints.get("full_manuscript_page_limit")
    if full_limit is not None and total_pages is not None and total_pages > full_limit:
        issues.append({"kind": "full-manuscript-page-limit", "pages": total_pages, "limit": full_limit})
    size_limit = constraints.get("max_pdf_megabytes")
    if pdf.exists() and size_limit and pdf.stat().st_size > size_limit * 1_000_000:
        issues.append({"kind": "pdf-file-size-limit", "bytes": pdf.stat().st_size, "limit_MB": size_limit})
    aux = build / f"{stem}.aux"
    main_pages = None
    if aux.exists():
        match = re.search(r"\\newlabel\{pw:main-end\}\{\{[^}]*\}\{(\d+)\}", aux.read_text(errors="replace"))
        if match:
            main_pages = int(match.group(1))
    limit = constraints.get("main_page_limit")
    if limit is not None and main_pages is not None and main_pages > limit:
        issues.append({"kind": "main-page-limit", "pages": main_pages, "limit": limit})
    if limit is not None and main_pages is None:
        warnings.append({"kind": "page-count-unknown", "note": "Need correctly placed main-text boundary and manual count"})
    if constraints.get("anonymous") is None:
        warnings.append({"kind": "anonymity-unverified", "note": "Confirm the selected venue and stage policy"})
    if constraints.get("page_limit_status") != "verified":
        warnings.append({"kind": "page-policy-unverified"})
    # These are conservative source-level estimates, not TeX macro expansion.
    rules = metadata.get("rules", {})
    headings = {re.sub(r"\s+", " ", h).strip().casefold() for h in re.findall(r"\\(?:sub)*section\*?(?:\[[^]]*\])?\s*\{([^{}]+)\}", text)}
    for section in rules.get("required_sections", []):
        if section.casefold() not in headings:
            issues.append({"kind": "required-section-missing", "section": section})
    def words(value):
        value = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", value)
        return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", value))
    estimates = {}
    for environment, key in (("abstract", "abstract_max_words"), ("IEEEImpStatement", "impact_statement_words")):
        match = re.search(r"\\begin\{" + environment + r"\}(.*?)\\end\{" + environment + r"\}", text, re.S)
        if key in rules:
            if not match:
                issues.append({"kind": "required-environment-missing", "environment": environment})
            else:
                count = words(match.group(1))
                estimates[environment] = count
                bound = rules[key]
                low, high = bound if isinstance(bound, list) else (0, bound)
                if not low <= count <= high:
                    issues.append({"kind": "estimated-word-limit", "environment": environment, "words": count, "range": [low, high]})
    if "keywords_min" in rules:
        match = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", text, re.S)
        count = len([x for x in match.group(1).split(",") if x.strip()]) if match else 0
        if not rules["keywords_min"] <= count <= rules["keywords_max"]:
            issues.append({"kind": "keyword-count", "count": count})
    if "title_max_words" in rules:
        # Balanced title arguments can contain macros and need manual checking.
        match = re.search(r"\\title(?:\[[^]]*\])?\{([^{}]*)\}", text)
        if match and re.search(r"\\[A-Za-z@]+", match.group(1)):
            warnings.append({"kind": "title-word-count-requires-manual-check", "note": "The title contains a macro; count its rendered expansion"})
        elif match and words(match.group(1)) > rules["title_max_words"]:
            issues.append({"kind": "estimated-title-word-limit", "words": words(match.group(1))})
        elif not match:
            warnings.append({"kind": "title-word-count-requires-manual-check"})
    if rules:
        warnings.append({"kind": "source-word-count-estimate", "note": "TeX macros may change rendered counts; manually verify"})
    pdftotext = shutil.which("pdftotext")
    if pdf.exists() and pdftotext:
        proc = subprocess.run([pdftotext, str(pdf), "-"], capture_output=True, text=True, errors="replace")
        if proc.returncode:
            issues.append({"kind": "pdf-text-extraction-failed"})
        elif constraints.get("anonymous") and "AUTHOR INFORMATION PENDING" in proc.stdout:
            issues.append({"kind": "author-block-visible-in-anonymous-mode"})
        elif constraints.get("anonymous"):
            rendered = re.sub(r"\s+", " ", proc.stdout)
            for author in re.findall(r"\\author(?:\[[^]]*\])?\s*\{([^{}]+)\}", text):
                author = re.sub(r"\s+", " ", author).strip()
                if "\\" not in author and author.casefold() not in {"anonymous", "anonymous submission", "anonymous authors"} and len(author.split()) >= 2 and author in rendered:
                    issues.append({"kind": "author-block-visible-in-anonymous-mode", "note": "A literal author field is visible in the rendered PDF"})
            warnings.append({"kind": "manual-anonymity-review-required", "note": "Literal author check does not audit acknowledgments, funding, links, assets or PDF metadata"})
    report = {"project": str(project), "entrypoint": entrypoint, "issues": issues, "warnings": warnings, "main_pages": main_pages, "total_pages": total_pages, "word_count_estimates": estimates, "limits": constraints, "static_check_passed": not issues, "anonymity_check": "partial-literal-author-check; requires-full-review", "visual_check": "not-performed-by-this-command", "official_system_check": "not-performed", "semantic_fidelity": "requires-review"}
    if build.exists():
        write_json(build / "format-check.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if issues else 0


def protected_content(project, main=None):
    metadata = read_json(project / "project.json") if (project / "project.json").exists() else {}
    chunks = tex_content(project, main or metadata.get("entrypoint", "main.tex"))
    text = "\n".join(value for _, value in chunks)
    citation = Counter(re.findall(r"\\(?:cite\w*|label|(?:eq|page|auto|c)?ref)\*?(?:\[[^]]*\])*\{([^{}]+)\}", text))
    macros = Counter(re.findall(r"\\(?:newcommand|renewcommand|providecommand|DeclareMathOperator)\*?\{?\\([A-Za-z@]+)", text))
    macros.update(re.findall(r"\\(?:def|gdef|edef|xdef)\s*\\([A-Za-z@]+)", text))
    # Collect full braced bodies; preserving a macro name alone is insufficient.
    definitions = []
    definition_start = r"\\(?:newcommand|renewcommand|providecommand|DeclareMathOperator)\*?\s*(?:\{\\[A-Za-z@]+\}|\\[A-Za-z@]+)(?:\s*\[[^]]*\]){0,2}\s*\{|\\(?:def|gdef|edef|xdef)\s*\\[A-Za-z@]+[^{}]*\{"
    for match in re.finditer(definition_start, text):
        depth, end = 1, match.end()
        while end < len(text) and depth:
            if text[end] in "{}":
                backslashes, cursor = 0, end - 1
                while cursor >= 0 and text[cursor] == "\\":
                    backslashes += 1
                    cursor -= 1
                if backslashes % 2 == 0:
                    depth += 1 if text[end] == "{" else -1
            end += 1
        definitions.append(re.sub(r"\s+", "", text[match.start():end]))
    maths = re.findall(r"\$\$.*?\$\$|(?<!\\)\$(?!\$).*?(?<!\\)\$|\\\[.*?\\\]|\\\(.*?\\\)|\\begin\{(?:equation|align|gather|multline)\*?\}.*?\\end\{(?:equation|align|gather|multline)\*?\}", text, re.S)
    maths = Counter(re.sub(r"\s+", "", value) for value in maths)
    numbers = Counter(re.findall(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text))
    bibliographies = Counter(re.sub(r"\s+", " ", p.read_text(encoding="utf-8")).strip() for p in bibliography_paths(project, chunks))
    return {"citation_and_labels": citation, "macro_names": macros, "macro_definitions": Counter(definitions), "math_expressions": maths, "numeric_tokens": numbers, "bibliography_sources": bibliographies}


def compare_projects(args):
    before = protected_content(Path(args.before).resolve(), getattr(args, "before_main", None))
    after = protected_content(Path(args.after).resolve(), getattr(args, "after_main", None))
    changes = {}
    for key in before:
        removed, added = before[key] - after[key], after[key] - before[key]
        if removed or added:
            changes[key] = {"removed": dict(removed), "added": dict(added)}
    report = {"changes": changes, "requires_semantic_review": True, "limitations": "Static lexical audit; macro expansion, arbitrary TeX constructs and scientific meaning are not validated. Numeric changes may be harmless bibliography/format changes and need context."}
    if args.report:
        path = Path(args.report).resolve()
        if path.exists():
            raise ValueError(f"Refusing overwrite: {path}")
        write_json(path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if changes else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    init = sub.add_parser("init")
    init.add_argument("--venue", required=True)
    init.add_argument("--version")
    init.add_argument("--mode", choices=["submission", "final"], default="submission")
    init.add_argument("--output", required=True)
    compile_p = sub.add_parser("compile")
    compile_p.add_argument("--project", required=True)
    compile_p.add_argument("--main", help="Override project.json entrypoint; otherwise main.tex")
    compile_p.add_argument("--engine", choices=["pdflatex", "xelatex", "lualatex"], help="Override template engine; otherwise pdflatex")
    check = sub.add_parser("check")
    check.add_argument("--project", required=True)
    check.add_argument("--main", help="Override project.json entrypoint")
    compare = sub.add_parser("compare")
    compare.add_argument("--before", required=True)
    compare.add_argument("--after", required=True)
    compare.add_argument("--before-main", help="Original project TeX entrypoint; defaults to project.json then main.tex")
    compare.add_argument("--after-main", help="Revised project TeX entrypoint; defaults to project.json then main.tex")
    compare.add_argument("--report")
    args = parser.parse_args()
    try:
        if args.command == "list":
            registry = read_json(ROOT / "registry.json")
            print(json.dumps(registry, ensure_ascii=False, indent=2))
            return 0
        function = {"init": init_project, "compile": compile_project, "check": check_project, "compare": compare_projects}[args.command]
        return function(args) or 0
    except (ValueError, OSError, KeyError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
