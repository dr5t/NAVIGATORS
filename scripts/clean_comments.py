#!/usr/bin/env python3
"""
Navigators Codebase Comment Removal Script

Safely removes comments from:
- Python (.py)
- JavaScript (.js, .cjs, .mjs)
- CSS (.css)
- HTML (.html)
- SQL (.sql)
- Shell (.sh)

Rules:
- Does NOT touch string literals, docstrings, JSON values, SQL queries, or configuration.
- Preserves shebangs (line 1 #!...).
- Preserves newline counts in multiline comment blocks to keep line alignment.
- Validates syntax of Python and JavaScript files after cleaning.
- Preserves all functional code and executable logic.
"""

import os
import re
import sys
import io
import ast
import tokenize
import tempfile
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_DIRS = {
    ".git",
    "venv",
    ".venv",
    "node_modules",
    "vendor",
    ".gemini",
    "docs",
    "data",
    "__pycache__",
    ".pytest_cache",
}

def is_excluded(path: Path) -> bool:
    if path.name == "clean_comments.py":
        return True
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    return False

def clean_python(source: str) -> str:
    tokens = list(tokenize.tokenize(io.BytesIO(source.encode("utf-8")).readline))
    lines = source.splitlines(keepends=True)
    comments = [tok for tok in tokens if tok.type == tokenize.COMMENT]
    
    for tok in reversed(comments):
        start_row, start_col = tok.start
        end_row, end_col = tok.end
        if start_row == 1 and tok.string.startswith("#!"):
            continue
        line_idx = start_row - 1
        line = lines[line_idx]
        prefix = line[:start_col]
        suffix = line[end_col:]
        if prefix.strip() == "" and suffix.strip() == "":
            lines[line_idx] = "\n" if line.endswith("\n") else ""
        else:
            lines[line_idx] = prefix.rstrip() + suffix
            
    cleaned = "".join(lines)
    ast.parse(cleaned)
    return cleaned

from typing import Any, List

def clean_js(code: str) -> str:
    out: List[str] = []
    i = 0
    n = len(code)
    # Stack elements: 'CODE' or dict(type='TEMPLATE') or dict(type='EXPR', depth=1)
    stack: List[Any] = ["CODE"]

    while i < n:
        current = stack[-1]
        c = code[i]
        next_c = code[i+1] if i + 1 < n else ""

        if current == "CODE" or (isinstance(current, dict) and current["type"] == "EXPR"):
            if c == "'" or c == '"':
                quote = c
                out.append(quote)
                i += 1
                while i < n and code[i] != quote:
                    if code[i] == "\\":
                        out.append(code[i])
                        i += 1
                    if i < n:
                        out.append(code[i])
                        i += 1
                if i < n:
                    out.append(code[i])
                    i += 1
                continue

            if c == "`":
                out.append("`")
                stack.append({"type": "TEMPLATE"})
                i += 1
                continue

            if c == "/" and next_c == "/":
                i += 2
                while i < n and code[i] != "\n":
                    i += 1
                if i < n and code[i] == "\n":
                    out.append("\n")
                    i += 1
                continue

            if c == "/" and next_c == "*":
                i += 2
                while i + 1 < n and not (code[i] == "*" and code[i+1] == "/"):
                    if code[i] == "\n":
                        out.append("\n")
                    i += 1
                if i + 1 < n:
                    i += 2
                continue

            if isinstance(current, dict) and current["type"] == "EXPR":
                if c == "{":
                    current["depth"] += 1
                    out.append(c)
                    i += 1
                    continue
                elif c == "}":
                    current["depth"] -= 1
                    if current["depth"] == 0:
                        stack.pop()
                    out.append(c)
                    i += 1
                    continue

            out.append(c)
            i += 1

        elif isinstance(current, dict) and current["type"] == "TEMPLATE":
            if c == "`":
                out.append("`")
                stack.pop()
                i += 1
                continue
            if c == "$" and next_c == "{":
                out.append("${")
                stack.append({"type": "EXPR", "depth": 1})
                i += 2
                continue
            if c == "\\":
                out.append(code[i])
                i += 1
            if i < n:
                out.append(code[i])
                i += 1

    return "".join(out)

def clean_css(css: str) -> str:
    out = []
    i = 0
    n = len(css)
    while i < n:
        c = css[i]
        if c in ("'", '"'):
            quote = c
            start = i
            i += 1
            while i < n and css[i] != quote:
                if css[i] == "\\":
                    i += 1
                i += 1
            if i < n:
                i += 1
            out.append(css[start:i])
            continue
        if c == "/" and i + 1 < n and css[i+1] == "*":
            i += 2
            newlines = 0
            while i + 1 < n and not (css[i] == "*" and css[i+1] == "/"):
                if css[i] == "\n":
                    newlines += 1
                i += 1
            if i + 1 < n:
                i += 2
            out.append("\n" * newlines)
            continue
        out.append(c)
        i += 1
    return "".join(out)

def clean_html(html: str) -> str:
    pattern = re.compile(r"<!--.*?-->", re.DOTALL)
    def repl(m):
        newlines = m.group(0).count("\n")
        return "\n" * newlines
    return pattern.sub(repl, html)

def clean_sql(sql: str) -> str:
    out = []
    i = 0
    n = len(sql)
    while i < n:
        c = sql[i]
        if c == "'":
            start = i
            i += 1
            while i < n:
                if sql[i] == "'":
                    if i + 1 < n and sql[i+1] == "'":
                        i += 2
                        continue
                    else:
                        i += 1
                        break
                i += 1
            out.append(sql[start:i])
            continue
        if c == "-" and i + 1 < n and sql[i+1] == "-":
            i += 2
            while i < n and sql[i] != "\n":
                i += 1
            if i < n and sql[i] == "\n":
                out.append("\n")
                i += 1
            continue
        if c == "/" and i + 1 < n and sql[i+1] == "*":
            i += 2
            newlines = 0
            while i + 1 < n and not (sql[i] == "*" and sql[i+1] == "/"):
                if sql[i] == "\n":
                    newlines += 1
                i += 1
            if i + 1 < n:
                i += 2
            out.append("\n" * newlines)
            continue
        out.append(c)
        i += 1
    return "".join(out)

def clean_shell(sh: str) -> str:
    lines = sh.splitlines(keepends=True)
    out = []
    for idx, line in enumerate(lines):
        if idx == 0 and line.startswith("#!"):
            out.append(line)
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            out.append("\n" if line.endswith("\n") else "")
        else:
            in_single = False
            in_double = False
            cut_idx = None
            for col, ch in enumerate(line):
                if ch == "'" and not in_double:
                    in_single = not in_single
                elif ch == '"' and not in_single:
                    in_double = not in_double
                elif ch == "#" and not in_single and not in_double:
                    cut_idx = col
                    break
            if cut_idx is not None:
                line_part = line[:cut_idx].rstrip()
                if line.endswith("\n"):
                    line_part += "\n"
                out.append(line_part)
            else:
                out.append(line)
    return "".join(out)

def process_file(file_path: Path, dry_run: bool = False) -> bool:
    if is_excluded(file_path):
        return False

    suffix = file_path.suffix.lower()

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False

    new_content = None

    if suffix == ".py":
        try:
            new_content = clean_python(content)
        except Exception as e:
            print(f"Error cleaning Python {file_path}: {e}")
            return False

    elif suffix in (".js", ".cjs", ".mjs"):
        try:
            new_content = clean_js(content)
            # Validate JS syntax
            with tempfile.NamedTemporaryFile(suffix=suffix, mode="w", delete=True) as tmp:
                tmp.write(new_content)
                tmp.flush()
                res = subprocess.run(["node", "--check", tmp.name], capture_output=True, text=True)
                if res.returncode != 0:
                    print(f"Syntax validation failed on JS {file_path}: {res.stderr}")
                    return False
        except Exception as e:
            print(f"Error cleaning JS {file_path}: {e}")
            return False

    elif suffix == ".css":
        new_content = clean_css(content)

    elif suffix == ".html":
        new_content = clean_html(content)

    elif suffix == ".sql":
        new_content = clean_sql(content)

    elif suffix == ".sh":
        new_content = clean_shell(content)

    if new_content is not None and new_content != content:
        if not dry_run:
            file_path.write_text(new_content, encoding="utf-8")
        return True

    return False

def main():
    dry_run = "--dry-run" in sys.argv
    print(f"Starting comment cleanup (dry_run={dry_run})...")
    
    modified = []
    
    for ext in ("*.py", "*.js", "*.cjs", "*.mjs", "*.css", "*.html", "*.sql", "*.sh"):
        for p in ROOT.rglob(ext):
            if process_file(p, dry_run=dry_run):
                modified.append(p)

    print(f"Comment cleanup finished. Modified {len(modified)} files.")
    for p in sorted(modified):
        print(f"  {p.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
