"""
MCP server: pretty resume PDF rendering via local LaTeX.

This is intended to be launched over stdio and called from the app's PDF download
endpoint. It requires a working LaTeX installation that provides `pdflatex`
(e.g., MiKTeX on Windows).
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any

from fastmcp import FastMCP

mcp = FastMCP("career-copilot-resume-pdf")


_LATEX_PREAMBLE = r"""
\documentclass[10.5pt]{article}
\usepackage[margin=0.6in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{parskip}
\usepackage{tabularx}
\usepackage{ragged2e}
\usepackage{setspace}
\usepackage{ifthen}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\setstretch{0.98}
\setlist[itemize]{leftmargin=*, topsep=2pt, itemsep=1.5pt, parsep=0pt, partopsep=0pt}
\definecolor{muted}{RGB}{80,80,80}
\definecolor{rulegray}{RGB}{210,210,210}
\titleformat{\section}{\large\bfseries}{}{0em}{}[\vspace{-0.35em}\color{rulegray}\titlerule]
\titlespacing*{\section}{0pt}{0.6em}{0.4em}
"""

_LATEX_DOC_START = r"\begin{document}"
_LATEX_DOC_END = r"\end{document}"


def _escape_tex(s: str) -> str:
    # Minimal TeX escaping for common resume text.
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def _looks_like_header(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.endswith(":") and len(s) <= 40:
        return True
    if s.isupper() and 3 <= len(s) <= 40:
        return True
    return False


def _split_header(lines: list[str]) -> tuple[str | None, list[str], list[str]]:
    """
    Heuristic:
      - First non-empty line is the name
      - Subsequent non-empty lines until first detected header become contact lines
    """
    cleaned = [ln.rstrip() for ln in lines]
    # find first non-empty
    i = 0
    while i < len(cleaned) and not cleaned[i].strip():
        i += 1
    if i >= len(cleaned):
        return None, [], []
    name = cleaned[i].strip()
    contact: list[str] = []
    i += 1
    while i < len(cleaned):
        ln = cleaned[i].strip()
        if not ln:
            i += 1
            continue
        if _looks_like_header(ln):
            break
        # Stop contact if it looks like role line (short and title-like)
        if len(ln) <= 32 and ("engineer" in ln.lower() or "scientist" in ln.lower()):
            contact.append(ln)
            i += 1
            continue
        contact.append(ln)
        i += 1
        if len(contact) >= 3:
            # keep header tight: role + 2 contact-ish lines max
            break
    rest = cleaned[i:] if i < len(cleaned) else []
    return name, contact, rest


def _to_latex(resume_text: str) -> str:
    """
    Convert plain resume text into a simple, nicer LaTeX layout:
    - Detect section headers (e.g., "Experience", "Skills:")
    - Convert leading '-'/'•' bullets into itemize
    """
    text = (resume_text or "").strip()
    # Remove markdown bold markers that may appear in LLM output.
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    lines = [ln.rstrip() for ln in text.splitlines()]
    name, contact_lines, content_lines = _split_header(lines)

    out: list[str] = []
    in_list = False

    def end_list() -> None:
        nonlocal in_list
        if in_list:
            out.append(r"\end{itemize}")
            in_list = False

    if name:
        out.append(r"{\LARGE\bfseries " + _escape_tex(name) + r"}")
        if contact_lines:
            # Combine into a compact line with separators when possible
            compact = "  |  ".join(_escape_tex(x) for x in contact_lines if x.strip())
            out.append(r"{\small\color{muted} " + compact + r"}")
        out.append(r"\vspace{0.2em}")
        out.append(r"\color{rulegray}\rule{\linewidth}{0.4pt}")
        out.append(r"\vspace{0.3em}")

    for raw in content_lines:
        line = raw.strip()
        if not line:
            end_list()
            out.append("")
            continue

        if _looks_like_header(line):
            end_list()
            hdr = line.rstrip(":").strip()
            out.append(rf"\section*{{{_escape_tex(hdr)}}}")
            continue

        bullet_match = re.match(r"^([-*]|•)\s+(.*)$", line)
        if bullet_match:
            if not in_list:
                out.append(r"\begin{itemize}")
                in_list = True
            item = bullet_match.group(2).strip()
            out.append(rf"\item {_escape_tex(item)}")
            continue

        end_list()
        out.append(_escape_tex(line))

    end_list()

    body = "\n".join(out).strip()
    if not body:
        body = r"\section*{Resume}\n(Empty)"

    return "\n".join(
        [
            _LATEX_PREAMBLE,
            _LATEX_DOC_START,
            body,
            _LATEX_DOC_END,
        ]
    )


def _compile_pdflatex(tex: str) -> bytes:
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        raise RuntimeError(
            "pdflatex not found. Install a LaTeX distribution (MiKTeX on Windows) "
            "or disable MCP PDF rendering."
        )

    with tempfile.TemporaryDirectory(prefix="career-copilot-resume-") as td:
        tex_path = os.path.join(td, "resume.tex")
        pdf_path = os.path.join(td, "resume.pdf")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex)

        # Run twice for layout stability.
        for _ in range(2):
            proc = subprocess.run(
                [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path],
                cwd=td,
                capture_output=True,
                text=True,
                timeout=45,
            )
            if proc.returncode != 0:
                tail = (proc.stdout or "")[-2000:] + "\n" + (proc.stderr or "")[-2000:]
                raise RuntimeError(f"pdflatex failed:\n{tail}")

        with open(pdf_path, "rb") as f:
            return f.read()


@mcp.tool()
def render_resume_pdf(resume_text: str) -> dict[str, Any]:
    """
    Render resume text to a "pretty" LaTeX PDF.

    Returns:
      {"pdf_base64": "<base64-encoded pdf bytes>"}.
    """
    tex = _to_latex(resume_text)
    pdf = _compile_pdflatex(tex)
    return {"pdf_base64": base64.b64encode(pdf).decode("ascii")}


if __name__ == "__main__":
    mcp.run()

