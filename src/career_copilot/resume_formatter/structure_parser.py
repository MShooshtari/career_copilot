"""
Extract visual style profile from a resume PDF using PyMuPDF.
Also provides shared data classes and plain-text resume parsing
used by the PDF and DOCX builders.
"""

from __future__ import annotations

import json
import re
from typing import Iterator
from dataclasses import asdict, dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_SECTIONS: set[str] = {
    "summary",
    "professional summary",
    "executive summary",
    "objective",
    "profile",
    "about me",
    "experience",
    "work experience",
    "professional experience",
    "employment history",
    "employment",
    "career history",
    "education",
    "academic background",
    "academic history",
    "skills",
    "technical skills",
    "core competencies",
    "key skills",
    "competencies",
    "projects",
    "personal projects",
    "open source",
    "side projects",
    "certifications",
    "certificates",
    "licenses",
    "awards",
    "achievements",
    "honors",
    "languages",
    "volunteer",
    "volunteering",
    "volunteer experience",
    "publications",
    "references",
    "contact",
    "contact information",
    "interests",
    "hobbies",
}

BULLET_CHARS: set[str] = {"•", "·", "○", "▪", "▸", "›", "◦", "▷"}

# Maps PDF/system font names to a reportlab-compatible family string
_FONT_FAMILY_MAP: dict[str, str] = {
    "helvetica": "Helvetica",
    "arial": "Helvetica",
    "calibri": "Helvetica",
    "myriad": "Helvetica",
    "gill": "Helvetica",
    "verdana": "Helvetica",
    "tahoma": "Helvetica",
    "trebuchet": "Helvetica",
    "times": "Times-Roman",
    "georgia": "Times-Roman",
    "garamond": "Times-Roman",
    "palatino": "Times-Roman",
    "cambria": "Times-Roman",
    "courier": "Courier",
}


# ---------------------------------------------------------------------------
# Shared data classes
# ---------------------------------------------------------------------------


@dataclass
class StyleProfile:
    """Captures the visual style of the original resume."""

    name_font_size: float = 18.0
    name_bold: bool = True
    header_font_size: float = 13.0
    header_bold: bool = True
    header_all_caps: bool = False
    body_font_size: float = 11.0
    contact_font_size: float = 10.0
    base_font_family: str = "Helvetica"
    margin_left: float = 50.0
    margin_top: float = 50.0
    sections: list = field(default_factory=list)
    bullet_char: str = "•"
    line_spacing: float = 1.2
    has_header_block: bool = True
    # Alignment: "left" | "center"
    name_align: str = "left"
    # Colors as hex strings e.g. "#2c3e50"
    name_color: str = "#000000"
    header_color: str = "#000000"
    body_color: str = "#000000"
    # Raw font family name from the PDF (e.g. "Calibri") for system font embedding
    raw_font_name: str = ""
    # Bold phrases at body size (e.g. company names, job titles) for post-processing
    bold_phrases: list = field(default_factory=list)
    # Whether the original resume has horizontal rules under section headers
    has_section_rule: bool = False
    # Separator line properties (color hex, thickness in pts, Word border val)
    section_rule_color: str = "#000000"
    section_rule_thickness: float = 0.5
    section_rule_style: str = "single"

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "StyleProfile":
        data = json.loads(s)
        return cls(**data)

    @classmethod
    def default(cls) -> "StyleProfile":
        return cls()


_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "nor", "so", "yet", "for",
    "in", "on", "at", "to", "of", "by", "as", "is", "was", "are",
    "were", "be", "been", "with", "from", "that", "this", "which",
    "it", "its", "not", "no", "if", "up", "do", "did", "has", "have",
    "had", "may", "can", "will", "all", "both", "each", "per",
})


def _extract_bold_phrases_for_line(line_spans: list[dict], body_size: float) -> list[str]:
    """Return bold phrases for a single line (list of spans), filtered and sorted longest-first."""
    seen: set[str] = set()
    phrases: list[str] = []
    for s in line_spans:
        if not s.get("bold"):
            continue
        if abs(s["size"] - body_size) > 1.0:
            continue
        text = s["text"].strip()
        lower = text.lower().rstrip(":")
        if len(text) < 2 or lower in KNOWN_SECTIONS:
            continue
        if text not in seen:
            seen.add(text)
            phrases.append(text)
    return sorted(phrases, key=len, reverse=True)


def _extract_bold_phrases_per_line(
    lines_data: list[list[dict]], body_size: float
) -> list[list[str]]:
    """
    Return per-line bold phrase map: [[line_text, phrase1, phrase2, ...], ...]
    lines_data: list of lines, each line is a list of span/run dicts with text/size/bold keys.
    Only includes lines that actually have bold phrases.
    """
    result = []
    for line_spans in lines_data:
        phrases = _extract_bold_phrases_for_line(line_spans, body_size)
        if phrases:
            line_text = " ".join(s["text"] for s in line_spans)
            result.append([line_text] + phrases)
    return result


def _apply_phrase_bold(line: str, phrase: str) -> str:
    """Wrap *phrase* in ** within *line*, skipping segments already wrapped."""
    parts = re.split(r"(\*\*.+?\*\*)", line)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # already bold
            out.append(part)
        else:
            out.append(re.sub(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", lambda m: f"**{m.group(0)}**", part, flags=re.IGNORECASE))
    return "".join(out)


def _line_words(s: str) -> frozenset[str]:
    return frozenset(re.findall(r"\w+", s.lower()))


def apply_original_bold(text: str, bold_phrases: list) -> str:
    """
    Post-process LLM-generated resume text: wrap phrases bold in the original with **
    markers, but only on lines that closely match the original line where that phrase
    appeared.  Skips the name line and section headers.
    """
    if not bold_phrases:
        return text

    # Build per-line lookup: [(orig_word_set, [phrase, ...]), ...]
    orig_entries: list[tuple[frozenset[str], list[str]]] = []
    for entry in bold_phrases:
        if isinstance(entry, list) and len(entry) >= 2:
            orig_entries.append((_line_words(entry[0]), entry[1:]))
        elif isinstance(entry, str):
            # Legacy flat format — treat each phrase as its own "line"
            orig_entries.append((_line_words(entry), [entry]))

    lines = text.splitlines()
    result = []
    for i, line in enumerate(lines):
        if i == 0:
            result.append(line)
            continue
        stripped = line.strip().rstrip(":")
        if stripped.lower() in KNOWN_SECTIONS or (
            stripped.isupper() and 1 < len(stripped) < 45
        ):
            result.append(line)
            continue

        line_wds = _line_words(stripped)
        if not line_wds:
            result.append(line)
            continue

        # Find the original line with the highest Jaccard similarity
        best_phrases: list[str] = []
        best_score = 0.0
        for orig_wds, phrases in orig_entries:
            union = len(line_wds | orig_wds)
            if union == 0:
                continue
            score = len(line_wds & orig_wds) / union
            if score > best_score:
                best_score = score
                best_phrases = phrases

        # Only apply bold if there's meaningful overlap with the matched original line
        if best_score >= 0.5 and best_phrases:
            for phrase in best_phrases:
                line = _apply_phrase_bold(line, phrase)

        result.append(line)
    return "\n".join(result)


def split_inline_bold(text: str) -> list[tuple[str, bool]]:
    """Split a line containing **markers** into (segment, is_bold) pairs.

    Example: 'Worked at **Acme Corp** as **Engineer**'
      → [('Worked at ', False), ('Acme Corp', True), (' as ', False), ('Engineer', True)]
    """
    parts = re.split(r"\*\*(.+?)\*\*", text)
    return [(part, bool(i % 2)) for i, part in enumerate(parts) if part]


@dataclass
class ResumeElement:
    """A single structural element of a plain-text resume."""

    kind: str  # "name" | "contact" | "section_header" | "bullet" | "body" | "blank"
    text: str


# ---------------------------------------------------------------------------
# Plain-text resume parser (shared by PDF and DOCX builders)
# ---------------------------------------------------------------------------


_CONTACT_INFO_RE = re.compile(
    r"@|https?://|www\.|linkedin\.|github\.|\.com\b|\b\d{3}[-.\s]\d{3}|\(\d{3}\)"
)


def _is_contact_info_line(line: str) -> bool:
    """Return True if the line looks like a contact detail (email, URL, phone, etc.)."""
    return bool(_CONTACT_INFO_RE.search(line))


def parse_resume_text(text: str, profile: StyleProfile) -> list[ResumeElement]:
    """
    Parse a plain-text resume into a list of ResumeElements.

    Detection order:
      1. First non-empty line → name
      2. Lines before the first section header → contact info
      3. Lines matching known section names → section header
      4. Lines starting with bullet chars or "- " / "* " → bullet
      5. Everything else → body
    """
    elements: list[ResumeElement] = []
    lines = text.strip().splitlines()
    if not lines:
        return elements

    section_names = {s.lower().rstrip(":") for s in profile.sections}
    section_names.update(KNOWN_SECTIONS)

    name_found = False
    in_contact_block = True  # true until first section header

    for line in lines:
        stripped = line.strip()

        if not stripped:
            elements.append(ResumeElement("blank", ""))
            continue

        text_lower = stripped.rstrip(":").lower()

        # Section header detection
        is_section = text_lower in section_names or (
            profile.header_all_caps and stripped.isupper() and len(stripped) < 45
        )

        if not name_found:
            elements.append(ResumeElement("name", stripped))
            name_found = True
            continue

        if is_section:
            in_contact_block = False
            header_text = stripped.upper() if profile.header_all_caps else stripped.rstrip(":")
            elements.append(ResumeElement("section_header", header_text))
            continue

        if in_contact_block:
            if _is_contact_info_line(stripped):
                elements.append(ResumeElement("contact", stripped))
            elif len(stripped) <= 80:
                # Short non-contact lines in header zone → title/tagline
                plain = re.sub(r"\*\*", "", stripped)
                elements.append(ResumeElement("tagline", plain))
            else:
                # Long lines signal end of header zone → body
                in_contact_block = False
                elements.append(ResumeElement("body", stripped))
            continue

        # Bullet detection
        if stripped and stripped[0] in BULLET_CHARS:
            elements.append(ResumeElement("bullet", stripped[1:].lstrip()))
            continue

        if len(stripped) > 2 and stripped[0] in "-*" and stripped[1] == " ":
            elements.append(ResumeElement("bullet", stripped[2:]))
            continue

        elements.append(ResumeElement("body", stripped))

    return elements


# ---------------------------------------------------------------------------
# PDF structure extractor
# ---------------------------------------------------------------------------


def _color_to_hex(color_int: int) -> str:
    """Convert a pymupdf integer color (0xRRGGBB) to a CSS hex string."""
    r = (color_int >> 16) & 0xFF
    g = (color_int >> 8) & 0xFF
    b = color_int & 0xFF
    return f"#{r:02x}{g:02x}{b:02x}"


def _base_font_name(font_name: str) -> str:
    """Strip subset prefix and weight/style suffixes to get the bare family name.

    Examples:
      'BCDEEE+Calibri-Bold' → 'Calibri'
      'Calibri-Bold'        → 'Calibri'
      'ArialMT'             → 'Arial'
      'Arial-BoldMT'        → 'Arial'
    """
    # Drop subset prefix e.g. "BCDEEE+Calibri"
    if "+" in font_name:
        font_name = font_name.split("+")[-1]
    # Take everything before the first dash
    base = font_name.split("-")[0].strip()
    # Strip trailing "MT" or "PS"
    for suffix in ("MT", "PS"):
        if base.upper().endswith(suffix) and len(base) > len(suffix):
            base = base[: -len(suffix)].strip()
    return base


def _normalize_font_family(font_name: str) -> str:
    lower = font_name.lower()
    for key, mapped in _FONT_FAMILY_MAP.items():
        if key in lower:
            return mapped
    return "Helvetica"


def _is_bold(font_name: str, flags: int) -> bool:
    return bool(flags & 16) or "bold" in font_name.lower()


def _find_body_size(spans: list[dict]) -> float:
    """Return the most common font size (body text)."""
    from collections import Counter

    rounded = [round(s["size"] * 2) / 2 for s in spans]
    if not rounded:
        return 11.0
    return Counter(rounded).most_common(1)[0][0]


def _find_header_size(spans: list[dict], name_size: float, body_size: float) -> float:
    """Return the font size used for section headers (between name and body)."""
    sizes = sorted({round(s["size"] * 2) / 2 for s in spans}, reverse=True)
    candidates = [s for s in sizes if body_size < s < name_size - 0.5]
    if candidates:
        return candidates[-1]
    # Headers may be same size as body but bold — return body size
    return body_size


def _extract_sections(
    spans: list[dict], header_size: float, body_size: float
) -> list[str]:
    sections: list[str] = []
    for span in spans:
        text_lower = span["text"].strip().rstrip(":").lower()
        is_header_sized = abs(span["size"] - header_size) < 1.0 or (
            span["bold"] and abs(span["size"] - body_size) < 1.0
        )
        if is_header_sized and text_lower in KNOWN_SECTIONS:
            display = span["text"].strip().rstrip(":")
            if display not in sections:
                sections.append(display)
    return sections


def parse_resume_structure(pdf_bytes: bytes) -> StyleProfile:
    """
    Analyse a resume PDF and return a StyleProfile capturing its visual style.
    Falls back to StyleProfile.default() if the PDF has no extractable text.
    """
    try:
        import pymupdf
    except ImportError:
        return StyleProfile.default()

    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return StyleProfile.default()

    spans: list[dict] = []
    pdf_lines_data: list[list[dict]] = []
    page = doc[0]
    try:
        blocks = page.get_text("dict")["blocks"]
    except Exception:
        doc.close()
        return StyleProfile.default()

    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            line_span_list: list[dict] = []
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                entry = {
                    "text": text,
                    "size": float(span.get("size", 11.0)),
                    "font": span.get("font", "Helvetica"),
                    "flags": int(span.get("flags", 0)),
                    "bold": _is_bold(
                        span.get("font", ""), int(span.get("flags", 0))
                    ),
                    "bbox": span.get("bbox", (50.0, 50.0, 500.0, 60.0)),
                    "color": int(span.get("color", 0)),
                }
                spans.append(entry)
                line_span_list.append(entry)
            if line_span_list:
                pdf_lines_data.append(line_span_list)
    page_width = page.rect.width
    doc.close()

    if not spans:
        return StyleProfile.default()

    # Font size roles
    sizes = sorted({s["size"] for s in spans}, reverse=True)
    name_size = sizes[0]
    body_size = _find_body_size(spans)
    header_size = _find_header_size(spans, name_size, body_size)
    contact_size = min(sizes) if len(sizes) > 2 else max(body_size - 1.0, 8.0)

    # Base font family (from most common body-sized spans)
    body_spans = [s for s in spans if abs(s["size"] - body_size) < 1.0]
    base_font = "Helvetica"
    raw_font = ""
    if body_spans:
        raw_font = _base_font_name(body_spans[0]["font"])
        base_font = _normalize_font_family(body_spans[0]["font"])

    # Name styling
    name_spans = [s for s in spans if abs(s["size"] - name_size) < 0.5]
    name_bold = any(s["bold"] for s in name_spans) if name_spans else True

    # Header styling
    header_spans = [s for s in spans if abs(s["size"] - header_size) < 0.5]
    header_bold = any(s["bold"] for s in header_spans) if header_spans else True
    header_all_caps = (
        bool(header_spans)
        and all(s["text"].isupper() for s in header_spans[:6] if len(s["text"]) > 2)
    )

    # Sections in order
    sections = _extract_sections(spans, header_size, body_size)

    # Bullet character
    bullet_char = "•"
    for s in spans:
        ch = s["text"].lstrip()[:1]
        if ch in BULLET_CHARS:
            bullet_char = ch
            break

    # Margins
    lefts = [s["bbox"][0] for s in spans]
    tops = [s["bbox"][1] for s in spans]
    margin_left = round(min(lefts), 1) if lefts else 50.0
    margin_top = round(min(tops), 1) if tops else 50.0

    # Has name/contact header block
    has_header_block = bool(name_spans)

    # Name alignment: use the combined bounding box of all name spans
    # (the name may be split across multiple spans, so one span's center is unreliable)
    name_align = "left"
    if name_spans:
        combined_left = min(s["bbox"][0] for s in name_spans)
        combined_right = max(s["bbox"][2] for s in name_spans)
        text_center_x = (combined_left + combined_right) / 2
        if abs(text_center_x - page_width / 2) < page_width * 0.15:
            name_align = "center"

    # Colors (most common color per role)
    from collections import Counter

    def _dominant_color(span_list: list[dict]) -> str:
        if not span_list:
            return "#000000"
        counts = Counter(s["color"] for s in span_list)
        return _color_to_hex(counts.most_common(1)[0][0])

    name_color = _dominant_color(name_spans)
    header_color = _dominant_color(header_spans)
    body_color = _dominant_color(body_spans)

    # Detect horizontal separator lines (drawn paths spanning >40% of page width)
    pdf_has_section_rule = False
    pdf_rule_color = header_color
    pdf_rule_thickness = 0.5
    try:
        for drawing in page.get_drawings():
            for item in drawing.get("items", []):
                if item[0] == "l":
                    x0, y0, x1, y1 = item[1].x, item[1].y, item[2].x, item[2].y
                    if abs(y1 - y0) < 2 and abs(x1 - x0) > page_width * 0.4:
                        pdf_has_section_rule = True
                        pdf_rule_thickness = float(drawing.get("width") or 0.5)
                        stroke = drawing.get("color")
                        if stroke and len(stroke) >= 3:
                            r, g, b = int(stroke[0]*255), int(stroke[1]*255), int(stroke[2]*255)
                            pdf_rule_color = f"#{r:02x}{g:02x}{b:02x}"
                        break
            if pdf_has_section_rule:
                break
    except Exception:
        pass

    return StyleProfile(
        name_font_size=round(name_size, 1),
        name_bold=name_bold,
        header_font_size=round(header_size, 1),
        header_bold=header_bold,
        header_all_caps=header_all_caps,
        body_font_size=round(body_size, 1),
        contact_font_size=round(contact_size, 1),
        base_font_family=base_font,
        margin_left=margin_left,
        margin_top=margin_top,
        sections=sections,
        bullet_char=bullet_char,
        line_spacing=1.2,
        has_header_block=has_header_block,
        name_align=name_align,
        name_color=name_color,
        header_color=header_color,
        body_color=body_color,
        raw_font_name=raw_font,
        bold_phrases=_extract_bold_phrases_per_line(pdf_lines_data, body_size),
        has_section_rule=pdf_has_section_rule,
        section_rule_color=pdf_rule_color,
        section_rule_thickness=pdf_rule_thickness,
        section_rule_style="single",
    )


def parse_resume_structure_docx(docx_bytes: bytes) -> StyleProfile:
    """
    Analyse a resume .docx file and return a StyleProfile capturing its visual style.
    Falls back to StyleProfile.default() if the file cannot be read.
    """
    try:
        from collections import Counter
        from io import BytesIO

        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        doc = Document(BytesIO(docx_bytes))
    except Exception:
        return StyleProfile.default()

    # --- Helpers that walk the style inheritance chain ---

    def _eff_size(run, para) -> float:
        if run.font.size:
            return run.font.size.pt
        style = para.style
        while style:
            if style.font.size:
                return style.font.size.pt
            style = style.base_style
        return 11.0

    def _eff_bold(run, para) -> bool:
        if run.bold is not None:
            return bool(run.bold)
        style = para.style
        while style:
            if style.font.bold is not None:
                return bool(style.font.bold)
            style = style.base_style
        return False

    def _eff_font_name(run, para) -> str:
        if run.font.name:
            return run.font.name
        style = para.style
        while style:
            if style.font.name:
                return style.font.name
            style = style.base_style
        return "Calibri"

    def _eff_color(run) -> str:
        # Try run-level XML first (handles theme colors that python-docx can't resolve)
        try:
            from docx.oxml.ns import qn as _qn
            rPr = run._r.find(_qn("w:rPr"))
            if rPr is not None:
                color_el = rPr.find(_qn("w:color"))
                if color_el is not None:
                    val = color_el.get(_qn("w:val"))
                    if val and val.upper() != "AUTO" and len(val) == 6:
                        return f"#{val.lower()}"
        except Exception:
            pass
        # Fallback: python-docx resolved color
        try:
            if run.font.color and run.font.color.type is not None:
                rgb = run.font.color.rgb
                return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        except Exception:
            pass
        return "#000000"

    # --- Collect all paragraphs (body + table cells) ---

    all_paras = list(doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                all_paras.extend(cell.paragraphs)

    # --- Build run-level data ---

    runs_data: list[dict] = []
    para_meta: list[dict] = []  # one entry per paragraph for alignment lookup

    def _bottom_border_props(para) -> dict | None:
        """Return {"color": "#rrggbb", "thickness": pt} if para has a bottom border, else None."""
        try:
            from docx.oxml.ns import qn as _qn2
            pPr = para._p.find(_qn2("w:pPr"))
            if pPr is not None:
                pBdr = pPr.find(_qn2("w:pBdr"))
                if pBdr is not None:
                    bottom = pBdr.find(_qn2("w:bottom"))
                    if bottom is not None and bottom.get(_qn2("w:val"), "none") not in ("none", ""):
                        # w:sz is in eighths of a point
                        sz = int(bottom.get(_qn2("w:sz"), "4"))
                        thickness = sz / 8.0
                        raw_color = bottom.get(_qn2("w:color"), "auto")
                        color = f"#{raw_color.lower()}" if raw_color not in ("auto", "") and len(raw_color) == 6 else None
                        style = bottom.get(_qn2("w:val"), "single")
                        return {"color": color, "thickness": thickness, "style": style}
        except Exception:
            pass
        return None

    for para in all_paras:
        para_runs = []
        for run in para.runs:
            if not run.text.strip():
                continue
            entry = {
                "text": run.text.strip(),
                "size": float(_eff_size(run, para)),
                "bold": _eff_bold(run, para),
                "font": _eff_font_name(run, para),
                "color": _eff_color(run),
            }
            runs_data.append(entry)
            para_runs.append(entry)
        if para_runs:
            para_meta.append({"runs": para_runs, "align": para.alignment, "border": _bottom_border_props(para)})

    if not runs_data:
        return StyleProfile.default()

    # --- Font size roles ---

    from collections import Counter

    sizes = sorted({r["size"] for r in runs_data}, reverse=True)
    name_size = sizes[0]
    size_counts = Counter(round(r["size"] * 2) / 2 for r in runs_data)
    body_size = float(size_counts.most_common(1)[0][0])
    candidates = [s for s in sizes if body_size < s < name_size - 0.5]
    header_size = candidates[-1] if candidates else body_size

    name_runs = [r for r in runs_data if abs(r["size"] - name_size) < 0.5]
    header_runs = [r for r in runs_data if abs(r["size"] - header_size) < 0.5]

    section_rule_thickness = 0.5
    section_rule_color_raw = None
    # Detect border style from any paragraph in the document (name paragraph typically
    # carries the resume's decorative border which we apply under section headers too)
    has_section_rule = False
    section_rule_thickness = 0.5
    section_rule_color_raw = None
    section_rule_style_val = "single"
    from docx.oxml.ns import qn as _qn3
    for para in all_paras:
        props = _bottom_border_props(para)
        if props:
            has_section_rule = True
            section_rule_thickness = props["thickness"]
            section_rule_color_raw = props["color"]
            section_rule_style_val = props.get("style", "single")
            break
    body_runs = [r for r in runs_data if abs(r["size"] - body_size) < 1.0]

    name_bold = any(r["bold"] for r in name_runs) if name_runs else True
    header_bold = any(r["bold"] for r in header_runs) if header_runs else True
    header_all_caps = bool(header_runs) and all(
        r["text"].isupper() for r in header_runs[:6] if len(r["text"]) > 2
    )

    # raw_font from body runs (like PDF version), not the name
    raw_font = _base_font_name(body_runs[0]["font"]) if body_runs else "Calibri"
    base_font = _normalize_font_family(raw_font)

    # --- Name alignment: check the paragraph that contains name-sized text ---

    name_align = "left"
    for pm in para_meta:
        if any(abs(r["size"] - name_size) < 0.5 for r in pm["runs"]):
            if pm["align"] == WD_ALIGN_PARAGRAPH.CENTER:
                name_align = "center"
            break

    # --- Colors ---

    def dominant_color(run_list: list[dict]) -> str:
        if not run_list:
            return "#000000"
        counts = Counter(r["color"] for r in run_list)
        return counts.most_common(1)[0][0]

    section_rule_color = section_rule_color_raw or dominant_color(header_runs)
    section_rule_style = section_rule_style_val

    # --- Sections ---

    sections = list(dict.fromkeys(
        r["text"].rstrip(":")
        for r in runs_data
        if abs(r["size"] - header_size) < 0.5
        and r["text"].strip().rstrip(":").lower() in KNOWN_SECTIONS
    ))

    # --- Bullet character ---

    bullet_char = "•"
    for pm in para_meta:
        for r in pm["runs"]:
            ch = r["text"][:1]
            if ch in BULLET_CHARS:
                bullet_char = ch
                break

    # --- Bold body phrases ---

    bold_phrases_list = _extract_bold_phrases_per_line(
        [[{"text": r["text"], "size": r["size"], "bold": r["bold"]} for r in pm["runs"]]
         for pm in para_meta],
        body_size,
    )

    # --- Margins ---

    margin_left = 72.0
    margin_top = 72.0
    try:
        sec = doc.sections[0]
        margin_left = sec.left_margin.pt if sec.left_margin else 72.0
        margin_top = sec.top_margin.pt if sec.top_margin else 72.0
    except Exception:
        pass

    return StyleProfile(
        name_font_size=round(name_size, 1),
        name_bold=name_bold,
        header_font_size=round(header_size, 1),
        header_bold=header_bold,
        header_all_caps=header_all_caps,
        body_font_size=round(body_size, 1),
        contact_font_size=round(min(sizes), 1) if len(sizes) > 2 else max(round(body_size - 1.0, 1), 8.0),
        base_font_family=base_font,
        margin_left=margin_left,
        margin_top=margin_top,
        sections=sections,
        bullet_char=bullet_char,
        line_spacing=1.2,
        has_header_block=bool(name_runs),
        name_align=name_align,
        name_color=dominant_color(name_runs),
        header_color=dominant_color(header_runs),
        body_color=dominant_color(body_runs),
        raw_font_name=raw_font,
        bold_phrases=bold_phrases_list,
        has_section_rule=has_section_rule,
        section_rule_color=section_rule_color,
        section_rule_thickness=section_rule_thickness,
        section_rule_style=section_rule_style,
    )
