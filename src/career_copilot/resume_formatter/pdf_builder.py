"""
Generate a formatted PDF from improved resume text using the original's StyleProfile.
Uses reportlab platypus for layout.
"""

from __future__ import annotations

import os
from io import BytesIO
from xml.sax.saxutils import escape

from career_copilot.resume_formatter.structure_parser import (
    StyleProfile,
    parse_resume_text,
    split_inline_bold,
)

# Maps a clean family name to (regular_filename, bold_filename) in the system fonts dir
_WINDOWS_FONT_FILES: dict[str, tuple[str, str]] = {
    "Calibri": ("calibri.ttf", "calibrib.ttf"),
    "Arial": ("arial.ttf", "arialbd.ttf"),
    "Georgia": ("georgia.ttf", "georgiab.ttf"),
    "Verdana": ("verdana.ttf", "verdanab.ttf"),
    "Tahoma": ("tahoma.ttf", "tahomabd.ttf"),
    "Trebuchet MS": ("trebuc.ttf", "trebucbd.ttf"),
    "Times New Roman": ("times.ttf", "timesbd.ttf"),
    "Courier New": ("cour.ttf", "courbd.ttf"),
    "Garamond": ("GARA.TTF", "GARABD.TTF"),
    "Cambria": ("cambria.ttc", "cambriab.ttf"),
    "Gill Sans MT": ("GILSANUB.TTF", "GILSANUB.TTF"),
}

_FONT_DIRS = [
    r"C:\Windows\Fonts",
    os.path.expanduser("~/Library/Fonts"),
    "/usr/share/fonts/truetype",
    "/usr/share/fonts",
]

_registered: dict[str, tuple[str, str]] = {}  # cache: family → (reg_name, bold_name)


def _embed_system_font(family: str) -> tuple[str, str]:
    """Try to find and register the system font. Returns (regular_name, bold_name).
    Falls back to built-in Helvetica names if not found."""
    if family in _registered:
        return _registered[family]

    files = _WINDOWS_FONT_FILES.get(family)
    if not files:
        return ("Helvetica", "Helvetica-Bold")

    reg_file, bold_file = files
    for font_dir in _FONT_DIRS:
        reg_path = os.path.join(font_dir, reg_file)
        bold_path = os.path.join(font_dir, bold_file)
        if os.path.exists(reg_path) and os.path.exists(bold_path):
            try:
                from reportlab.pdfbase import pdfmetrics
                from reportlab.pdfbase.ttfonts import TTFont

                reg_name = f"Sys-{family}"
                bold_name = f"Sys-{family}-Bold"
                pdfmetrics.registerFont(TTFont(reg_name, reg_path))
                pdfmetrics.registerFont(TTFont(bold_name, bold_path))
                _registered[family] = (reg_name, bold_name)
                return (reg_name, bold_name)
            except Exception:
                pass

    return ("Helvetica", "Helvetica-Bold")


def _rl_font(family: str, raw_font: str, bold: bool = False) -> str:
    """Return a reportlab font name, preferring an embedded system font over built-ins."""
    reg_name, bold_name = _embed_system_font(raw_font)
    if reg_name != "Helvetica":
        # Successfully embedded the system font
        return bold_name if bold else reg_name

    # Fall back to built-in reportlab fonts
    mapping: dict[str, tuple[str, str]] = {
        "Helvetica": ("Helvetica", "Helvetica-Bold"),
        "Times-Roman": ("Times-Roman", "Times-Bold"),
        "Courier": ("Courier", "Courier-Bold"),
    }
    regular, bold_name_builtin = mapping.get(family, ("Helvetica", "Helvetica-Bold"))
    return bold_name_builtin if bold else regular


def _safe(text: str) -> str:
    """Escape XML special chars for reportlab Paragraph."""
    return escape(text)


def _markup(text: str, bold_font: str) -> str:
    """Convert **markers** to reportlab <font> bold tags, with XML escaping."""
    parts = split_inline_bold(text)
    if len(parts) == 1 and not parts[0][1]:
        # No bold markers — fast path
        return escape(text)
    out = []
    for segment, is_bold in parts:
        escaped = escape(segment)
        out.append(f'<font name="{bold_font}">{escaped}</font>' if is_bold else escaped)
    return "".join(out)


def generate_formatted_pdf(improved_text: str, profile: StyleProfile) -> bytes:
    """
    Render the improved resume text into a PDF that clones the visual style
    described by *profile*.  Returns raw PDF bytes.
    """
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import pt
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    buffer = BytesIO()
    margin = max(float(profile.margin_left), 36.0)  # min 0.5"
    top_margin = max(float(profile.margin_top), 36.0)

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=top_margin,
        bottomMargin=50,
    )

    # --- Styles ---
    raw = profile.raw_font_name
    body_font = _rl_font(profile.base_font_family, raw, bold=False)
    body_font_bold = _rl_font(profile.base_font_family, raw, bold=True)
    name_font = _rl_font(profile.base_font_family, raw, bold=profile.name_bold)
    header_font = _rl_font(profile.base_font_family, raw, bold=profile.header_bold)

    name_align = TA_CENTER if profile.name_align == "center" else TA_LEFT

    name_style = ParagraphStyle(
        "ResumeName",
        fontName=name_font,
        fontSize=profile.name_font_size,
        leading=profile.name_font_size * 1.3,
        alignment=name_align,
        textColor=HexColor(profile.name_color),
        spaceAfter=3,
    )
    contact_style = ParagraphStyle(
        "ResumeContact",
        fontName=body_font,
        fontSize=profile.contact_font_size,
        leading=profile.contact_font_size * 1.4,
        alignment=name_align,
        textColor=HexColor(profile.body_color),
        spaceAfter=2,
    )
    header_style = ParagraphStyle(
        "ResumeSectionHeader",
        fontName=header_font,
        fontSize=profile.header_font_size,
        leading=profile.header_font_size * 1.4,
        spaceBefore=10,
        spaceAfter=4,
        textColor=HexColor(profile.header_color),
        borderWidth=0,
    )
    body_style = ParagraphStyle(
        "ResumeBody",
        fontName=body_font,
        fontSize=profile.body_font_size,
        leading=profile.body_font_size * profile.line_spacing,
        textColor=HexColor(profile.body_color),
        spaceAfter=2,
    )
    bullet_style = ParagraphStyle(
        "ResumeBullet",
        fontName=body_font,
        fontSize=profile.body_font_size,
        leading=profile.body_font_size * profile.line_spacing,
        textColor=HexColor(profile.body_color),
        leftIndent=14,
        firstLineIndent=0,
        spaceAfter=2,
    )

    # --- Build story ---
    elements = parse_resume_text(improved_text, profile)
    story = []

    for el in elements:
        if el.kind == "blank":
            story.append(Spacer(1, 4))
        elif el.kind == "name":
            story.append(Paragraph(_safe(el.text), name_style))
        elif el.kind == "contact":
            story.append(Paragraph(_markup(el.text, body_font_bold), contact_style))
        elif el.kind == "section_header":
            story.append(Paragraph(_safe(el.text), header_style))
        elif el.kind == "bullet":
            story.append(
                Paragraph(
                    f"{_safe(profile.bullet_char)}\u00a0{_markup(el.text, body_font_bold)}",
                    bullet_style,
                )
            )
        else:  # body
            story.append(Paragraph(_markup(el.text, body_font_bold), body_style))

    if not story:
        story.append(Paragraph("Resume could not be generated.", body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
