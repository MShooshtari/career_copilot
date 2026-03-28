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

# Tokens that indicate a bold variant of a font file
_BOLD_TOKENS = ("bold", "bd", "b")
# Tokens that indicate a regular (non-bold, non-italic) variant
_REGULAR_TOKENS = ("regular", "roman", "book", "light", "medium", "")


def _scan_font_dir(font_dir: str, family_lower: str) -> tuple[str | None, str | None]:
    """Scan *font_dir* for TTF/OTF files matching *family_lower*.

    Returns (regular_path, bold_path); either may be None if not found.
    Prefers files whose name after stripping the family prefix is a known
    weight token (e.g. "calibriregular.ttf" → regular, "calibribd.ttf" → bold).
    """
    if not os.path.isdir(font_dir):
        return None, None

    reg_path: str | None = None
    bold_path: str | None = None
    family_nospace = family_lower.replace(" ", "")

    try:
        entries = os.listdir(font_dir)
    except OSError:
        return None, None

    for fname in entries:
        ext = os.path.splitext(fname)[1].lower()
        if ext not in (".ttf", ".otf"):
            continue
        base = os.path.splitext(fname.lower())[0].replace("-", "").replace("_", "").replace(" ", "")

        if not base.startswith(family_nospace):
            continue

        suffix = base[len(family_nospace) :]
        full_path = os.path.join(font_dir, fname)

        if suffix in _BOLD_TOKENS:
            bold_path = full_path
        elif suffix in _REGULAR_TOKENS and reg_path is None:
            reg_path = full_path

    return reg_path, bold_path


def _embed_system_font(family: str) -> tuple[str, str]:
    """Find and register a system font by family name. Returns (regular_name, bold_name).
    Tries the hardcoded filename table first, then scans font directories dynamically.
    Falls back to built-in Helvetica if the font cannot be found or embedded."""
    if family in _registered:
        return _registered[family]

    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    def _try_register(reg_path: str, bold_path: str) -> tuple[str, str] | None:
        try:
            reg_name = f"Sys-{family}"
            bold_name = f"Sys-{family}-Bold"
            pdfmetrics.registerFont(TTFont(reg_name, reg_path))
            pdfmetrics.registerFont(TTFont(bold_name, bold_path))
            _registered[family] = (reg_name, bold_name)
            return (reg_name, bold_name)
        except Exception:
            return None

    # 1. Hardcoded table — fast path for common fonts
    files = _WINDOWS_FONT_FILES.get(family)
    if files:
        reg_file, bold_file = files
        for font_dir in _FONT_DIRS:
            reg_path = os.path.join(font_dir, reg_file)
            bold_path = os.path.join(font_dir, bold_file)
            if os.path.exists(reg_path) and os.path.exists(bold_path):
                result = _try_register(reg_path, bold_path)
                if result:
                    return result

    # 2. Dynamic scan — handles fonts not in the hardcoded table
    family_lower = family.lower()
    for font_dir in _FONT_DIRS:
        reg_path, bold_path = _scan_font_dir(font_dir, family_lower)
        if reg_path:
            # Use regular as bold fallback if no dedicated bold file found
            result = _try_register(reg_path, bold_path or reg_path)
            if result:
                return result

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
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        HRFlowable,
        PageTemplate,
        Paragraph,
        Spacer,
    )

    buffer = BytesIO()
    margin = max(float(profile.margin_left), 36.0)  # min 0.5"
    top_margin = max(float(profile.margin_top), 36.0)

    page_w, page_h = A4  # (595.28, 841.89) pts
    bottom_margin = 50.0
    frame_w = page_w - 2 * margin
    frame_h = page_h - top_margin - bottom_margin

    # Zero-padding frame so _text_width exactly equals the frame's available width,
    # preventing ReportLab from centering Tables that would otherwise overflow the interior.
    _content_frame = Frame(
        margin,
        bottom_margin,
        frame_w,
        frame_h,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="normal",
    )
    doc = BaseDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
    )
    doc.addPageTemplates([PageTemplate(id="Normal", frames=[_content_frame], pagesize=A4)])

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
        leftIndent=0,
        firstLineIndent=0,
        spaceAfter=profile.name_space_after,
    )
    contact_style = ParagraphStyle(
        "ResumeContact",
        fontName=body_font,
        fontSize=profile.contact_font_size,
        leading=profile.contact_font_size * 1.4,
        alignment=name_align,
        textColor=HexColor(profile.body_color),
        leftIndent=0,
        firstLineIndent=0,
        spaceAfter=profile.contact_space_after,
    )
    header_style = ParagraphStyle(
        "ResumeSectionHeader",
        fontName=header_font,
        fontSize=profile.header_font_size,
        leading=profile.header_font_size * 1.4,
        spaceBefore=profile.header_space_before,
        spaceAfter=profile.header_space_after,
        textColor=HexColor(profile.header_color),
        leftIndent=0,
        firstLineIndent=0,
        borderWidth=0,
    )
    tagline_style = ParagraphStyle(
        "ResumeTagline",
        fontName=header_font,
        fontSize=profile.header_font_size,
        leading=profile.header_font_size * 1.4,
        alignment=name_align,
        textColor=HexColor(profile.header_color),
        leftIndent=0,
        firstLineIndent=0,
        spaceAfter=profile.tagline_space_after,
    )
    body_style = ParagraphStyle(
        "ResumeBody",
        fontName=body_font,
        fontSize=profile.body_font_size,
        leading=profile.body_font_size * profile.line_spacing,
        textColor=HexColor(profile.body_color),
        leftIndent=0,
        firstLineIndent=0,
        spaceAfter=profile.body_space_after,
    )
    bullet_style = ParagraphStyle(
        "ResumeBullet",
        fontName=body_font,
        fontSize=profile.body_font_size,
        leading=profile.body_font_size * profile.line_spacing,
        textColor=HexColor(profile.body_color),
        leftIndent=14,
        firstLineIndent=0,
        spaceAfter=profile.bullet_space_after,
    )

    # --- Build story ---
    elements = parse_resume_text(improved_text, profile)
    story = []

    _text_width = frame_w  # exact interior width (frame has 0 padding)
    _RIGHT_COL = 160.0  # pts — enough for dates and locations

    def _split_row(left_markup: str, right_text_raw: str, left_style, right_size: float) -> object:
        from reportlab.lib.enums import TA_LEFT, TA_RIGHT
        from reportlab.platypus import Table
        from reportlab.platypus import TableStyle as TS

        # Left cell is always left-aligned (tab-stop-style layout); right cell right-aligned
        left_split_style = ParagraphStyle(
            left_style.name + "_split",
            parent=left_style,
            alignment=TA_LEFT,
        )
        right_style = ParagraphStyle(
            "SplitRight",
            fontName=body_font,
            fontSize=right_size,
            leading=right_size * profile.line_spacing,
            textColor=HexColor(profile.body_color),
            alignment=TA_RIGHT,
        )
        left_col = _text_width - _RIGHT_COL
        t = Table(
            [
                [
                    Paragraph(left_markup, left_split_style),
                    Paragraph(_safe(right_text_raw), right_style),
                ]
            ],
            colWidths=[left_col, _RIGHT_COL],
            spaceBefore=0,
            spaceAfter=left_style.spaceAfter,
        )
        t.setStyle(
            TS(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        return t

    for el in elements:
        if el.kind == "blank":
            story.append(Spacer(1, 4))
        elif el.kind == "name":
            story.append(Paragraph(_safe(el.text), name_style))
        elif el.kind == "contact":
            markup = _markup(el.text, body_font_bold)
            if el.right_text:
                story.append(
                    _split_row(markup, el.right_text, contact_style, profile.contact_font_size)
                )
            else:
                story.append(Paragraph(markup, contact_style))
        elif el.kind == "tagline":
            story.append(Paragraph(_safe(el.text), tagline_style))
        elif el.kind == "section_header":
            story.append(Paragraph(_safe(el.text), header_style))
            if el.has_rule:
                rule_color = HexColor(profile.section_rule_color)
                thick = profile.section_rule_thickness
                if profile.section_rule_style in (
                    "thickThinSmallGap",
                    "thinThickSmallGap",
                    "thickThinMediumGap",
                    "thinThickMediumGap",
                    "thickThinLargeGap",
                    "thinThickLargeGap",
                ):
                    story.append(
                        HRFlowable(
                            width=_text_width,
                            thickness=thick * 0.65,
                            color=rule_color,
                            spaceAfter=1,
                            spaceBefore=1,
                            hAlign="LEFT",
                        )
                    )
                    story.append(
                        HRFlowable(
                            width=_text_width,
                            thickness=thick * 0.25,
                            color=rule_color,
                            spaceAfter=2,
                            spaceBefore=1,
                            hAlign="LEFT",
                        )
                    )
                else:
                    story.append(
                        HRFlowable(
                            width=_text_width,
                            thickness=thick,
                            color=rule_color,
                            spaceAfter=2,
                            spaceBefore=1,
                            hAlign="LEFT",
                        )
                    )
        elif el.kind == "header_rule":
            rule_color = HexColor(profile.section_rule_color)
            thick = profile.section_rule_thickness
            if profile.section_rule_style in (
                "thickThinSmallGap",
                "thinThickSmallGap",
                "thickThinMediumGap",
                "thinThickMediumGap",
                "thickThinLargeGap",
                "thinThickLargeGap",
            ):
                story.append(
                    HRFlowable(
                        width=_text_width,
                        thickness=thick * 0.65,
                        color=rule_color,
                        spaceAfter=1,
                        spaceBefore=2,
                        hAlign="LEFT",
                    )
                )
                story.append(
                    HRFlowable(
                        width=_text_width,
                        thickness=thick * 0.25,
                        color=rule_color,
                        spaceAfter=4,
                        spaceBefore=1,
                        hAlign="LEFT",
                    )
                )
            else:
                story.append(
                    HRFlowable(
                        width=_text_width,
                        thickness=thick,
                        color=rule_color,
                        spaceAfter=4,
                        spaceBefore=2,
                        hAlign="LEFT",
                    )
                )
        elif el.kind == "bullet":
            story.append(
                Paragraph(
                    f"{_safe(profile.bullet_char)}\u00a0{_markup(el.text, body_font_bold)}",
                    bullet_style,
                )
            )
        else:  # body
            markup = _markup(el.text, body_font_bold)
            if el.right_text:
                story.append(_split_row(markup, el.right_text, body_style, profile.body_font_size))
            else:
                story.append(Paragraph(markup, body_style))

    if not story:
        story.append(Paragraph("Resume could not be generated.", body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
