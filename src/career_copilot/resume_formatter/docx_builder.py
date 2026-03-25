"""
Generate a formatted Word (.docx) document from improved resume text
using the original's StyleProfile.
"""

from __future__ import annotations

from io import BytesIO

from career_copilot.resume_formatter.structure_parser import (
    StyleProfile,
    parse_resume_text,
    split_inline_bold,
)


def _docx_font(family: str) -> str:
    """Map a reportlab-style font family to a Word font name."""
    mapping = {
        "Helvetica": "Calibri",
        "Times-Roman": "Times New Roman",
        "Courier": "Courier New",
    }
    return mapping.get(family, "Calibri")


def _pt_to_cm(pt_val: float) -> float:
    return pt_val / 28.35


def generate_formatted_docx(improved_text: str, profile: StyleProfile) -> bytes:
    """
    Render the improved resume text into a Word document that clones the visual
    style described by *profile*.  Returns raw .docx bytes.
    """
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt, RGBColor

    def _hex_to_rgb(hex_color: str) -> RGBColor:
        h = hex_color.lstrip("#")
        return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    doc = Document()

    # Page margins
    margin_cm = Cm(max(_pt_to_cm(profile.margin_left), 1.27))  # min 0.5"
    top_cm = Cm(max(_pt_to_cm(profile.margin_top), 1.27))
    for section in doc.sections:
        section.left_margin = margin_cm
        section.right_margin = margin_cm
        section.top_margin = top_cm
        section.bottom_margin = Cm(1.77)

    font_name = profile.raw_font_name or _docx_font(profile.base_font_family)
    center = WD_ALIGN_PARAGRAPH.CENTER
    left = WD_ALIGN_PARAGRAPH.LEFT
    name_alignment = center if profile.name_align == "center" else left

    name_color_rgb = _hex_to_rgb(profile.name_color)
    header_color_rgb = _hex_to_rgb(profile.header_color)
    body_color_rgb = _hex_to_rgb(profile.body_color)

    # Configure Word styles so bold/color/size are set at the style level,
    # which Word renders reliably regardless of run-level inheritance quirks.
    for style_name, font_size, bold, color_rgb, alignment, sp_before, sp_after in [
        ("Title", profile.name_font_size, profile.name_bold, name_color_rgb, name_alignment, 0, 2),
        ("Heading 1", profile.header_font_size, profile.header_bold, header_color_rgb, left, 8, 4),
    ]:
        try:
            s = doc.styles[style_name]
            s.font.name = font_name
            s.font.size = Pt(font_size)
            s.font.bold = bold
            s.font.color.rgb = color_rgb
            s.paragraph_format.alignment = alignment
            s.paragraph_format.space_before = Pt(sp_before)
            s.paragraph_format.space_after = Pt(sp_after)
        except Exception:
            pass

    def add_runs(p, text: str, font_size: float, color_rgb, base_bold: bool = False) -> None:
        """Add one or more runs to *p*, honouring **inline bold** markers."""
        for segment, is_bold in split_inline_bold(text):
            run = p.add_run(segment)
            run.font.name = font_name
            run.font.size = Pt(font_size)
            run.font.bold = base_bold or is_bold
            run.font.color.rgb = color_rgb

    elements = parse_resume_text(improved_text, profile)

    for el in elements:
        if el.kind == "blank":
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(2)

        elif el.kind == "name":
            doc.add_paragraph(el.text, style="Title")

        elif el.kind == "contact":
            p = doc.add_paragraph()
            p.alignment = name_alignment
            p.paragraph_format.space_after = Pt(1)
            add_runs(p, el.text, profile.contact_font_size, body_color_rgb)

        elif el.kind == "section_header":
            doc.add_paragraph(el.text, style="Heading 1")

        elif el.kind == "bullet":
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Pt(14)
            p.paragraph_format.space_after = Pt(1)
            # Bullet character as its own non-bold run, then the content
            run = p.add_run(f"{profile.bullet_char} ")
            run.font.name = font_name
            run.font.size = Pt(profile.body_font_size)
            run.font.color.rgb = body_color_rgb
            add_runs(p, el.text, profile.body_font_size, body_color_rgb)

        else:  # body
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(2)
            add_runs(p, el.text, profile.body_font_size, body_color_rgb)

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.read()
