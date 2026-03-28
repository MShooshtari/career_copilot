"""
MCP server for resume formatting.

Exposes four tools:
  - parse_resume_structure       : Extract visual style from an original resume PDF
  - parse_resume_structure_docx  : Extract visual style from an original resume Word (.docx)
  - generate_pdf                 : Render improved resume text as a styled PDF
  - generate_docx                : Render improved resume text as a styled Word doc

Run:
    python mcp_server/server.py

Registers at http://127.0.0.1:8001/sse (SSE transport).
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

# Make the src package importable when running from the project root
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from career_copilot.resume_formatter.docx_builder import generate_formatted_docx
from career_copilot.resume_formatter.pdf_builder import generate_formatted_pdf
from career_copilot.resume_formatter.structure_parser import (
    StyleProfile,
    parse_resume_structure,
    parse_resume_structure_docx,
)

mcp = FastMCP(
    "resume-formatter",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@mcp.tool()
def parse_resume_structure_tool(pdf_bytes_b64: str) -> str:
    """
    Analyse a resume PDF and extract its visual style profile.

    Args:
        pdf_bytes_b64: Base64-encoded bytes of the original resume PDF.

    Returns:
        JSON string representing the StyleProfile (font sizes, margins,
        sections, bullet character, etc.).
    """
    pdf_bytes = base64.b64decode(pdf_bytes_b64)
    profile = parse_resume_structure(pdf_bytes)
    return profile.to_json()


@mcp.tool()
def parse_resume_structure_docx_tool(docx_bytes_b64: str) -> str:
    """
    Analyse a resume Word (.docx) file and extract its visual style profile.

    Args:
        docx_bytes_b64: Base64-encoded bytes of the original resume .docx file.

    Returns:
        JSON string representing the StyleProfile (font sizes, margins,
        sections, bullet character, etc.).
    """
    docx_bytes = base64.b64decode(docx_bytes_b64)
    profile = parse_resume_structure_docx(docx_bytes)
    return profile.to_json()


@mcp.tool()
def generate_pdf_tool(improved_text: str, style_profile_json: str) -> str:
    """
    Render improved resume text as a PDF that clones the original's visual style.

    Args:
        improved_text:      Plain-text improved resume (output from the LLM agent).
        style_profile_json: JSON string returned by parse_resume_structure_tool.

    Returns:
        Base64-encoded PDF bytes.
    """
    profile = StyleProfile.from_json(style_profile_json)
    pdf_bytes = generate_formatted_pdf(improved_text, profile)
    return base64.b64encode(pdf_bytes).decode()


@mcp.tool()
def generate_docx_tool(improved_text: str, style_profile_json: str) -> str:
    """
    Render improved resume text as a Word (.docx) document cloning the original style.

    Args:
        improved_text:      Plain-text improved resume (output from the LLM agent).
        style_profile_json: JSON string returned by parse_resume_structure_tool.

    Returns:
        Base64-encoded .docx bytes.
    """
    profile = StyleProfile.from_json(style_profile_json)
    docx_bytes = generate_formatted_docx(improved_text, profile)
    return base64.b64encode(docx_bytes).decode()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(mcp.streamable_http_app(), host="0.0.0.0", port=8001)
