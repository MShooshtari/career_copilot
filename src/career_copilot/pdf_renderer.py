from __future__ import annotations

import base64
import json
import os
from typing import Any

from career_copilot.mcp_client import McpStdioClient, McpStdioServerSpec
from career_copilot.resume_pdf import build_resume_pdf


def _get_pdf_renderer_mode() -> str:
    return (os.environ.get("RESUME_PDF_RENDERER") or "builtin").strip().lower()


def _get_mcp_spec() -> McpStdioServerSpec:
    """
    Configure how to launch the MCP server subprocess.

    Env vars:
      - CAREER_COPILOT_RESUME_MCP_COMMAND (default: python)
      - CAREER_COPILOT_RESUME_MCP_ARGS (default: JSON array for -m module)
    """
    command = (os.environ.get("CAREER_COPILOT_RESUME_MCP_COMMAND") or "python").strip()
    args_raw = os.environ.get("CAREER_COPILOT_RESUME_MCP_ARGS")
    if args_raw:
        args = json.loads(args_raw)
        if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
            raise ValueError("CAREER_COPILOT_RESUME_MCP_ARGS must be a JSON array of strings")
    else:
        args = ["-m", "career_copilot.mcp_servers.resume_pdf_server"]
    return McpStdioServerSpec(command=command, args=args)


def render_resume_pdf(text: str) -> bytes:
    """
    Render resume text to PDF bytes.

    Modes:
      - builtin: current simple PDF using PyMuPDF (no external tools needed)
      - mcp: call local MCP server that uses LaTeX/pdflatex
    """
    mode = _get_pdf_renderer_mode()
    if mode in ("builtin", "simple"):
        return build_resume_pdf(text or "")

    if mode in ("mcp", "latex", "mcp_latex"):
        client = McpStdioClient(_get_mcp_spec())
        try:
            result: Any = client.call_tool("render_resume_pdf", {"resume_text": text or ""})
        finally:
            client.close()

        if not isinstance(result, dict) or "pdf_base64" not in result:
            raise RuntimeError("MCP render_resume_pdf returned an unexpected result")
        return base64.b64decode(result["pdf_base64"])

    raise ValueError(f"Unknown RESUME_PDF_RENDERER mode: {mode!r}")

