"""
Minimal MCP (Model Context Protocol) stdio client.

Used to launch a local MCP server subprocess and call a tool via JSON-RPC over
stdio using Content-Length framing (per MCP stdio transport spec).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass
from typing import Any


class McpProtocolError(RuntimeError):
    pass


def _encode_message(payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _read_framed_message(stream) -> dict[str, Any]:
    # Read headers until blank line
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            raise McpProtocolError("Unexpected EOF while reading MCP headers")
        if line in (b"\r\n", b"\n"):
            break
        try:
            k, v = line.decode("ascii", errors="replace").split(":", 1)
        except ValueError as e:
            raise McpProtocolError(f"Invalid header line: {line!r}") from e
        headers[k.strip().lower()] = v.strip()

    if "content-length" not in headers:
        raise McpProtocolError("Missing Content-Length header")
    try:
        n = int(headers["content-length"])
    except ValueError as e:
        raise McpProtocolError(f"Invalid Content-Length: {headers['content-length']!r}") from e
    body = stream.read(n)
    if body is None or len(body) != n:
        raise McpProtocolError("Unexpected EOF while reading MCP body")
    try:
        return json.loads(body.decode("utf-8"))
    except Exception as e:
        raise McpProtocolError("Invalid JSON body from MCP server") from e


@dataclass
class McpStdioServerSpec:
    command: str
    args: list[str]
    env: dict[str, str] | None = None


class McpStdioClient:
    def __init__(self, spec: McpStdioServerSpec) -> None:
        self._spec = spec
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._next_id = 1

    def start(self) -> None:
        if self._proc and self._proc.poll() is None:
            return
        env = os.environ.copy()
        if self._spec.env:
            env.update(self._spec.env)
        self._proc = subprocess.Popen(
            [self._spec.command, *self._spec.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # MCP initialize handshake (minimal)
        self._request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "career_copilot", "version": "0.1"},
            },
        )
        self._notify("notifications/initialized", {})

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if not proc:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
        finally:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            try:
                if proc.stderr:
                    proc.stderr.close()
            except Exception:
                pass

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.start()
        return self._request("tools/call", {"name": name, "arguments": arguments})

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        proc = self._proc
        if not proc or not proc.stdin:
            raise McpProtocolError("MCP server not running")
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        proc.stdin.write(_encode_message(payload))
        proc.stdin.flush()

    def _request(self, method: str, params: dict[str, Any]) -> Any:
        with self._lock:
            proc = self._proc
            if not proc or not proc.stdin or not proc.stdout:
                raise McpProtocolError("MCP server not running")
            req_id = self._next_id
            self._next_id += 1
            payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
            proc.stdin.write(_encode_message(payload))
            proc.stdin.flush()

            while True:
                msg = _read_framed_message(proc.stdout)
                # Responses have id; notifications don't.
                if msg.get("id") != req_id:
                    continue
                if "error" in msg and msg["error"]:
                    raise McpProtocolError(str(msg["error"]))
                return msg.get("result")

