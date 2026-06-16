"""A small synchronous wrapper around an MCP stdio server.

The MCP Python SDK is async (anyio), but our agent loop in `run_agent.py` is
synchronous. Rather than scatter `asyncio.run(...)` calls (which would spawn and
kill the server subprocess on every tool call), this wrapper runs a single
persistent asyncio event loop on a background thread and keeps one MCP session
open for the wrapper's lifetime.

Key correctness detail: the stdio + session async context managers are entered
*and* exited inside the same task (`_serve`), and torn down only when `close()`
sets the shutdown event. Entering in one task and exiting in another trips
anyio's "cancel scope in a different task" guard, so we deliberately avoid that.
Individual `list_tools` / `call_tool` requests run as separate tasks on the same
loop, which is fine — they only do request/response over the session streams.

Usage:
    client = MCPStdioClient("npx", ["-y", "firecrawl-mcp"], env={...})
    tools = client.list_tools()             # list of mcp.types.Tool
    text = client.call_tool("firecrawl_search", {"query": "..."})
    client.close()

Or as a context manager:
    with MCPStdioClient(...) as client:
        ...
"""

from __future__ import annotations

import asyncio
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPStdioClient:
    """Persistent, thread-safe sync handle to one MCP server over stdio."""

    def __init__(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        ready_timeout: float = 60.0,
    ):
        self._params = StdioServerParameters(command=command, args=list(args), env=env)
        self._session: ClientSession | None = None
        self._shutdown: asyncio.Event | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="mcp-stdio-loop", daemon=True
        )
        self._thread.start()

        self._serve_future = asyncio.run_coroutine_threadsafe(self._serve(), self._loop)
        if not self._ready.wait(timeout=ready_timeout):
            self.close()
            raise TimeoutError(
                f"MCP server {command!r} did not become ready in {ready_timeout}s"
            )
        if self._error is not None:
            self.close()
            raise RuntimeError(f"Failed to start MCP server {command!r}") from self._error

    async def _serve(self) -> None:
        """Hold the stdio + session contexts open until close() signals shutdown."""
        self._shutdown = asyncio.Event()
        try:
            async with stdio_client(self._params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._shutdown.wait()
        except BaseException as exc:  # surface startup failures to __init__
            self._error = exc
            self._ready.set()

    def _run(self, coro):
        """Schedule a coroutine on the background loop and block for its result."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def list_tools(self):
        """Return the server's tools (list of `mcp.types.Tool`)."""
        assert self._session is not None
        return self._run(self._session.list_tools()).tools

    def call_tool(self, name: str, arguments: dict | None = None) -> str:
        """Call a tool and return its text content (joined across text blocks).

        Tool errors are not raised — they're returned as an "Error: ..." string so
        the agent loop can feed them back to the model and let it recover.
        """
        assert self._session is not None
        result = self._run(self._session.call_tool(name, arguments or {}))
        parts = [
            block.text
            for block in result.content
            if getattr(block, "type", None) == "text"
        ]
        text = "\n".join(parts) if parts else "(tool returned no text content)"
        if getattr(result, "isError", False):
            return f"Error from tool {name!r}: {text}"
        return text

    def close(self) -> None:
        """Shut the session down (in its own task) and stop the loop thread."""
        if self._shutdown is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._shutdown.set)
            try:
                self._serve_future.result(timeout=10)
            except Exception:
                pass
        if not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    def __enter__(self) -> "MCPStdioClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
