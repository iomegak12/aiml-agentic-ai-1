"""Stream an agent run and print reasoning, answer tokens, tool calls and tool results."""

import json
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command, Interrupt

DIM, CYAN, YELLOW, RESET = "\033[2m", "\033[36m", "\033[33m", "\033[0m"
MAX_RESULT_CHARS = 400


def _source(namespace: tuple[str, ...]) -> str:
    """'' for the main agent, '[subagent] ' for events from a `task` subagent."""
    return "[subagent] " if any(s.startswith("tools:") for s in namespace) else ""


def _content_parts(message: AIMessage) -> list[tuple[str, str]]:
    """Split a streamed chunk into ("reasoning" | "text", text) parts."""
    if isinstance(message.content, str):
        return [("text", message.content)] if message.content else []

    parts = []
    for block in message.content:
        if isinstance(block, str):
            parts.append(("text", block))
        elif block.get("type") == "text" and block.get("text"):
            parts.append(("text", block["text"]))
        elif block.get("type") == "reasoning":
            for summary in block.get("summary", []):
                if summary.get("text"):
                    parts.append(("reasoning", summary["text"]))
            if block.get("reasoning"):  # standard (v1) reasoning block
                parts.append(("reasoning", block["reasoning"]))
    return parts


def _short(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + " ..."


class _Printer:
    """Prints streamed output, starting a new line whenever the kind of output changes."""

    def __init__(self) -> None:
        self._mode: str | None = None

    def stream_text(self, source: str, kind: str, text: str) -> None:
        if self._mode != kind:
            self._newline()
            label = f"{DIM}🧠 {source}thinking: " if kind == "reasoning" else f"🤖 {source}"
            print(label, end="")
            self._mode = kind
        print(text, end="", flush=True)
        if kind == "reasoning":
            print(RESET, end="")

    def line(self, text: str) -> None:
        self._newline()
        print(text, flush=True)

    def _newline(self) -> None:
        if self._mode is not None:
            print(RESET)
        self._mode = None

    def finish(self) -> None:
        self._newline()


def stream_agent(
    agent: Any, payload: str | dict | Command, config: dict[str, Any]
) -> tuple[Interrupt, ...]:
    """Run the agent with streaming output.

    Args:
        agent: A compiled deep agent (with a checkpointer).
        payload: A user message, an input dict, or a Command(resume=...).
        config: Thread config, e.g. ticket_agent.new_thread().

    Returns:
        Pending interrupts (e.g. a plan waiting for approval); empty when the run finished.
    """
    if isinstance(payload, str):
        payload = {"messages": [{"role": "user", "content": payload}]}

    printer = _Printer()
    for chunk in agent.stream(
        payload,
        config=config,
        stream_mode=["updates", "messages"],
        subgraphs=True,
        version="v2",
    ):
        source = _source(chunk["ns"])

        if chunk["type"] == "messages":
            message, _metadata = chunk["data"]
            if isinstance(message, AIMessage):  # chunks while streaming; a full message if the model didn't stream
                for kind, text in _content_parts(message):
                    printer.stream_text(source, kind, text)

        elif chunk["type"] == "updates":
            for node, update in chunk["data"].items():
                if not isinstance(update, dict) or not isinstance(update.get("messages"), list):
                    continue
                for message in update["messages"]:
                    if isinstance(message, AIMessage):
                        for call in message.tool_calls:
                            printer.line(f"{CYAN}🔧 {source}{call['name']}({_short(call['args'])}){RESET}")
                    elif isinstance(message, ToolMessage):
                        printer.line(f"{YELLOW}   ↳ {source}{message.name}: {_short(message.content)}{RESET}")

    printer.finish()
    return agent.get_state(config).interrupts
