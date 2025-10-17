# src/agents/mallsupport_agent.py
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

from livekit import rtc
from livekit.agents import llm
from livekit.agents.voice import Agent, ModelSettings

# Plugins
from livekit.plugins import elevenlabs, openai, silero

# ---- Import only the knowledge base tool ----
from src.tools.mallsupport_tools import query_knowledge_base

logger = logging.getLogger("dubai-mall-support-agent")
logger.setLevel(logging.INFO)


def load_prompt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8").strip()


EVE_MALL_PROMPT = load_prompt("src/prompts/mall_support.txt")


class MallSupportAgent(Agent):
    """
    Dubai Mall Customer Support Agent:
    - Handles queries from customers
    - Single tool: query_knowledge_base
    - RAG-based answers for knowledge base
    - Keeps responses concise and polite
    """

    def __init__(self, cfg: dict, room: rtc.Room, chat_ctx=None) -> None:
        self.room = room
        self.cfg = cfg
        super().__init__(
            instructions=EVE_MALL_PROMPT,
            chat_ctx=chat_ctx,
            tools=[query_knowledge_base],
        )

        self.actions = {
            "Query Knowledge Base": query_knowledge_base,
        }
        self.function_to_action = {v: k for k, v in self.actions.items()}

        # Optional: keys to filter from tool results before sending
        self.tool_result_filters = {
            query_knowledge_base: ["internal_id", "metadata"],
        }

        self.tool_cards = {
            query_knowledge_base: "knowledge_base_result",
        }

        self.visible_tools = {query_knowledge_base}

    async def _send_websocket_message(
        self, action: str, result: Dict[str, Any] = None, tool_func=None
    ):
        """Send WebSocket message with action, filtered result, and card_name."""
        message = {"action": action}

        if result is not None:
            if tool_func in self.tool_result_filters:
                for key in self.tool_result_filters[tool_func]:
                    result.pop(key, None)

            message["result"] = result
            message["card_name"] = self.tool_cards.get(tool_func, "generic")

        try:
            await self.room.local_participant.send_text(
                json.dumps(message), topic="lk.transcription"
            )
            logger.info("Sent WebSocket message: %s", message)
        except Exception as e:
            logger.error("Failed to send WebSocket message: %s", e)

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool | llm.RawFunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[llm.ChatChunk | str, None]:
        """Custom LLM node that captures full response text and executes tools."""

        activity = self._get_activity_or_raise()
        assert activity.llm is not None, "llm_node called but no LLM node is available"
        assert isinstance(activity.llm, llm.LLM)

        tool_choice = model_settings.tool_choice if model_settings else llm.NOT_GIVEN
        activity_llm = activity.llm
        conn_options = activity.session.conn_options.llm_conn_options

        buffer: list[str] = []
        pending_tools: list[tuple[str, callable, dict]] = []

        async with activity_llm.chat(
            chat_ctx=chat_ctx,
            tools=tools,
            tool_choice=tool_choice,
            conn_options=conn_options,
        ) as stream:
            async for chunk in stream:
                if isinstance(chunk, str):
                    buffer.append(chunk)

                elif isinstance(chunk, llm.ChatChunk):
                    if chunk.delta and chunk.delta.content:
                        buffer.append(chunk.delta.content)

                    if chunk.delta and chunk.delta.tool_calls:
                        for tool_call in chunk.delta.tool_calls:
                            tool_name = tool_call.name
                            tool_args = tool_call.arguments or "{}"
                            if isinstance(tool_args, str):
                                try:
                                    tool_args = json.loads(tool_args)
                                except json.JSONDecodeError:
                                    logger.warning(
                                        "Invalid JSON for %s: %s", tool_name, tool_args
                                    )
                                    tool_args = {}

                            tool_function = None
                            action_name = None
                            for name, func in self.actions.items():
                                if func.__name__ == tool_name:
                                    tool_function = func
                                    action_name = name
                                    break

                            if tool_function and action_name:
                                await self._send_websocket_message(action_name)
                                pending_tools.append(
                                    (action_name, tool_function, tool_args)
                                )

                yield chunk

        # Capture final LLM response
        self.last_llm_response = "".join(buffer).strip()

        # Execute queued tools and send results
        for action_name, tool_function, tool_args in pending_tools:
            if tool_function not in self.visible_tools:
                logger.info("Skipping execution of %s (not visible)", action_name)
                continue

            try:
                if asyncio.iscoroutinefunction(tool_function):
                    result = await tool_function(**tool_args)
                else:
                    result = tool_function(**tool_args)

                await self._send_websocket_message(
                    action_name, result, tool_func=tool_function
                )
                logger.info("Sent result for %s: %s", action_name, result)

            except Exception as e:
                await self._send_websocket_message(
                    action_name, {"error": str(e)}, tool_func=tool_function
                )
                logger.error("Tool execution failed for %s: %s", action_name, e)

    async def on_enter(self):
        """Speaks immediately after the agent becomes active (e.g., after handover)."""
        await self.session.generate_reply()
