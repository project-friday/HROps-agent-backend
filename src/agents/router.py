# src/agents/router.py
from livekit.agents import Agent, function_tool, RunContext
from livekit import rtc
from src.agents.job_application import JobApplicationAgent
from src.agents.onboarding import OnboardingAgent
from livekit.plugins import assemblyai, elevenlabs,openai, silero,deepgram
from pathlib import Path
from typing import AsyncGenerator,Dict, Any
import logging
from livekit.agents.voice import Agent,ModelSettings
from livekit.plugins import openai, silero, assemblyai
from livekit.plugins import elevenlabs
from livekit.agents import llm
import asyncio
import aiofiles
import json
from pathlib import Path
# from custom.livekit.plugins import murfai
from dotenv import load_dotenv
from livekit import rtc
def load_prompt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8").strip()
ROUTER_INSTRUCTIONS = load_prompt("src/prompts/router.txt")
class RouterAgent(Agent):
    def __init__(self,room:rtc.Room):
        self.room=room
        super().__init__(instructions=ROUTER_INSTRUCTIONS,
                        #  stt=assemblyai.STT(language),
                        stt=deepgram.STT(language='es'),
                        llm=openai.LLM(model="gpt-4.1"),
                        vad=silero.VAD.load(),
                         tts=elevenlabs.TTS(
                # voice_id="wlmwDR77ptH6bKHZui0l",
                # voice_id="H8bdWZHK2OgZwTN7ponr",
                # voice_id="hHjbwzYZW17oh0p05AKv",
                voice_id="kjHz50TasdqbpbfK4uaN",
                model="eleven_turbo_v2_5",
                language='es'
            )
                        # tts=openai.TTS(model="gpt-4o-mini-tts", voice="shimmer"),


                         )
    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool | llm.RawFunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[llm.ChatChunk | str, None]:
            """Custom LLM node that captures full response text."""

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
                        print("🤖 LLM str chunk:", chunk)

                    elif isinstance(chunk, llm.ChatChunk):
                        if chunk.delta and chunk.delta.content:
                            buffer.append(chunk.delta.content)

                        if chunk.delta and chunk.delta.tool_calls:
                            print("🛠️ Tool calls:", chunk.delta.tool_calls)

                            for tool_call in chunk.delta.tool_calls:
                                tool_name = tool_call.name
                                tool_args = tool_call.arguments or "{}"

                                # 🔑 Parse args safely (JSON string → dict)
                                if isinstance(tool_args, str):
                                    try:
                                        tool_args = json.loads(tool_args)
                                    except json.JSONDecodeError:
                                        print(f"⚠️ Invalid JSON for {tool_name}: {tool_args}")
                                        tool_args = {}

                                tool_function = None
                                action_name = None
                                for name, func in self.actions.items():
                                    if func.__name__ == tool_name:
                                        tool_function = func
                                        action_name = name
                                        break

                                if tool_function and action_name:
                                    # Send "action started"
                

                                    # Queue tool execution after LLM finishes
                                    pending_tools.append((action_name, tool_function, tool_args))

                    yield chunk

            # Capture final LLM response
            self.last_llm_response = "".join(buffer).strip()
            print("✅ Full LLM response captured:", self.last_llm_response)

            # # Execute queued tools and send results
            # for action_name, tool_function, tool_args in pending_tools:
            #     if tool_function not in self.visible_tools:
            #         print(f"🚫 Skipping execution of {action_name} (not visible)")
            #         continue

            #     try:
            #         if asyncio.iscoroutinefunction(tool_function):
            #             result = await tool_function(**tool_args)
            #         else:
            #             result = tool_function(**tool_args)

            #         await self._send_websocket_message(action_name, result, tool_func=tool_function)
            #         print(f"✅ Sent result for {action_name}: {result}")

            #     except Exception as e:
            #         await self._send_websocket_message(action_name, {"error": str(e)}, tool_func=tool_function)
            #         print(f"❌ Tool execution failed for {action_name}: {e}")


    @function_tool
    async def go_onboarding(self, context: RunContext[dict]):
        agent = context.session.current_agent
        # Generic, smooth transition
        return (
            OnboardingAgent(room=agent.room, chat_ctx=context.session._chat_ctx),
        )

    @function_tool
    async def go_applications(self, context: RunContext[dict]):
        agent = context.session.current_agent
        return (
            JobApplicationAgent(room=agent.room, chat_ctx=context.session._chat_ctx),
        )

    # --- speaks immediately after the router becomes active ---
    async def on_enter(self):
        await self.session.say("Hey there, I’m Eve, speaking from Walmart Talent Acquisition Team. How may I help you?")

    

#______________________________________________________________________________________________________________#