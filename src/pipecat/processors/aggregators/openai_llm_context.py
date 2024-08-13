#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from dataclasses import dataclass
import io
import json

from typing import List

from PIL import Image

from pipecat.frames.frames import Frame, VisionImageRawFrame, FunctionCallInProgressFrame, FunctionCallResultFrame
from pipecat.processors.frame_processor import FrameProcessor


from openai._types import NOT_GIVEN, NotGiven

from openai.types.chat import (
    ChatCompletionToolParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionMessageParam
)

# JSON custom encoder to handle bytes arrays so that we can log contexts
# with images to the console.


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, io.BytesIO):
            # Convert the first 8 bytes to an ASCII hex string
            return (f"{obj.getbuffer()[0:8].hex()}...")
        return super().default(obj)


class OpenAILLMContext:

    def __init__(
        self,
        messages: List[ChatCompletionMessageParam] | None = None,
        tools: List[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN
    ):
        self.messages: List[ChatCompletionMessageParam] = messages if messages else [
        ]
        self.tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = tool_choice
        self.tools: List[ChatCompletionToolParam] | NotGiven = tools

    @staticmethod
    def from_messages(messages: List[dict]) -> "OpenAILLMContext":
        context = OpenAILLMContext()

        for message in messages:
            if "name" not in message:
                message["name"] = message["role"]
            context.add_message(message)
        return context

    @staticmethod
    def from_image_frame(frame: VisionImageRawFrame) -> "OpenAILLMContext":
        """
        For images, we are deviating from the OpenAI messages shape. OpenAI
        expects images to be base64 encoded, but other vision models may not.
        So we'll store the image as bytes and do the base64 encoding as needed
        in the LLM service.
        """
        context = OpenAILLMContext()
        buffer = io.BytesIO()
        Image.frombytes(
            frame.format,
            frame.size,
            frame.image
        ).save(
            buffer,
            format="JPEG")
        context.add_message({
            "content": frame.text,
            "role": "user",
            "data": buffer,
            "mime_type": "image/jpeg"
        })
        return context

    def add_message(self, message: ChatCompletionMessageParam):
        self.messages.append(message)

    def get_messages(self) -> List[ChatCompletionMessageParam]:
        return self.messages

    def get_messages_json(self) -> str:
        return json.dumps(self.messages, cls=CustomEncoder)

    def set_tool_choice(
        self, tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven
    ):
        self.tool_choice = tool_choice

    def set_tools(self, tools: List[ChatCompletionToolParam] | NotGiven = NOT_GIVEN):
        if tools != NOT_GIVEN and len(tools) == 0:
            tools = NOT_GIVEN

        self.tools = tools
    
    async def call_function(
            self,
            f: callable,
            *,
            function_name: str,
            tool_call_id: str,
            arguments: str,
            llm: FrameProcessor) -> None:

        # Push a SystemFrame downstream. This frame will let our assistant context aggregator
        # know that we are in the middle of a function call. Some contexts/aggregators may
        # not need this. But some definitely do (Anthropic, for example).
        await llm.push_frame(FunctionCallInProgressFrame(
            function_name=function_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
        ))

        # Define a callback function that pushes a FunctionCallResultFrame downstream.
        async def function_call_result_callback(result):
            await llm.push_frame(FunctionCallResultFrame(
                function_name=function_name,
                tool_call_id=tool_call_id,
                arguments=arguments,
                result=result))
        await f(function_name=function_name, tool_call_id=tool_call_id, arguments=arguments,
                context=self, result_callback=function_call_result_callback)


@dataclass
class OpenAILLMContextFrame(Frame):
    """Like an LLMMessagesFrame, but with extra context specific to the OpenAI
    API. The context in this message is also mutable, and will be changed by the
    OpenAIContextAggregator frame processor.

    """
    context: OpenAILLMContext
