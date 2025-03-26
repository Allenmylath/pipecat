"""
AgentLLMService - Integration of Agno Agent with Pipecat

This module provides integration between the Agno Agent framework and Pipecat's
frame-based architecture. The AgentLLMService converts between Pipecat frames and
Agno Agent Messages, allowing the powerful Agent capabilities to be used within
the Pipecat pipeline.

Example usage:

```python
from pipecat.services.agent import AgentLLMService

# Create an AgentLLMService with default settings
agent_service = AgentLLMService(
    model="gpt-4o",
    reasoning=True,  # Enable step-by-step reasoning
    memory=True      # Enable conversation memory
)

# Register it with your pipeline
pipeline.register_processor(agent_service)

# Or manually register tools
agent_service.register_function("search_web", search_web_function)
```

The service handles conversion between Pipecat's frame-based architecture and
Agno's Agent-based system, enabling:

1. Persistent memory and context across interactions
2. Step-by-step reasoning capabilities
3. Knowledge retrieval and tool usage
4. Structured outputs through function calling
"""

import asyncio
import json
from collections import deque
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, cast

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    VisionImageRawFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService

try:
    from agno.agent.agent import Agent
    from agno.media import Image
    from agno.models.base import Model
    from agno.models.message import Message
    from agno.run.messages import RunMessages
    from agno.run.response import RunEvent, RunResponse
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use AgentLLMService, you need to `pip install agno`. Set appropriate environment variables for your model providers."
    )
    raise Exception(f"Missing module: {e}")


class AgentLLMService(LLMService):
    """
    AgentLLMService integrates the agno Agent framework with pipecat's LLMService architecture.
    
    This service allows you to use agno Agent capabilities (memory, reasoning, knowledge, etc.)
    within the pipecat frame-based architecture. It converts frames to formats that the Agent
    can understand and translates Agent responses back into frames.
    """

    def __init__(
        self,
        *,
        agent: Optional[Agent] = None,
        agent_config: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        reasoning: bool = False,
        memory: bool = True,
        storage_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the AgentLLMService.
        
        Args:
            agent: An existing Agent instance to use
            agent_config: Configuration to create a new Agent if one is not provided
            model: Model identifier to use if agent_config does not specify one
            reasoning: Enable agent reasoning (step-by-step thinking)
            memory: Enable agent memory to remember conversation
            storage_config: Configuration for Agent storage backend
            **kwargs: Additional arguments passed to LLMService
        """
        super().__init__(**kwargs)
        
        # Initialize agent or create one with provided config
        self.agent = agent
        if self.agent is None:
            # Prepare agent config
            config = agent_config or {}
            
            # Configure storage if requested
            if storage_config and not config.get("storage"):
                try:
                    from agno.storage.memory import MemoryStorage
                    storage = MemoryStorage(**storage_config)
                    config["storage"] = storage
                except ImportError as e:
                    logger.warning(f"Failed to create storage: {e}")
            
            # Configure memory
            if memory and not config.get("memory"):
                try:
                    from agno.memory.agent import AgentMemory
                    config["memory"] = AgentMemory()
                except ImportError as e:
                    logger.warning(f"Failed to create memory: {e}")
            
            # Configure reasoning
            if reasoning:
                config["reasoning"] = True
            
            # Configure model
            if model and not config.get("model"):
                model_instance = self._create_model_for_agent(model)
                config["model"] = model_instance
            
            # Create agent instance
            self.agent = Agent(**config)
            self.agent.initialize_agent()
        
        # Set model name from the agent's model
        if self.agent.model:
            self.set_model_name(self.agent.model.id)
        
        # Map to store function calls in progress
        self._function_calls_in_progress = {}
        
        # Semaphore to control concurrent agent runs
        self._agent_semaphore = asyncio.Semaphore(1)
        
        # Last run response from the agent
        self._last_response = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate metrics."""
        return True

    def _create_model_for_agent(self, model_id: str) -> Model:
        """Create a model instance for the agent based on the provided model ID."""
        if model_id.startswith("gpt-"):
            # OpenAI model
            from agno.models.openai import OpenAIChat
            return OpenAIChat(id=model_id)
        elif model_id.startswith("claude-"):
            # Anthropic model
            from agno.models.anthropic import Claude
            return Claude(id=model_id)
        else:
            # Default to OpenAI
            from agno.models.openai import OpenAIChat
            return OpenAIChat(id=model_id)

    async def set_model(self, model: str):
        """Update the model used by the Agent."""
        if self.agent and self.agent.model:
            if hasattr(self.agent.model, "id"):
                self.agent.model.id = model
                self.set_model_name(model)
            else:
                # Create a new model
                new_model = self._create_model_for_agent(model)
                self.agent.model = new_model
                self.set_model_name(model)
        else:
            # Create a new model and set it on the agent
            new_model = self._create_model_for_agent(model)
            if self.agent:
                self.agent.model = new_model
            self.set_model_name(model)
            
    def get_agent_status(self) -> Dict[str, Any]:
        """Get the current status of the Agent.
        
        Useful for diagnostics and monitoring.
        """
        status = {
            "service_type": "AgentLLMService",
            "model_name": self.model_name,
            "connected": self.agent is not None
        }
        
        if self.agent:
            status.update({
                "agent_id": self.agent.agent_id,
                "session_id": self.agent.session_id,
                "reasoning_enabled": self.agent.reasoning,
                "memory_enabled": self.agent.memory is not None,
                "storage_enabled": self.agent.storage is not None,
                "num_tools": len(self.agent.tools) if self.agent.tools else 0
            })
            
            # Add memory stats if available
            if self.agent.memory:
                status["memory_stats"] = {
                    "num_messages": len(self.agent.memory.messages) if self.agent.memory.messages else 0,
                    "num_runs": len(self.agent.memory.runs) if self.agent.memory.runs else 0,
                    "has_summary": self.agent.memory.summary is not None
                }
                
        return status
        
    async def flush_function_calls(self):
        """Clear all function calls in progress."""
        self._function_calls_in_progress.clear()
        
    async def sync_tools_with_agent(self):
        """Synchronize registered functions with the Agent's tools.
        
        This ensures that any tools registered with the LLMService
        are also available to the Agent.
        """
        if not self.agent or not hasattr(self.agent, 'tools'):
            return
            
        # Convert registered functions to Agent-compatible tools
        agent_tools = []
        
        if self._callbacks:
            for function_name, callback in self._callbacks.items():
                if function_name is not None:  # Skip the catch-all handler (None)
                    # Create a Function object that can be used by the Agent
                    try:
                        from agno.tools.function import Function
                        
                        # Create a wrapper function that can be used by the Agent
                        def create_wrapper(fn_name, cb):
                            async def wrapper(*args, **kwargs):
                                # This wrapper will be called by the Agent
                                result = await cb(fn_name, kwargs, None, None)
                                return result
                            return wrapper
                            
                        wrapper_fn = create_wrapper(function_name, callback)
                        wrapper_fn.__name__ = function_name
                        
                        # Create a Function object with the wrapper
                        func = Function.from_callable(wrapper_fn)
                        agent_tools.append(func)
                    except Exception as e:
                        logger.warning(f"Failed to convert function {function_name} to Agent tool: {e}")
        
        # Update the Agent's tools
        if agent_tools:
            if self.agent.tools is None:
                self.agent.tools = agent_tools
            else:
                # Add tools that aren't already in the Agent's tools
                existing_names = set()
                if isinstance(self.agent.tools, list):
                    for tool in self.agent.tools:
                        if hasattr(tool, 'name'):
                            existing_names.add(tool.name)
                        elif callable(tool) and hasattr(tool, '__name__'):
                            existing_names.add(tool.__name__)
                
                for tool in agent_tools:
                    if hasattr(tool, 'name') and tool.name not in existing_names:
                        self.agent.tools.append(tool)
                        
            # Update the Agent's model with the new tools
            if self.agent.model:
                self.agent.update_model()

    async def _convert_context_to_messages(self, context: OpenAILLMContext) -> List[Message]:
        """Convert an OpenAILLMContext to a list of agno Message objects."""
        messages = context.get_messages()
        result_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            msg_content = msg.get("content", "")
            msg_name = msg.get("name")
            images = []
            text_content = ""
            
            # Process content that might be a list (multimodal)
            if isinstance(msg_content, list):
                for item in msg_content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_content += item.get("text", "")
                        elif item.get("type") == "image_url":
                            # Handle image URLs
                            image_url = item.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:image/"):
                                # Handle base64 encoded images
                                import base64
                                import io
                                from PIL import Image as PILImage
                                
                                # Extract format and base64 content
                                parts = image_url.split(";base64,")
                                if len(parts) == 2:
                                    format_str = parts[0].split("/")[1]
                                    base64_data = parts[1]
                                    
                                    # Decode base64 to bytes
                                    image_bytes = base64.b64decode(base64_data)
                                    
                                    # Create image object
                                    img_buffer = io.BytesIO(image_bytes)
                                    pil_image = PILImage.open(img_buffer)
                                    
                                    # Create agno Image object
                                    images.append(Image(
                                        content=image_bytes,
                                        format=format_str,
                                        size=pil_image.size
                                    ))
            else:
                text_content = msg_content
            
            # Create the message with proper attributes
            message_kwargs = {
                "role": role,
                "content": text_content,
            }
            
            if msg_name:
                message_kwargs["name"] = msg_name
                
            if images:
                message_kwargs["images"] = images
                
            result_messages.append(Message(**message_kwargs))
        
        return result_messages

    async def _convert_image_to_agent_image(self, frame: VisionImageRawFrame) -> Image:
        """Convert a VisionImageRawFrame to an agno Image object."""
        return Image(
            content=frame.image,
            format=frame.format,
            size=frame.size,
            alt_text=frame.text
        )

    async def _update_settings(self, settings: Dict[str, Any]):
        """Update service settings."""
        for key, value in settings.items():
            if key == "model":
                await self.set_model(value)
            elif key == "reasoning":
                if self.agent:
                    self.agent.reasoning = bool(value)
            elif key == "memory":
                if value and self.agent and self.agent.memory is None:
                    from agno.memory.agent import AgentMemory
                    self.agent.memory = AgentMemory()
            else:
                logger.info(f"Setting {key}={value} for agent")
                if self.agent:
                    setattr(self.agent, key, value)

    async def _process_context_frame(self, context: OpenAILLMContext, images: List[Image] = None):
        """Process an OpenAILLMContext frame by running the agent and streaming the results back as frames."""
        
        # Use semaphore to ensure only one agent run at a time
        async with self._agent_semaphore:
            try:
                # Signal start of LLM response
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_ttfb_metrics()
                
                # Convert context to agent messages
                messages = await self._convert_context_to_messages(context)
                
                # Make sure we have the latest tools synced with the agent
                await self.sync_tools_with_agent()
                
                # Add images to the messages if provided
                if images:
                    if len(messages) > 0 and messages[-1].role == "user":
                        # Add images to the last user message
                        messages[-1].images = images
                    else:
                        # Create a new user message with images
                        messages.append(Message(role="user", content="", images=images))
                
                # Check if we have a primary message or need to use messages as a list
                if len(messages) == 0:
                    logger.warning("No messages to send to agent")
                    await self.push_frame(ErrorFrame("No messages to send to agent"))
                    await self.push_frame(LLMFullResponseEndFrame())
                    return
                
                primary_message = messages[-1]  # Last message is primary
                
                # Process with agent
                if self.agent.stream:
                    # Stream the agent response
                    async for response_chunk in self.agent.arun(primary_message, messages=messages, stream=True):
                        await self.stop_ttfb_metrics()
                        
                        # Handle tool calls
                        if (response_chunk.tools and 
                            response_chunk.event != RunEvent.tool_call_completed.value):
                            for tool_call in response_chunk.tools:
                                tool_call_id = tool_call.get("tool_call_id")
                                function_name = tool_call.get("function", {}).get("name")
                                arguments = tool_call.get("function", {}).get("arguments", "{}")
                                
                                if function_name and tool_call_id:
                                    await self.push_frame(
                                        FunctionCallInProgressFrame(
                                            function_name=function_name,
                                            tool_call_id=tool_call_id,
                                            arguments=json.loads(arguments) if isinstance(arguments, str) else arguments
                                        )
                                    )
                                    
                                    # Call the function if it's registered
                                    if self.has_function(function_name):
                                        self._function_calls_in_progress[tool_call_id] = function_name
                                        await self.call_function(
                                            context=context,
                                            function_name=function_name,
                                            arguments=json.loads(arguments) if isinstance(arguments, str) else arguments,
                                            tool_call_id=tool_call_id,
                                            run_llm=False
                                        )
                        
                        # Handle content
                        if response_chunk.content and isinstance(response_chunk.content, str):
                            await self.push_frame(LLMTextFrame(response_chunk.content))
                else:
                    # Non-streaming response
                    if len(messages) > 1:
                        # Process with full message context
                        response = await self.agent.arun(messages=messages, stream=False)
                    else:
                        # Process with single message
                        response = await self.agent.arun(primary_message, stream=False)
                    
                    await self.stop_ttfb_metrics()
                    self._last_response = response
                    
                    # Handle tool calls
                    if response.tools:
                        for tool_call in response.tools:
                            tool_call_id = tool_call.get("tool_call_id")
                            function_name = tool_call.get("function", {}).get("name")
                            arguments = tool_call.get("function", {}).get("arguments", "{}")
                            
                            if function_name and tool_call_id:
                                await self.push_frame(
                                    FunctionCallInProgressFrame(
                                        function_name=function_name,
                                        tool_call_id=tool_call_id,
                                        arguments=json.loads(arguments) if isinstance(arguments, str) else arguments
                                    )
                                )
                                
                                # Call the function if it's registered
                                if self.has_function(function_name):
                                    self._function_calls_in_progress[tool_call_id] = function_name
                                    await self.call_function(
                                        context=context,
                                        function_name=function_name,
                                        arguments=json.loads(arguments) if isinstance(arguments, str) else arguments,
                                        tool_call_id=tool_call_id,
                                        run_llm=False
                                    )
                                else:
                                    logger.warning(f"Function {function_name} not registered with service")
                    
                    # Handle content
                    if response.content is not None:
                        if isinstance(response.content, str):
                            await self.push_frame(LLMTextFrame(response.content))
                        else:
                            # Handle structured output
                            try:
                                json_str = json.dumps(response.content)
                                await self.push_frame(LLMTextFrame(json_str))
                            except Exception as e:
                                logger.error(f"Error converting response content to JSON: {e}")
                                await self.push_frame(LLMTextFrame(str(response.content)))
                    else:
                        # No content in the response
                        logger.warning("No content in the agent response")
                        await self.push_frame(LLMTextFrame("No content in response"))
                
                # Estimate token usage (approximate)
                if self._last_response:
                    # Try to get metrics from agent if available
                    metrics = self._last_response.metrics or {}
                    prompt_tokens = metrics.get("prompt_tokens", [0])
                    completion_tokens = metrics.get("completion_tokens", [0])
                    
                    # Sum them if they're lists, or use as is
                    p_tokens = sum(prompt_tokens) if isinstance(prompt_tokens, list) else prompt_tokens
                    c_tokens = sum(completion_tokens) if isinstance(completion_tokens, list) else completion_tokens
                    
                    tokens = LLMTokenUsage(
                        prompt_tokens=p_tokens,
                        completion_tokens=c_tokens,
                        total_tokens=p_tokens + c_tokens
                    )
                    await self.start_llm_usage_metrics(tokens)
                    
                # Also handle any memory updates from agent to ensure persistence
                if self.agent and self.agent.memory and self.agent.storage:
                    try:
                        self.agent.write_to_storage()
                    except Exception as e:
                        logger.warning(f"Failed to update agent storage: {e}")
                
                # Signal the end of a response
                await self.push_frame(LLMFullResponseEndFrame())
                
            except Exception as e:
                logger.error(f"Error running agent: {e}")
                await self.push_frame(ErrorFrame(f"Error running agent: {e}"))
                await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and route them to appropriate handlers."""
        await super().process_frame(frame, direction)

        try:
            if isinstance(frame, OpenAILLMContextFrame):
                # Process an OpenAILLMContext frame
                context = frame.context
                await self._process_context_frame(context)
                
            elif isinstance(frame, LLMMessagesFrame):
                # Convert LLMMessagesFrame to OpenAILLMContext and process
                context = OpenAILLMContext.from_messages(frame.messages)
                await self._process_context_frame(context)
                
            elif isinstance(frame, VisionImageRawFrame):
                # Handle image frames by creating a context with the image
                context = OpenAILLMContext()
                agent_image = await self._convert_image_to_agent_image(frame)
                # For simplicity, we'll add this as a user message with just the image
                context.add_message({"role": "user", "content": frame.text or ""})
                
                # Now process the context
                await self._process_context_frame(context, images=[agent_image])
                
            elif isinstance(frame, LLMUpdateSettingsFrame):
                # Handle settings updates
                await self._update_settings(frame.settings)
                
            elif isinstance(frame, FunctionCallResultFrame):
                # Handle function call results
                await self._handle_function_call_result(frame)
                
            elif isinstance(frame, CancelFrame):
                # Handle cancellation
                await self.flush_function_calls()
                # Pass through the cancel frame
                await self.push_frame(frame, direction)
                
            elif isinstance(frame, EndFrame):
                # Handle end frame - clean up resources
                await self.flush_function_calls()
                # Pass through the end frame
                await self.push_frame(frame, direction)
                
            else:
                # Pass through any other frames
                await self.push_frame(frame, direction)
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            await self.push_frame(ErrorFrame(f"Error processing frame: {e}"), direction)
