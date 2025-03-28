"""
Minimal AgentLLM implementation that integrates Agno Agents with Pipecat.
This version focuses on direct agent.arun() execution in a separate task.
"""

from typing import Optional
import asyncio

from agno.agent import Agent

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    StartFrame,
    StartInterruptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ai_services import LLMService
from loguru import logger


class AgentLLM(LLMService):
    """AgentLLM service that integrates Agno agents with Pipecat.
    
    This service uses Agno Agent to process LLM messages and run the agent in an async task.
    It handles interruptions by canceling the running task.
    """

    def __init__(
        self,
        *,
        agent: Agent,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._agent = agent
        self._model_name = agent.model.id if agent.model else "unknown"
        self.set_model_name(self._model_name)
        self._llm_task: Optional[asyncio.Task] = None

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        # Initialize the agent if needed
        if not self._agent.model:
            raise ValueError("Agent must have a model set")
        
        # Ensure the agent has the right tools
        self._agent.update_model()
        
    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        # Cancel any ongoing tasks
        await self._cancel_llm_task()
        
    async def _cancel_llm_task(self):
        """Cancel the LLM task if it exists."""
        if self._llm_task:
            logger.debug(f"{self} Cancelling LLM task")
            await self.cancel_task(self._llm_task)
            self._llm_task = None

    async def _handle_interruptions(self, frame: StartInterruptionFrame):
        """Handle interruptions by canceling the running task."""
        await super()._handle_interruptions(frame)
        await self._cancel_llm_task()
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and route them appropriately."""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, LLMMessagesFrame):
            # Start LLM processing in a separate task
            await self._cancel_llm_task()  # Cancel existing task if any
            self._llm_task = self.create_task(self._process_llm_messages(frame))
            logger.debug(f"{self} Created LLM task")

    async def _process_llm_messages(self, frame: LLMMessagesFrame):
        """Process LLM messages frame and run the agent."""
        try:
            # Signal the start of a response
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
            
            # Start TTFB metrics
            await self.start_ttfb_metrics()
            
            # Simply pass the messages directly to agent.arun() - Agno will handle the message processing
            response_iter = self._agent.arun(frame.messages, stream=True)
            
            async for response in response_iter:
                # Stop TTFB metrics after the first response
                await self.stop_ttfb_metrics()
                
                # Stream text content to LLMTextFrame
                if isinstance(response.content, str) and response.content:
                    await self.push_frame(LLMTextFrame(response.content))
            
            # Signal end metrics and end of response
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())
            
        except asyncio.CancelledError:
            logger.info(f"{self} LLM task was cancelled")
            # Don't push end frame if canceled
            raise
            
        except Exception as e:
            logger.exception(f"Error processing LLM messages: {e}")
            await self.push_error(ErrorFrame(f"Error processing LLM messages: {e}"))
            await self.push_frame(LLMFullResponseEndFrame())
