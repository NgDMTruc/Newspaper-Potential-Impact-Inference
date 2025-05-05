"""Chatbot API endpoints for handling chat interactions.

This module provides endpoints for chat interactions, including regular chat,
streaming chat, message history management, and chat history clearing.
"""

import json
from typing import List

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from fastapi.responses import StreamingResponse

from app.api.v1.auth import get_current_session
from app.core.config import settings
from app.core.langraph.graph import LangGraphAgent
from app.core.langraph.tools.web_scraper import analyze_news_impact
from app.core.limiter import limiter
from app.core.logging import logger
from app.models.session import Session
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    Message,
    StreamResponse,
    NewsImpactRequest,
    NewsImpactResponse,
)

router = APIRouter()
agent = LangGraphAgent()


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat"][0])
async def chat(
    request: Request,
    chat_request: ChatRequest,
    session: Session = Depends(get_current_session),
):
    """Process a chat request using LangGraph.

    Args:
        request: The FastAPI request object for rate limiting.
        chat_request: The chat request containing messages.
        session: The current session from the auth token.

    Returns:
        ChatResponse: The processed chat response.

    Raises:
        HTTPException: If there's an error processing the request.
    """
    try:
        logger.info(
            "chat_request_received",
            session_id=session.id,
            message_count=len(chat_request.messages),
        )

        # Process the request through the LangGraph
        result = await agent.get_response(chat_request.messages, session.id, user_id=session.user_id)

        logger.info("chat_request_processed", session_id=session.id)

        return ChatResponse(messages=result)
    except Exception as e:
        logger.error("chat_request_failed", session_id=session.id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


import requests

@router.post("/news/impact", response_model=NewsImpactResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat"][0])
async def analyze_news_impact_endpoint(
    request: Request,
    news_request: NewsImpactRequest,
    # session: Session = Depends(get_current_session),
):
    """Analyze the potential impact of a news article using Ollama LLM API directly."""
    try:
        # Compose the prompt for LLM
        prompt = (
            f"Analyze the potential impact of this news article on the {news_request.field} sector. "
            f"URL: {news_request.url}\n"
            f"Please extract the main content and provide:\n"
            f"- Key events or developments\n- Potential short-term and long-term consequences\n- Specific insights related to {news_request.field}"
        )
        ollama_url = "http://localhost:11434/v1/chat/completions"
        payload = {
            "model": "deepseek-r1:1.5b",  # or your preferred model name
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        response = requests.post(ollama_url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        llm_content = data["choices"][0]["message"]["content"]
        return NewsImpactResponse(content=llm_content)
    except Exception as e:
        logger.error("news_impact_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to analyze news impact: " + str(e))

    try:
        # logger.info(
        #     "news_impact_request_received",
        #     session_id=session.id,
        #     url=news_request.url,
        #     field=news_request.field,
        # )

        # Analyze the news impact
        question = analyze_news_impact(str(news_request.url), news_request.field)
        # result = await agent.get_response(question)

        if question is None:
            raise HTTPException(status_code=400, detail="Unable to extract or analyze the news article content.")

    #     logger.info("news_impact_request_processed", session_id=session.id)

        return NewsImpactResponse(content=question)
    except Exception as e:
    #     logger.error("news_impact_request_failed", session_id=session.id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat_stream"][0])
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    session: Session = Depends(get_current_session),
):
    """Process a chat request using LangGraph with streaming response.

    Args:
        request: The FastAPI request object for rate limiting.
        chat_request: The chat request containing messages.
        session: The current session from the auth token.

    Returns:
        StreamingResponse: A streaming response of the chat completion.

    Raises:
        HTTPException: If there's an error processing the request.
    """
    try:
        logger.info(
            "stream_chat_request_received",
            session_id=session.id,
            message_count=len(chat_request.messages),
        )

        async def event_generator():
            """Generate streaming events.

            Yields:
                str: Server-sent events in JSON format.

            Raises:
                Exception: If there's an error during streaming.
            """
            try:
                full_response = ""
                async for chunk in agent.get_stream_response(
                    chat_request.messages, session.id, user_id=session.user_id
                ):
                    full_response += chunk
                    response = StreamResponse(content=chunk, done=False)
                    yield f"data: {json.dumps(response.model_dump())}\n\n"

                # Send final message indicating completion
                final_response = StreamResponse(content="", done=True)
                yield f"data: {json.dumps(final_response.model_dump())}\n\n"

            except Exception as e:
                logger.error(
                    "stream_chat_request_failed",
                    session_id=session.id,
                    error=str(e),
                    exc_info=True,
                )
                error_response = StreamResponse(content=str(e), done=True)
                yield f"data: {json.dumps(error_response.model_dump())}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error(
            "stream_chat_request_failed",
            session_id=session.id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/messages", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["messages"][0])
async def get_session_messages(
    request: Request,
    session: Session = Depends(get_current_session),
):
    """Get all messages for a session.

    Args:
        request: The FastAPI request object for rate limiting.
        session: The current session from the auth token.

    Returns:
        ChatResponse: All messages in the session.

    Raises:
        HTTPException: If there's an error retrieving the messages.
    """
    try:
        messages = await agent.get_chat_history(session.id)
        return ChatResponse(messages=messages)
    except Exception as e:
        logger.error("get_messages_failed", session_id=session.id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/messages")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["messages"][0])
async def clear_chat_history(
    request: Request,
    session: Session = Depends(get_current_session),
):
    """Clear all messages for a session.

    Args:
        request: The FastAPI request object for rate limiting.
        session: The current session from the auth token.

    Returns:
        dict: A message indicating the chat history was cleared.
    """
    try:
        await agent.clear_chat_history(session.id)
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        logger.error("clear_chat_history_failed", session_id=session.id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))