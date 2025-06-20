"""Utility & helper functions."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


import os
from typing import Any, Dict, List, Optional
from langchain_core.language_models import BaseChatModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_chat_model(model_name: str) -> BaseChatModel:
    """Load a chat model based on the model name."""
    provider, model = model_name.split("/", 1)
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        # Explicitly pass the API key from environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return ChatOpenAI(model=model, temperature=0, api_key=api_key)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        return ChatAnthropic(model=model, temperature=0, api_key=api_key)
    # Add other providers as needed
    
    raise ValueError(f"Unsupported model provider: {provider}")

def extract_description(yaml):
    return yaml['task_description']['description']

    