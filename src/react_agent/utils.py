"""Utility & helper functions."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
import requests
import json
import os
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv

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


def get_model_output(api_url, yaml_description=None):
    """
    Get data from API and process it with task model.
    
    Args:
        api_url: API endpoint URL
        input_data: Data to send to API
        yaml_description: YAML description for generating fake data if API fails
        
    Returns:
        Processed output from the task model
    """
    try:
        # If we have YAML description, use it to generate additional fake data
        if yaml_description:
            fake_data = generate_fake_data_with_openai(yaml_description)
            response = requests.post(api_url, json=fake_data)          
        
        response.raise_for_status()
        api_data = response.json()

        return api_data
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request to {api_url}: {e}")
        
        # If API fails and we have YAML description, generate fake data as fallback
        if yaml_description:
            print("Falling back to generating fake data...")
            return generate_fake_data_with_openai(yaml_description)
        
        return {"error": str(e)}

def generate_fake_data_with_openai(yaml_description: dict, num_samples: int = 1, model_name: str = "openai/gpt-4.1-nano-2025-04-14") -> Union[dict, list]:
    """
    Generate fake data using OpenAI API based on YAML format description.
    
    Args:
        yaml_description: YAML dict containing format description
        num_samples: Number of fake data samples to generate
        model_name: OpenAI model to use (format: "openai/model_name")
        
    Returns:
        Generated fake data matching the specified format
    """
    try:
        # Extract format information from YAML
        format_info = extract_format_from_yaml(yaml_description)
        
        # Create prompt for OpenAI
        prompt = create_data_generation_prompt(format_info, num_samples)
        
        # Load OpenAI model
        chat_model = load_chat_model(model_name)
        
        # Generate data
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content=prompt)]
        response = chat_model.invoke(messages)
        
        # Parse response
        generated_text = get_message_text(response)
        
        # Try to parse as JSON
        try:
            fake_data = json.loads(generated_text)
            return fake_data
        except json.JSONDecodeError:
            # If not valid JSON, return as text
            return {"generated_data": generated_text}
            
    except Exception as e:
        print(f"Error generating fake data: {e}")
        return {"error": str(e)}


def extract_format_from_yaml(yaml_description: dict) -> str:
    """Extract format information from YAML description."""
    format_parts = []
    
    if 'task_description' in yaml_description:
        task_desc = yaml_description['task_description']
    if isinstance(task_desc, dict) and 'description' in task_desc:
            format_parts.append(f"Task: {task_desc['description']}")
    
    format_parts.append(f"Input Format: {yaml_description['model_information']['input_format']}")
    
    return "\n".join(format_parts)


def create_data_generation_prompt(format_info: str, num_samples: int) -> str:
    """Create a prompt for OpenAI to generate fake data."""
    prompt = f"""
Based on the following format description, generate {num_samples} realistic fake data samples.

Format Description:
{format_info}

Requirements:
1. Generate exactly {num_samples} samples
2. Follow the specified format exactly
3. Use realistic, diverse data values
4. Return the result as valid JSON
5. Make sure all required fields are included
6. Use appropriate data types (strings, numbers, booleans, etc.)

Please return only the JSON data without any additional text or explanation.
"""
    return prompt