import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Load environment variables from .env file
load_dotenv()


def get_llm():
    # Read API key and model name from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    # Fail fast if API key is missing
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is missing in .env")

    # Initialize and return Claude chat model
    return ChatAnthropic(
        model=model,
        api_key=api_key,
        temperature=0,  # Deterministic output
    )
