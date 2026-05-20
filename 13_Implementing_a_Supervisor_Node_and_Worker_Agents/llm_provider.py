import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Load environment variables
load_dotenv()


def get_llm():
    # Read Anthropic API key
    api_key = os.getenv("ANTHROPIC_API_KEY")

    # Read model name
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    # Validate API key presence
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is missing in .env")

    # Return configured Claude LLM
    return ChatAnthropic(
        model=model,
        api_key=api_key,
        temperature=0,
    )
