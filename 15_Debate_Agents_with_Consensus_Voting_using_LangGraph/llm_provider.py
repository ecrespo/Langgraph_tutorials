import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()


def get_llm():
    # Read API key and model name from environment
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Fail fast if API key is missing
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing in .env")

    # Initialize and return Gemini chat model
    return ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key,
        temperature=0  # Deterministic output
    )
