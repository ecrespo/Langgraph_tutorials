import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()


def get_llm():
    # Read Gemini API key
    api_key = os.getenv("GEMINI_API_KEY")

    # Read Gemini model name (default if not set)
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Ensure API key is available
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing in .env")

    # Return configured Gemini LLM
    return ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key,
        temperature=0
    )
