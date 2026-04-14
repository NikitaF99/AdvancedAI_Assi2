
import os
from typing import Optional, Any
import torch
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

def clean_text(text: str) -> str:
    if not text:
        return ""
    return text.replace("\u0120", " ").replace("\u010a", "\n").strip()


def get_groq_api_key() -> Optional[str]:
    load_dotenv()
    return os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API")