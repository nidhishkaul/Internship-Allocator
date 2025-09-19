# resume_parser.py
import re, json, os, io
from groq import Groq
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st

load_dotenv()
try:
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_KEY = os.getenv("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None


def extract_text_from_pdf_bytes(b: bytes) -> str:
    """Extract text from PDF bytes safely."""
    try:
        reader = PdfReader(io.BytesIO(b))
        return " ".join([p.extract_text() or "" for p in reader.pages])
    except Exception:
        return ""


def safe_json_from_llm(text: str, model_name: str = "llama-3.1-8b-instant"):
    """Send resume text to Groq LLM and get structured JSON back."""
    if not groq_client:
        raise RuntimeError("GROQ_API_KEY not set in environment.")

    cleaned = re.sub(r"\s+", " ", text).strip()

    prompt = f"""
You are a resume parser. Read the resume text and return ONLY a valid JSON object with the exact keys:

name, email, education, skills, sector_interests, location_pref, experience, projects

Rules:
- If a field is not present, set its value to an empty string ("").
- All fields must be plain text strings (no arrays, no objects).
- For multiple entries (education, skills, experience, projects), return a single string with entries separated by semicolons.
- Do not invent data. Use only what is in the text.
- Return only valid JSON, no extra text, no markdown.

Resume text:
{cleaned}
"""

    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful structured-data extractor."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=800,
    )

    raw = response.choices[0].message.content.strip()

    # Extract first {...}
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError("LLM did not return JSON-like output.")

    json_str = m.group()
    try:
        parsed = json.loads(json_str)
    except Exception as e:
        raise ValueError("Failed to parse JSON from LLM output: " + str(e))

    # Ensure all keys exist
    for k in [
        "name",
        "email",
        "education",
        "skills",
        "sector_interests",
        "location_pref",
        "experience",
        "projects",
    ]:
        parsed.setdefault(k, "")

    return parsed


def fallback_extract(text: str):
    """Fallback extractor if LLM is unavailable."""
    t = text.lower()

    # Email
    email_match = re.search(r"[a-z0-9_.+-]+@[a-z0-9-]+\.[a-z0-9-.]+", t)
    email = email_match.group(0) if email_match else ""

    # Simple skills dictionary
    KEYS = [
        "python", "sql", "excel", "pandas", "numpy", "machine learning",
        "flutter", "dart", "git", "aws", "tensorflow", "react", "canva"
    ]
    skills = [k for k in KEYS if k in t]

    # Education (very naive)
    if "b.tech" in t or "btech" in t:
        edu = "B.Tech"
    elif "m.tech" in t or "mtech" in t:
        edu = "M.Tech"
    elif "mba" in t:
        edu = "MBA"
    else:
        edu = ""

    # Experience (basic heuristic)
    exp_m = re.search(r"(\d+)\s+year", t)
    exp = exp_m.group(0) if exp_m else "Fresher"

    return {
        "name": "",
        "email": email,
        "education": edu,
        "skills": ", ".join(skills),
        "sector_interests": "",
        "location_pref": "",
        "experience": exp,
        "projects": "",
    }


def parse_resume_with_llm(file_bytes: bytes):
    """Main entrypoint for Streamlit. Takes resume file bytes and returns parsed JSON."""
    try:
        text = extract_text_from_pdf_bytes(file_bytes)
        if not text.strip():
            text = file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        text = file_bytes.decode("utf-8", errors="ignore")

    # Try LLM
    try:
        return safe_json_from_llm(text)
    except Exception:
        # Fallback if LLM fails
        return fallback_extract(text)
