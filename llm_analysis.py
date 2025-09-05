import requests

def analyze_transcription_with_ollama(transcript: str):
    prompt = f"""
You are a conversation analysis AI, made specially for real estate Industry. Analyze the following sales call transcript. Also How the agent is talking to the customer and how much the customer is inclined towards buying the property.

1. Summarize the conversation in 3-5 lines.
2. Rate the agent on a scale of 1-5 for:
   - Introduction
   - Product Knowledge
   - Objection Handling
   - Tone & Language
   - Closure Strategy

Transcript:
\"\"\"{transcript}\"\"\"

Respond in structured JSON.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "nous-hermes2",
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()
    return result["response"]
