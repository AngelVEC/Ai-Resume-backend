import os
from typing import Iterator
from google import genai
from google.genai import types

#initialize the gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

#for system prompt
SYSTEM_PROMPT = """You are an expert resume writer and career coach. Your job is to help users tailor their resume to a specific job description.

When given a resume and a job description:
1. Analyse the key skills, requirements, and keywords in the job description.
2. Rewrite and restructure the resume to highlight the most relevant experience, skills, and achievements.
3. Use strong action verbs and quantify achievements where possible.
4. Mirror the language and terminology used in the job description.
5. Keep the resume concise, professional, and ATS-friendly.

CRITICAL FORMATTING RULES:
- Output ONLY the resume or cover letter content itself. No preamble, no intro sentences, no explanations.
- Do NOT start with phrases like "Here is your resume", "I have tailored", "To highlight", or any similar commentary.
- Do NOT add any closing remarks or notes after the content.
- Begin your response directly with the person's name or the first line of the document.

When the user asks for follow-up edits, apply the requested changes to the most recent version of the resume.
Always return the full updated resume, formatted in plain text with clear sections."""

#gemini model
MODEL = "gemini-3.1-flash-lite-preview"

def get_model_config(extra_system: str = ""):
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT + extra_system
    )

#generating resume from gemini
def generate_resume_stream(
    resume_text: str,
    job_description: str,
    extra_prompt: str = "",
) -> Iterator[str]:
    """Stream a tailored resume from Gemini."""
    user_message = f"""Please tailor my resume for the following job description.

--- MY RESUME ---
{resume_text}

--- JOB DESCRIPTION ---
{job_description}
"""
    if extra_prompt.strip():
        user_message += f"\n--- EXTRA INSTRUCTIONS ---\n{extra_prompt.strip()}\n"

    response = client.models.generate_content_stream(
        model=MODEL,
        contents=user_message,
        config=get_model_config(),
        )
    
    for chunk in response:
        if chunk.text:
            yield chunk.text

# Follow-up chat with Gemini to refine the resume based on user feedback, maintaining conversation history for context
def chat_stream(
    conversation_history: list[dict],
    resume_text: str,
    job_description: str,
) -> Iterator[str]:
    """Stream a follow-up chat response from Gemini, maintaining conversation history."""
    context_note = f"""
For context, here is the user's original resume:
--- ORIGINAL RESUME ---
{resume_text}
--- JOB DESCRIPTION ---
{job_description}
Apply any requested edits to the latest version of the resume in the conversation."""

    last_user_msg = conversation_history[-1]["content"]

    # Build history from all messages except the last user message
    raw_history = []
    for msg in conversation_history[:-1]:
        role = "model" if msg["role"] == "assistant" else "user"
        raw_history.append(
            types.Content(role=role, parts=[types.Part(text=msg["content"])])
        )

    # Gemini history MUST start with 'user' and strictly alternate.
    # The first message is always the AI-generated resume (role=model),
    # so we prepend a dummy user turn to make it valid.
    if raw_history and raw_history[0].role == "model":
        dummy_user = types.Content(
            role="user",
            parts=[types.Part(text="Please generate a tailored resume based on my inputs.")]
        )
        raw_history.insert(0, dummy_user)

    #only for debugging
#    print("last_user_msg:", last_user_msg)
#    print("history length:", len(raw_history))
#    print("history roles:", [m.role for m in raw_history])

    chat = client.chats.create(
        model=MODEL,
        config=get_model_config(extra_system=context_note),
        history=raw_history
    )

    response = chat.send_message_stream(last_user_msg)

    print("response:", response)
    
    for chunk in response:
        if chunk.text:
            yield chunk.text

#generating cover letter from gemini based on the generated resume and job description
def generate_cover_letter_stream(
    resume_text: str,
    job_description: str,
) -> Iterator[str]:
    """Stream a cover letter from Gemini based on resume and job description."""
    user_message = f"""Please write a professional cover letter for the following job based on my resume.

--- MY RESUME ---
{resume_text}

--- JOB DESCRIPTION ---
{job_description}

Write a compelling, personalized cover letter that:
1. Opens with a strong hook that matches my background to the role
2. Highlights 2-3 of my most relevant achievements with specifics
3. Shows genuine enthusiasm for the company and role
4. Closes with a confident call to action
5. Keeps to 3-4 paragraphs, professional but personable tone

IMPORTANT: Output only the cover letter itself. Start directly with 'Dear Hiring Manager,' or equivalent. No preamble or commentary.
"""
    response = client.models.generate_content_stream(
        model=MODEL,
        contents=user_message,
        config=get_model_config(),
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text

# ATS match analysis
def analyse_match_stream(
    resume_text: str,
    job_description: str,
) -> Iterator[str]:
    """Stream a structured JSON match analysis between resume and JD."""
    user_message = f"""Analyse how well this resume matches the job description.
 
--- RESUME ---
{resume_text}
 
--- JOB DESCRIPTION ---
{job_description}
 
Return ONLY a JSON object with exactly this structure, no other text:
{{
  "score": <integer 0-100>,
  "summary": "<one sentence explanation of the score>",
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "gaps": ["<gap 1>", "<gap 2>", "<gap 3>"],
  "keywords": ["<missing keyword 1>", "<missing keyword 2>", "<missing keyword 3>", "<missing keyword 4>", "<missing keyword 5>"]
}}
 
Scoring guide:
- 80-100: Strong match, most requirements met
- 60-79: Good match, some gaps
- 40-59: Moderate match, significant gaps
- 0-39: Weak match, major gaps
 
Be specific and actionable. Keywords should be exact terms from the JD missing in the resume."""
 
    response = client.models.generate_content_stream(
        model=MODEL,
        contents=user_message,
        config=get_model_config(),
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text

#asking the AI to edit the section of the resume based on the instruction given by the user
def edit_section_stream(
    section_title: str,
    section_content: str,
    instruction: str,
    job_description: str,
) -> Iterator[str]:
    """Stream a rewrite of a single resume section."""
    user_message = f"""Rewrite only the '{section_title}' section of a resume.

Current content:
{section_content}

Instruction: {instruction}

Job description context:
{job_description}

Return ONLY the rewritten section content — no heading, no preamble, no explanation.
Keep the same format (bullet points if original used bullets, etc.)."""

    response = client.models.generate_content_stream(
        model=MODEL,
        contents=user_message,
        config=get_model_config(),
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text