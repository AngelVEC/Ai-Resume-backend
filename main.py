import os
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from parser import extract_resume_text
from gemini_client import generate_resume_stream, chat_stream, generate_cover_letter_stream, analyse_match_stream, edit_section_stream

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Resume AI API", version="1.0.0")

#Cors Configuration

#grab the cors from the .env file, split by comma, and strip whitespace
_raw_origins = os.environ.get("ALLOWED_ORIGINS")
allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error classification
def classify_error(e: Exception) -> dict:
    msg = str(e).lower()

    if any(k in msg for k in ["429", "rate limit", "quota", "resource exhausted", "too many requests"]):
        return {
            "error": "rate_limit",
            "title": "Rate limit reached",
            "message": "The AI API is temporarily rate-limited. Please wait a moment and try again.",
        }
    if any(k in msg for k in ["401", "403", "api key", "invalid api", "authentication", "permission"]):
        return {
            "error": "auth_error",
            "title": "API key error",
            "message": "Your Gemini API key is invalid or missing. Check your .env file and restart the server.",
        }
    if any(k in msg for k in ["503", "502", "500", "service unavailable", "overloaded", "internal server"]):
        return {
            "error": "server_error",
            "title": "AI service unavailable",
            "message": "The AI service is temporarily unavailable. Please try again in a few seconds.",
        }
    if any(k in msg for k in ["timeout", "timed out", "deadline"]):
        return {
            "error": "timeout",
            "title": "Request timed out",
            "message": "The AI took too long to respond. Please try again.",
        }
    if any(k in msg for k in ["safety", "blocked", "harm", "policy"]):
        return {
            "error": "content_blocked",
            "title": "Content blocked",
            "message": "The AI blocked this request due to content policy. Try rephrasing your inputs.",
        }

    # Generic fallback
    return {
        "error": "unknown_error",
        "title": "Something went wrong",
        "message": f"An unexpected error occurred: {str(e)}",
    }


#Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

#---------------------------------------------------------------------------- Left Panel ---------------------------------------------------------------------------

# Post /generate  — generate resume text from uploaded file and job description
@app.post("/generate")
async def generate(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    extra_prompt: str = Form(""),
):
    # Only accepting PDF and Docx file
    filename = resume.filename or ""
    if not (filename.lower().endswith(".pdf") or filename.lower().endswith(".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    # Limit file size to 5 MB
    file_bytes = await resume.read()
    if len(file_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must be under 5 MB.")

    try:
        resume_text = extract_resume_text(file_bytes, filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse resume: {str(e)}")

    if not resume_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from the uploaded file.")

    def event_stream():
        try:
            for chunk in generate_resume_stream(resume_text, job_description, extra_prompt):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True, 'resume_text': resume_text})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps(classify_error(e))}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Post /analyse  — ATS score analysis of resume vs job description (On left panel)
@app.post("/analyse")
async def analyse(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):
    filename = resume.filename or ""
    if not (filename.lower().endswith(".pdf") or filename.lower().endswith(".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    file_bytes = await resume.read()
    try:
        resume_text = extract_resume_text(file_bytes, filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse resume: {str(e)}")

    if not resume_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from the uploaded file.")

    def event_stream():
        try:
            for chunk in analyse_match_stream(resume_text, job_description):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps(classify_error(e))}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


#---------------------------------------------------------------------------- Right Panel ---------------------------------------------------------------------------

# Post /chat Refining the resume through a conversational interface
class ChatRequest(BaseModel):
    messages: list[dict]
    resume_text: str
    job_description: str


@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty.")

    def event_stream():
        try:
            for chunk in chat_stream(request.messages, request.resume_text, request.job_description):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps(classify_error(e))}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Post /cover-letter  — generate a cover letter based on generated resume and job description
class CoverLetterRequest(BaseModel):
    resume_text: str
    job_description: str


@app.post("/cover-letter")
async def cover_letter(request: CoverLetterRequest):
    if not request.resume_text or not request.job_description:
        raise HTTPException(status_code=400, detail="resume_text and job_description are required.")

    def event_stream():
        try:
            for chunk in generate_cover_letter_stream(request.resume_text, request.job_description):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps(classify_error(e))}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# Post /anaalyse-text - Analyse ATS score based on the generated resume and job desription (on the right panel)
class AnalyseTextRequest(BaseModel):
    resume_text: str
    job_description: str
 
 
@app.post("/analyse-text")
async def analyse_text(request: AnalyseTextRequest):
    if not request.resume_text or not request.job_description:
        raise HTTPException(status_code=400, detail="resume_text and job_description are required.")
 
    def event_stream():
        try:
            for chunk in analyse_match_stream(request.resume_text, request.job_description):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps(classify_error(e))}\n\n"
 
    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Post /edit-section - Edit a specific section of the resume based on instruction 
class EditSectionRequest(BaseModel):
    section_title: str
    section_content: str
    instruction: str
    job_description: str
 
 
@app.post("/edit-section")
async def edit_section(request: EditSectionRequest):
    def event_stream():
        try:
            for chunk in edit_section_stream(
                request.section_title,
                request.section_content,
                request.instruction,
                request.job_description,
            ):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps(classify_error(e))}\n\n"
 
    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Entry point for local dev
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)