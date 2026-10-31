import os
import fastapi
"""FastAPI server for the Support AI dashboard.

Endpoints
---------
GET  /              -> serves the dashboard (templates/index.html)
GET  /static/...    -> CSS / JS assets
GET  /api/health    -> lightweight status check used by the sidebar footer
POST /api/triage    -> {"ticket": str}       -> {"result": str, "sources": [...]}
POST /api/brief     -> {"account_id": str}   -> {"result": str, "account": {...}, "meta": {...}}

Run from the repo root:
    uvicorn main:app --reload
"""

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

# The RAG modules and task2 use paths relative to the repo root
# ("data/...", "./chroma_db", "./knowledge-base"), so pin the working
# directory before importing them. This lets uvicorn start from anywhere.
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

load_dotenv()

# Importing these builds / loads the Chroma vector stores once at startup.
# task1.py is intentionally NOT imported: it runs a retrieval at import time.
import task2
from KB_RAG import kb_retriever
from prompts import TRIAGE_PROMPT
from Triage_RAG import triage_retriever

logger = logging.getLogger("support_ai")

app = FastAPI(title="Support AI", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


# ---------------------------------------------------------------------------
# Error handling: the frontend reads `data.error`, so every failure is
# returned as {"error": "..."} rather than FastAPI's default {"detail": ...}.
# ---------------------------------------------------------------------------
@app.exception_handler(StarletteHTTPException)
async def http_error_handler(_: Request, exc: StarletteHTTPException):
    return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    message = first.get("msg", "Invalid request")
    return JSONResponse(
        {"error": f"Invalid {field or 'request'}: {message}"}, status_code=422
    )


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class TriageRequest(BaseModel):
    ticket: str = Field(..., min_length=1, max_length=20_000)


class BriefRequest(BaseModel):
    account_id: str = Field(..., min_length=1, max_length=64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_model():
    """Create the Groq chat model once (raises RuntimeError if no API key)."""
    return task2.create_model()


def normalise_ticket(raw: str) -> str:
    """Accept plain text or JSON like {"subject": "...", "body": "..."}."""
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw

    if isinstance(parsed, dict):
        subject = str(parsed.get("subject", "")).strip()
        body = str(parsed.get("body", "")).strip()
        parts = [f"Subject: {subject}" if subject else "", body]
        text = "\n".join(p for p in parts if p)
        return text or raw

    return raw


def format_ticket_context(docs) -> str:
    """Render retrieved historical tickets as prompt context."""
    if not docs:
        return "No similar historical tickets were found."

    blocks = []
    for doc in docs:
        m = doc.metadata
        header = (
            f"[Ticket {m.get('ticket_id', '?')} | {m.get('product', '?')} / "
            f"{m.get('product_area', '?')} | {m.get('category', '?')} | "
            f"{m.get('urgency', '?')}]"
        )
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def format_kb_context(docs) -> str:
    """Render retrieved knowledge-base chunks as prompt context."""
    if not docs:
        return "No relevant knowledge-base documentation was found."

    blocks = []
    for doc in docs:
        m = doc.metadata
        header = f"[KB: {m.get('category', '?')}/{m.get('source', '?')}]"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def collect_sources(ticket_docs, kb_docs) -> list[dict]:
    """Build a de-duplicated source list so the UI can show what the answer used."""
    sources, seen = [], set()

    for doc in ticket_docs:
        m = doc.metadata
        key = ("ticket", m.get("ticket_id"))
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "type": "ticket",
            "id": m.get("ticket_id", ""),
            "detail": f"{m.get('product', '')} / {m.get('category', '')}".strip(" /"),
        })

    for doc in kb_docs:
        m = doc.metadata
        section = (m.get("h3") or m.get("h2") or m.get("h1") or "").lstrip("# ").strip()
        key = ("kb", m.get("source"), section)
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "type": "kb",
            "id": m.get("source", ""),
            "detail": section,
        })

    return sources


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "llm_configured": bool(os.getenv("GROQ_API_KEY"))}


# Plain `def` (not async): both pipelines make blocking calls (embeddings,
# Chroma, Groq), so FastAPI runs them in its threadpool and the event loop
# stays free for other requests.
@app.post("/api/triage")
def triage(req: TriageRequest):
    ticket_text = normalise_ticket(req.ticket)
    if not ticket_text:
        raise HTTPException(status_code=400, detail="The ticket is empty.")

    try:
        ticket_docs = triage_retriever.invoke(ticket_text)
        kb_docs = kb_retriever.invoke(ticket_text)

        chain = TRIAGE_PROMPT | get_model()
        result = chain.invoke({
            "incoming_ticket": ticket_text,
            "triage_context": format_ticket_context(ticket_docs),
            "kb_context": format_kb_context(kb_docs),
        })
    except RuntimeError as exc:  # missing GROQ_API_KEY
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Triage failed")
        raise HTTPException(
            status_code=502, detail=f"The triage request failed: {exc}"
        )

    return {
        "result": result.content,
        "sources": collect_sources(ticket_docs, kb_docs),
    }


@app.post("/api/brief")
def brief(req: BriefRequest):
    account_id = req.account_id.strip().upper()

    try:
        accounts = task2.load_json(task2.ACCOUNTS_FILE)
        tickets = task2.load_json(task2.TICKETS_FILE)
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception("Could not load data files")
        raise HTTPException(status_code=500, detail=f"Could not load data files: {exc}")

    try:
        account = task2.get_account(accounts, account_id)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Account '{account_id}' was not found. Check the ID and try again.",
        )

    try:
        recent = task2.get_recent_tickets(tickets, account_id)
        flagged = task2.detect_ticket_risks(recent)
        packet = task2.build_fact_packet(account, recent, flagged)
        text = task2.generate_brief(packet)
    except RuntimeError as exc:  # missing GROQ_API_KEY
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Brief generation failed")
        raise HTTPException(
            status_code=502, detail=f"The brief could not be generated: {exc}"
        )

    return {
        "result": text,
        "account": account,
        "meta": {
            "tickets_reviewed": len(recent),
            "tickets_flagged": len(flagged),
            "window_days": 90,
        },
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)