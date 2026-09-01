"""
flask_main.py — Flask web server for the Support AI dashboard.

Exposes two POST endpoints:
  /api/triage  — wraps task1 logic (KB + historical ticket RAG)
  /api/brief   — wraps task2 logic (TAM account health brief)

Run with:
    python flask_main.py
"""

from __future__ import annotations
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# Bootstrap: make the project root importable so KB_RAG / Triage_RAG resolve.
# Adjust this path if flask_main.py is moved relative to the project root.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# The vector stores take a few seconds to warm up; we initialise them once
# on first request rather than at import time so the server starts instantly.
# ---------------------------------------------------------------------------
_kb_retriever: Any = None
_triage_retriever: Any = None


def _get_retrievers() -> tuple[Any, Any]:
    """Return (kb_retriever, triage_retriever), building them once."""
    global _kb_retriever, _triage_retriever
    if _kb_retriever is None:
        from KB_RAG import kb_retriever  # type: ignore[import]
        from Triage_RAG import triage_retriever  # type: ignore[import]
        _kb_retriever = kb_retriever
        _triage_retriever = triage_retriever
    return _kb_retriever, _triage_retriever

# LLM factory
def _make_llm() -> ChatGroq:
    """Instantiate a zero-temperature Groq LLM from the environment."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set, please add it to your .env file.")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0,
    )


# TASK 1 — Ticket Triage
TRIAGE_SYSTEM_PROMPT = """
SYSTEM:
You are a technical support triage agent.

Analyse the incoming ticket.

Use the retrieved historical tickets as supporting examples,
not as ground truth.

Use the knowledge base as authoritative product documentation.

First determine which retrieved items are relevant.
Ignore irrelevant retrieved context.

Then independently determine:
- product area
- issue category
- urgency P1–P4
- reasoning
- known issue
- relevant KB document
- responder team
- first response draft

Do not invent information that is not supported by
the ticket or relevant knowledge-base content.

[INCOMING TICKET]
{incoming_ticket}

[RETRIEVED HISTORICAL TICKETS]
{triage_context}

[RETRIEVED KNOWLEDGE BASE]
{kb_context}

IMPORTANT RULES:

- Historical tickets are examples only. They do NOT establish that an issue
  is officially known.
- Mark "Known Issue" as "Yes" ONLY if the knowledge base explicitly
  identifies the issue as a known issue.
- Do not create URLs, document names, team names, policies, or procedures.
- Only include a URL if that exact URL exists in the retrieved knowledge base.
- If information is unavailable, write "Not specified in the provided context."
- Do not infer a responder team unless the KB explicitly specifies one.
"""


def run_triage(ticket_text: str) -> str:
    """
    Run ticket triage using KB + historical ticket retrieval.

    Args:
        ticket_text: Raw text of the incoming support ticket.

    Returns:
        The LLM's structured triage analysis as a string.
    """
    kb_retriever, triage_retriever = _get_retrievers()

    triage_context = triage_retriever.invoke(ticket_text)
    kb_context = kb_retriever.invoke(ticket_text)

    prompt = ChatPromptTemplate.from_template(TRIAGE_SYSTEM_PROMPT)
    chain = prompt | _make_llm()

    result = chain.invoke({
        "incoming_ticket": ticket_text,
        "triage_context": triage_context,
        "kb_context": kb_context,
    })
    return result.content


# ===========================================================================
# TASK 2 — TAM Account Health Brief
# ===========================================================================

# Soft-coded reference date — matches the value used in task2.py.
_REFERENCE_DATE = datetime.strptime("14082026", "%d%m%Y").replace(
    hour=23, minute=59, second=59
)
_CUTOFF_DATE = _REFERENCE_DATE - timedelta(days=90)

# Keyword lists copied verbatim from task2.py so behaviour is identical.
_CHURN_SIGNALS = [
    "competitor", "competing vendor", "alternative vendor",
    "alternative provider", "switching vendor", "switch vendors",
    "switch provider", "replace the platform", "replacing the platform",
    "cancel", "cancellation", "terminate", "termination", "leave the platform",
]

_ESCALATION_SIGNALS = [
    "escalate", "escalation", "executive escalation", "executive complaint",
    "unacceptable", "extremely frustrated", "very frustrated", "frustrated with",
    "complaint", "urgent", "response time", "slow response", "delayed response",
]

BRIEF_PROMPT = """
You are a TAM account-health summarisation assistant.

Your job is to turn the supplied FACT PACKET into a concise,
actionable account brief.

IMPORTANT RULES:

1. Use ONLY information contained in the FACT PACKET.
2. Do not invent facts, numbers, customer statements, teams,
   dates, products, or recommendations presented as facts.
3. The Python application has already identified risk signals.
   Do not create additional churn or escalation flags.
4. Every flagged ticket MUST include the exact quote supplied
   in the FACT PACKET.
5. Do not modify, paraphrase, or fabricate ticket quotes.
6. Keep the executive summary to exactly 3-5 sentences.
7. Produce exactly these three sections:
   1. Executive Summary
   2. Open Risks & Flagged Issues
   3. Recommended Talking Points for the TAM
8. Be concise and actionable.
9. Distinguish account-level risks from ticket-level signals.
10. If information is not available, do not guess.

FACT PACKET:

{fact_packet}
"""


def _parse_ticket_date(date_string: str) -> datetime:
    """Parse an ISO-8601 ticket timestamp, stripping the timezone."""
    return datetime.fromisoformat(
        date_string.replace("Z", "+00:00")
    ).replace(tzinfo=None)


def _find_signals(text: str, signals: list[str]) -> list[str]:
    """Return every signal phrase that appears in *text* (case-insensitive)."""
    text_lower = text.lower()
    return [s for s in signals if s in text_lower]


def _sentence_containing(body: str, signal: str) -> str:
    """
    Extract the sentence from *body* that contains *signal*.

    Falls back to the entire body if no sentence boundary is found.
    """
    for sentence in re.split(r"(?<=[.!?])\s+", body.strip()):
        if signal.lower() in sentence.lower():
            return sentence.strip()
    return body.strip()


def _detect_ticket_risks(tickets: list[dict]) -> list[dict]:
    """
    Scan each ticket body for churn and escalation keyword signals.

    Args:
        tickets: List of ticket dicts (already filtered to the 90-day window).

    Returns:
        List of dicts — one per flagged ticket — with signals and quotes.
    """
    flagged: list[dict] = []

    for ticket in tickets:
        body = ticket.get("body", "")
        churn_hits = _find_signals(body, _CHURN_SIGNALS)
        escl_hits = _find_signals(body, _ESCALATION_SIGNALS)

        if not churn_hits and not escl_hits:
            continue

        signals: list[dict] = []

        for s in churn_hits:
            signals.append({
                "type": "Churn Risk",
                "trigger": s,
                "quote": _sentence_containing(body, s),
            })
        for s in escl_hits:
            signals.append({
                "type": "Escalation Signal",
                "trigger": s,
                "quote": _sentence_containing(body, s),
            })

        flagged.append({
            "ticket_id": ticket["ticket_id"],
            "subject": ticket.get("subject", ""),
            "created_at": ticket["created_at"],
            "signals": signals,
        })

    return flagged


def _detect_account_risks(account: dict) -> list[str]:
    """
    Derive structured risk strings from account-level fields.

    These are deterministic checks — no LLM involved.
    """
    risks: list[str] = []

    if account.get("health_status", "").lower() == "at risk":
        risks.append("Account health status is explicitly marked 'At Risk'.")

    open_tickets = account.get("open_tickets", 0)
    if open_tickets:
        risks.append(f"{open_tickets} open tickets are currently recorded.")

    p1_count = account.get("p1_tickets_last_30d", 0)
    if p1_count:
        risks.append(f"{p1_count} P1 ticket(s) were recorded in the last 30 days.")

    last_login = account.get("last_login_days_ago")
    if last_login is not None and last_login > 30:
        risks.append(f"Last login was {last_login} days ago.")

    renewal_date_str = account.get("renewal_date")
    if renewal_date_str:
        renewal_dt = datetime.strptime(renewal_date_str, "%Y-%m-%d")
        if renewal_dt < _REFERENCE_DATE:
            risks.append(
                f"Renewal date {renewal_date_str} has already passed "
                f"relative to the reference date."
            )

    for note in account.get("escalation_notes", []):
        risks.append(f"Escalation note: {note}")

    return risks


def run_brief(account_id: str) -> tuple[dict | None, str]:
    """
    Generate a TAM account health brief for the given account.

    Args:
        account_id: The account identifier, e.g. "ACC-1899".

    Returns:
        A tuple of (account_dict, brief_text).
        account_dict is None when the account is not found.
    """
    data_dir = PROJECT_ROOT / "data"
    accounts: list[dict] = json.loads((data_dir / "accounts.json").read_text())
    tickets: list[dict] = json.loads((data_dir / "tickets.json").read_text())

    # Locate the account — return early when missing.
    account = next((a for a in accounts if a["account_id"] == account_id), None)
    if account is None:
        return None, f"Account '{account_id}' not found in accounts.json."

    # Filter to tickets within the 90-day window for this account.
    recent_tickets = sorted(
        [
            t for t in tickets
            if t.get("account_id") == account_id
            and _CUTOFF_DATE <= _parse_ticket_date(t["created_at"]) <= _REFERENCE_DATE
        ],
        key=lambda t: (t["created_at"], t["ticket_id"]),
    )

    flagged_tickets = _detect_ticket_risks(recent_tickets)
    account_risks = _detect_account_risks(account)

    fact_packet = {
        "reference_date": "14082026",
        "90_day_cutoff": _CUTOFF_DATE.strftime("%Y-%m-%d"),
        "account": {
            "account_id": account["account_id"],
            "company": account["company"],
            "tam": account.get("tam"),
            "plan_tier": account.get("plan_tier"),
            "arr_usd": account.get("arr_usd"),
            "seats_licensed": account.get("seats_licensed"),
            "seats_active": account.get("seats_active"),
            "products": account.get("products", []),
            "health_status": account.get("health_status"),
            "usage_trend": account.get("usage_trend"),
            "open_tickets": account.get("open_tickets"),
            "p1_tickets_last_30d": account.get("p1_tickets_last_30d"),
            "customer_since": account.get("customer_since"),
            "renewal_date": account.get("renewal_date"),
            "last_qbr_date": account.get("last_qbr_date"),
            "primary_contact": account.get("primary_contact"),
            "nps_score": account.get("nps_score"),
            "last_login_days_ago": account.get("last_login_days_ago"),
            "integrations_active": account.get("integrations_active", []),
            "region": account.get("region"),
            "industry": account.get("industry"),
        },
        "account_risks": account_risks,
        "recent_ticket_count": len(recent_tickets),
        "recent_tickets": [
            {
                "ticket_id": t["ticket_id"],
                "subject": t.get("subject"),
                "product": t.get("product"),
                "category": t.get("category"),
                "urgency": t.get("urgency"),
                "status": t.get("status"),
                "created_at": t.get("created_at"),
            }
            for t in recent_tickets
        ],
        "flagged_tickets": flagged_tickets,
    }

    prompt = ChatPromptTemplate.from_template(BRIEF_PROMPT)
    chain = prompt | _make_llm()

    result = chain.invoke({
        "fact_packet": json.dumps(fact_packet, indent=2, ensure_ascii=False)
    })

    return account, result.content


# ===========================================================================
# Flask routes
# ===========================================================================

@app.route("/")
def index() -> str:
    """Serve the single-page dashboard."""
    return render_template("index.html")


@app.route("/api/triage", methods=["POST"])
def api_triage() -> Response:
    """
    POST /api/triage

    Body (JSON): { "ticket": "<ticket text>" }

    Returns (JSON): { "result": "<triage analysis>" }
                 or { "error": "<message>" } with a 4xx/5xx status.
    """
    body = request.get_json(silent=True) or {}
    ticket_text = (body.get("ticket") or "").strip()

    if not ticket_text:
        return jsonify({"error": "No ticket text provided."}), 400

    try:
        result = run_triage(ticket_text)
        return jsonify({"result": result})
    except RuntimeError as exc:
        # Configuration error (e.g. missing API key) — tell the client clearly.
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Triage error")
        return jsonify({"error": f"Internal error: {exc}"}), 500


@app.route("/api/brief", methods=["POST"])
def api_brief() -> Response:
    """
    POST /api/brief

    Body (JSON): { "account_id": "ACC-XXXX" }

    Returns (JSON): {
        "account": { ...account fields... },
        "result":  "<brief markdown>"
    }
    or { "error": "<message>" } with an appropriate status code.
    """
    body = request.get_json(silent=True) or {}
    account_id = (body.get("account_id") or "").strip().upper()

    if not account_id:
        return jsonify({"error": "No account_id provided."}), 400

    try:
        account, brief = run_brief(account_id)
        if account is None:
            return jsonify({"error": brief}), 404
        return jsonify({"account": account, "result": brief})
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Brief generation error")
        return jsonify({"error": f"Internal error: {exc}"}), 500


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    # debug=True gives live reloading during development.
    # In production use a proper WSGI server (gunicorn, etc.).
    app.run(debug=True, port=5000)