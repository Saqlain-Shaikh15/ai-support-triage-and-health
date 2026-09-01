import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from KB_RAG import kb_retriever


# ============================================================
# CONFIGURATION
# ============================================================

# Soft-coded reference date.
# Format: DDMMYYYY
DATE = "14082026"

ACCOUNTS_FILE = Path("data/accounts.json")
TICKETS_FILE = Path("data/tickets.json")

# ============================================================
# DATE HELPERS
# ============================================================

def parse_reference_date(date_string):
    return datetime.strptime(date_string, "%d%m%Y").replace(
        hour=23,
        minute=59,
        second=59,
        microsecond=999999,
    )


def parse_ticket_date(date_string):
    return datetime.fromisoformat(
        date_string.replace("Z", "+00:00")
    ).replace(tzinfo=None)


REFERENCE_DATE = parse_reference_date(DATE)
CUTOFF_DATE = REFERENCE_DATE - timedelta(days=90)


# ============================================================
# DATA LOADING
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# ACCOUNT LOOKUP
# ============================================================

def get_account(accounts, account_id):
    for account in accounts:
        if account.get("account_id") == account_id:
            return account

    raise ValueError(
        f"Account '{account_id}' was not found in accounts.json."
    )


# ============================================================
# TICKET FILTERING
# ============================================================

def get_recent_tickets(tickets, account_id):
    relevant_tickets = []

    for ticket in tickets:
        if ticket.get("account_id") != account_id:
            continue

        created_at = parse_ticket_date(ticket["created_at"])

        if CUTOFF_DATE <= created_at <= REFERENCE_DATE:
            relevant_tickets.append(ticket)

    # Explicit sorting makes processing deterministic.
    relevant_tickets.sort(
        key=lambda ticket: (
            ticket["created_at"],
            ticket["ticket_id"]
        )
    )

    return relevant_tickets


# ============================================================
# RISK DETECTION
# ============================================================

CHURN_SIGNALS = [
    "competitor",
    "competing vendor",
    "alternative vendor",
    "alternative provider",
    "switching vendor",
    "switch vendors",
    "switch provider",
    "replace the platform",
    "replacing the platform",
    "cancel",
    "cancellation",
    "terminate",
    "termination",
    "leave the platform",
]

ESCALATION_SIGNALS = [
    "escalate",
    "escalation",
    "executive escalation",
    "executive complaint",
    "unacceptable",
    "extremely frustrated",
    "very frustrated",
    "frustrated with",
    "complaint",
    "urgent",
    "response time",
    "slow response",
    "delayed response",
]


def find_matching_signals(text, signals):
    text_lower = text.lower()

    return [
        signal
        for signal in signals
        if signal in text_lower
    ]


def extract_sentence_with_signal(body, signal):
    """
    Return the original sentence containing the matching signal.

    The returned quote comes directly from the ticket body.
    """

    # Split while retaining the original text.
    sentences = re.split(r"(?<=[.!?])\s+", body.strip())

    for sentence in sentences:
        if signal.lower() in sentence.lower():
            return sentence.strip()

    # Fallback if the signal isn't inside a normal sentence.
    return body.strip()


def detect_ticket_risks(tickets):
    flagged_tickets = []

    for ticket in tickets:
        body = ticket.get("body", "")

        churn_matches = find_matching_signals(
            body,
            CHURN_SIGNALS
        )

        escalation_matches = find_matching_signals(
            body,
            ESCALATION_SIGNALS
        )

        if not churn_matches and not escalation_matches:
            continue

        signals = []

        for signal in churn_matches:
            signals.append({
                "type": "Churn Risk",
                "trigger": signal,
                "quote": extract_sentence_with_signal(
                    body,
                    signal
                ),
            })

        for signal in escalation_matches:
            signals.append({
                "type": "Escalation Signal",
                "trigger": signal,
                "quote": extract_sentence_with_signal(
                    body,
                    signal
                ),
            })

        flagged_tickets.append({
            "ticket_id": ticket["ticket_id"],
            "subject": ticket.get("subject", ""),
            "created_at": ticket["created_at"],
            "signals": signals,
        })

    return flagged_tickets


# ============================================================
# ACCOUNT-LEVEL RISK DETECTION
# ============================================================

def detect_account_risks(account):
    risks = []

    if account.get("health_status", "").lower() == "at risk":
        risks.append(
            "Account health status is explicitly marked 'At Risk'."
        )

    if account.get("open_tickets", 0) > 0:
        risks.append(
            f"{account['open_tickets']} open tickets are currently recorded."
        )

    if account.get("p1_tickets_last_30d", 0) > 0:
        risks.append(
            f"{account['p1_tickets_last_30d']} P1 ticket(s) were recorded "
            "in the last 30 days."
        )

    if account.get("last_login_days_ago") is not None:
        if account["last_login_days_ago"] > 30:
            risks.append(
                f"Last login was {account['last_login_days_ago']} days ago."
            )

    renewal_date = account.get("renewal_date")

    if renewal_date:
        renewal = datetime.strptime(
            renewal_date,
            "%Y-%m-%d"
        )

        if renewal < REFERENCE_DATE:
            risks.append(
                f"Renewal date {renewal_date} has already passed "
                f"relative to the reference date {DATE}."
            )

    for note in account.get("escalation_notes", []):
        risks.append(
            f"Escalation note: {note}"
        )

    return risks


# ============================================================
# FACT PACKET
# ============================================================

def build_fact_packet(account, recent_tickets, flagged_tickets):
    account_risks = detect_account_risks(account)

    return {
        "reference_date": DATE,
        "90_day_cutoff": CUTOFF_DATE.strftime("%Y-%m-%d"),

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
            "p1_tickets_last_30d": account.get(
                "p1_tickets_last_30d"
            ),
            "customer_since": account.get("customer_since"),
            "renewal_date": account.get("renewal_date"),
            "last_qbr_date": account.get("last_qbr_date"),
            "primary_contact": account.get("primary_contact"),
            "nps_score": account.get("nps_score"),
            "last_login_days_ago": account.get(
                "last_login_days_ago"
            ),
            "integrations_active": account.get(
                "integrations_active",
                []
            ),
            "region": account.get("region"),
            "industry": account.get("industry"),
        },

        "account_risks": account_risks,

        "recent_ticket_count": len(recent_tickets),

        "recent_tickets": [
            {
                "ticket_id": ticket["ticket_id"],
                "subject": ticket.get("subject"),
                "product": ticket.get("product"),
                "category": ticket.get("category"),
                "urgency": ticket.get("urgency"),
                "status": ticket.get("status"),
                "created_at": ticket.get("created_at"),
            }
            for ticket in recent_tickets
        ],

        "flagged_tickets": flagged_tickets,
    }


# ============================================================
# LLM SYNTHESIS
# ============================================================

def create_model():
    load_dotenv()

    api_key = __import__("os").getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file."
        )

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0,
    )


def generate_brief(fact_packet):
    model = create_model()

    template = """
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

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | model

    result = chain.invoke({
        "fact_packet": json.dumps(
            fact_packet,
            indent=2,
            ensure_ascii=False
        )
    })

    return result.content


# ============================================================
# MAIN
# ============================================================

def run(account_id):
    accounts = load_json(ACCOUNTS_FILE)
    tickets = load_json(TICKETS_FILE)

    account = get_account(
        accounts,
        account_id
    )

    recent_tickets = get_recent_tickets(
        tickets,
        account_id
    )

    flagged_tickets = detect_ticket_risks(
        recent_tickets
    )

    fact_packet = build_fact_packet(
        account,
        recent_tickets,
        flagged_tickets
    )

    brief = generate_brief(
        fact_packet
    )

    print("\n" + "=" * 70)
    print(f"TAM ACCOUNT HEALTH BRIEF — {account_id}")
    print("=" * 70)
    print()
    print(brief)
    print()


if __name__ == "__main__":
    run("ACC-1899")