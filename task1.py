from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from KB_RAG import kb_retriever
from Triage_RAG import triage_retriever
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile", 
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

template = """
SYSTEM:
You are a technical support triage agent.

Analyze the incoming ticket.

Use the retrieved historical tickets as supporting examples,
not as ground truth.

Use the knowledge base as authoritative product documentation.

First determine which retrieved items are relevant.
Ignore irrelevant retrieved context.

Then independently determine:
- product area
- issue category
- urgency P1-P4
- reasoning
- known issue
- relevant KB document
- responder team
- first response

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

prompt = ChatPromptTemplate.from_template(template=template)
chain = prompt | model

incoming_ticket = "Hey something is wrong with the billing system, billing went from 80 users to 100 users when we had 80 active users"

triage_context = triage_retriever.invoke(incoming_ticket)
print(triage_context)
# kb_context = kb_retriever.invoke(incoming_ticket)
# result = chain.invoke({"incoming_ticket": incoming_ticket, "triage_context": triage_context, "kb_context": kb_context})
# print(result.content)