from langchain_groq import ChatGroq
import os
from Triage_RAG import triage_retriever
from KB_RAG import kb_retriever
from dotenv import load_dotenv
from prompts import TRIAGE_PROMPT

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b", 
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

chain = TRIAGE_PROMPT | model

incoming_ticket = "Hey something is wrong with the billing system, billing went from 80 users to 100 users when we had 80 active users"

triage_context = triage_retriever.invoke(incoming_ticket)
# print(triage_context)
# kb_context = kb_retriever.invoke(incoming_ticket)
# result = chain.invoke({"incoming_ticket": incoming_ticket, "triage_context": triage_context, "kb_context": kb_context})
# print(result.content)