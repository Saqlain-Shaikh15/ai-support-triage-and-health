import json

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = "./chroma_db"

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Build ticket vector store
def build_ticket_vector_store() -> Chroma:
    """Build or load the support-ticket Chroma vector store.

    Each ticket from the JSON file is converted into a single
    LangChain Document. The ticket's subject, body, and other
    relevant fields are included in the document content for
    semantic search, while identifying and classification fields
    are stored as metadata.

    If the Chroma collection already contains documents, the
    existing vector store is returned.

    Returns:
        The initialized Chroma vector store.
    """

    # Load support tickets from the JSON file.
    with open("data/tickets.json", "r", encoding="utf-8") as file:
        tickets = json.load(file)

    documents: list[Document] = []
    ids: list[str] = []

    # Convert each ticket into a LangChain Document.
    for ticket in tickets:

        content = f"""
        Subject: {ticket["subject"]}
        Body: {ticket["body"]}
        Product: {ticket["product"]}
        Product Area: {ticket["product_area"]}
        Category: {ticket["category"]}
        Urgency: {ticket["urgency"]}
        """.strip()

        document = Document(
            page_content=content,
            metadata={
                "ticket_id": ticket["ticket_id"],
                "account_id": ticket["account_id"],
                "company": ticket["company"],
                "product": ticket["product"],
                "product_area": ticket["product_area"],
                "category": ticket["category"],
                "urgency": ticket["urgency"],
            },
        )

        documents.append(document)
        ids.append(ticket["ticket_id"])

    # Connect to the persistent Chroma collection.
    vector_store = Chroma(
        collection_name="support_tickets",
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )

    # Reuse the existing collection if tickets have already been stored.
    existing = vector_store.get()

    if existing["ids"]:
        print("Loading existing support ticket vector DB...")
        return vector_store

    # Embed and store all tickets in Chroma.
    vector_store.add_documents(
        documents=documents,
        ids=ids,
    )

    print(f"Stored {len(documents)} support tickets.")

    return vector_store

# Create vector store
vector_store = build_ticket_vector_store()

# Create retriever
triage_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
    },
)

triage_system_prompt = """
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
