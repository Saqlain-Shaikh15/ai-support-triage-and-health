from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import TypedDict
import os

KB_DIR = "./knowledge-base"
DB_DIR = "./chroma_db"

# TYPES
class MarkdownSection(TypedDict): 
    """Represents a semantically meaningful section of a Markdown document.""" 
    h1: str 
    h2: str 
    h3: str 
    title: str 
    content: str

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Markdown semantic chunking
def split_markdown_into_sections(markdown_text: str) -> list[MarkdownSection]:
    """
    Create semantic chunks from Markdown.

    Rules:
    - # is document context
    - ## is a major section
    - ### is a semantic subsection
    - Content directly under ## becomes its own chunk
    - ### content becomes its own chunk

    Args: 
        markdown_text: The raw Markdown document to split into sections. 
    Returns: 
        A list of Markdown sections containing their heading hierarchy and associated content.
    """

    lines = markdown_text.splitlines()

    current_h1 = ""
    current_h2 = ""
    current_h3 = ""
    current_content = []

    sections = []
    def save_section() -> None:
        """Save the current content as a semantic section if it is non-empty."""
        nonlocal current_content # allows us to modify the outer current_content

        content = "\n".join(current_content).strip()

        if not content:
            return

        # If we have an H3, that's our semantic section.
        # Otherwise, the H2 itself is the section.
        title = current_h3 or current_h2 or current_h1

        sections.append({
            "h1": current_h1,
            "h2": current_h2,
            "h3": current_h3,
            "title": title,
            "content": content
        })

        current_content = []

    for line in lines:

        # H1
        if line.startswith("# ") and not line.startswith("## "):

            save_section()

            current_h1 = line.strip()
            current_h2 = ""
            current_h3 = ""

            continue

        # H2
        if line.startswith("## "):

            save_section()

            current_h2 = line.strip()
            current_h3 = ""

            continue

        # H3
        if line.startswith("### "):

            save_section()

            current_h3 = line.strip()

            continue

        # Normal content
        current_content.append(line)

    # Save final section
    save_section()

    return sections


# Build Chroma vector store
def build_kb_vector_store() -> Chroma:
    """Build or load the knowledge-base Chroma vector store. 
    
    If a Chroma database already contains documents, the existing vector store is returned. Otherwise, all Markdown files under "KB_DIR" are processed, chunked, embedded, and stored.

    Returns: 
        The initialized Chroma vector store. 
    """

    vector_store = Chroma(
        collection_name="knowledge_base",
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )

    existing = vector_store.get()
    if existing["ids"]:
        print("Loading existing knowledge base vector DB...")
        return vector_store

    print("Building knowledge base vector DB...")

    documents = []
    ids = []

    # Walk through KB directories
    for root, dirs, files in os.walk(KB_DIR):

        for filename in files:

            if not filename.endswith(".md"):
                continue

            filepath = os.path.join(root, filename)

            # Category = parent directory
            category = os.path.basename(root)

            # Read Markdown
            with open(filepath, "r", encoding="utf-8") as f:
                markdown_text = f.read()

            # Create Semantic chunks
            sections = split_markdown_into_sections(markdown_text)

            for index, section in enumerate(sections):

                # Include heading hierarchy in the actual embedded text
                page_content = f"""
                {section["h1"]}

                {section["h2"]}

                {section["h3"]}

                {section["content"]}
                """.strip()

                document = Document(
                    page_content=page_content,
                    metadata={
                        "category": category,
                        "source": filename,
                        "h1": section["h1"],
                        "h2": section["h2"],
                        "h3": section["h3"],
                        "chunk_index": index
                    }
                )

                # Create a stable document ID
                chunk_id = (
                    f"{category}_"
                    f"{os.path.splitext(filename)[0]}_"
                    f"{index}"
                )

                documents.append(document)
                ids.append(chunk_id)

    # Add documents to Chroma
    vector_store.add_documents(
        documents=documents,
        ids=ids
    )

    print(f"Stored {len(documents)} KB chunks.")

    return vector_store

# Create vector store
kb_vector_store = build_kb_vector_store()

# Create retriever
kb_retriever = kb_vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)