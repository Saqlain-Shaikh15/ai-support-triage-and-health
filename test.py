from Triage_RAG import triage_retriever
from KB_RAG import kb_retriever

result = triage_retriever.invoke("Databridge Pro bulk archieve functionality")
print(result)

# result = kb_retriever.invoke("Databridge Pro bulk archieve functionality")
# print(result)