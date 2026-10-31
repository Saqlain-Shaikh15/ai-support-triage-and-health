# from Triage_RAG import triage_retriever
# from KB_RAG import kb_retriever

# result = triage_retriever.invoke("Databridge Pro bulk archieve functionality")
# print(result)

# # result = kb_retriever.invoke("Databridge Pro bulk archieve functionality")
# # print(result)

from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

print("API key loaded:", bool(api_key))

models = client.models.list()

for model in models.data:
    print(model.id)