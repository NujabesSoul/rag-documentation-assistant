from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

print("🚀 Starting RAG...")

# Local embeddings (free)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model

# OpenAI for generation (pennies per query)
llm = OpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
Settings.llm = llm

# Load only test docs
documents = SimpleDirectoryReader("data").load_data()
test_docs = [d for d in documents if 'test-doc' in d.metadata.get('file_name', '')]

print(f"✅ Loaded {len(test_docs)} test documents")

print("\n🔍 Creating index...")
index = VectorStoreIndex.from_documents(test_docs)
print("✅ Index created")

query_engine = index.as_query_engine()

print("\n❓ Testing query...")
response = query_engine.query("What are these documents about?")
print(f"\n📄 Response:\n{response}")

print("\n✅ Phase 1 complete!")
