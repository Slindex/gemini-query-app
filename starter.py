# System Utilities
import logging
import sys
import os
from dotenv import load_dotenv

# LLM Utilities
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.core import Settings



# For debugging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Loading environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Models instantiations and global configuration
llm = Gemini(api_key=GEMINI_API_KEY, model_name='models/gemini-pro')
embedding_model = GeminiEmbedding(api_key=GEMINI_API_KEY, model_name='models/embedding-001')
Settings.llm = llm
Settings.embed_model = embedding_model

# Checking if storage already exists
PERSIST_DIR = './storage'
if not os.path.exists(PERSIST_DIR):
    # Load documents and create the index
    documents = SimpleDirectoryReader('data').load_data()
    index = VectorStoreIndex.from_documents(documents)
    # Store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # Load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Index query
query_engine = index.as_query_engine()
response = query_engine.query('Haz un breve resumen del texto?')
print(response)