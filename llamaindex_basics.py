"""
connect custom data sources to llms
3 steps: data ingestion, data indexing, query inference
in langchain we create agents, here we work on data and integrate it with LLMs. we can integrate it with langchain
really good for RAGs with private datasets
"""
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import os

docs = SimpleDirectoryReader(os.getcwd()).load_data()
print('data loaded')

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print('embedding model created')

llm = OllamaLLM(model="llama3.1")
print("llm defined")

index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
print('Vector Index created')

query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("what do we know about charmander")
print('\n',response)