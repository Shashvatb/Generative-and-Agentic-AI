# optimize output of LLM so it refers to a large knowledge base outside of its training data
# finetuning is expensive so we use RAGs (especially if it changes a lot)
# we create a vector database and query it like a tool
# vector DB: take data, preprocess, chunk into smaller data, embedding, store in a DB
# can apply similarity based search etc in vector DB
# query is embedded and queried in vector DB, that response is context for an LLM (with prompt) and that is generated as output 

from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os
import bs4
from dotenv import load_dotenv
load_dotenv()

## Data ingestion
### from txt
loader = TextLoader("pokemon.txt")
text = loader.load()
# print(text)

### from pdf file
loader = PyPDFLoader('pokemon.pdf')
text = loader.load()
# print(text)

### from web
loader = WebBaseLoader(web_paths=("https://en.wikipedia.org/wiki/Pok%C3%A9mon",), 
                       bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                           class_=('mw-content-container',)
                       )))
text = loader.load()
# print(text)


## Data Transform
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(text)
# print(documents[0])


## Vector embeddings and store
embeddings = OllamaEmbeddings(model="llama2")
db = Chroma.from_documents(documents=documents, embedding=embeddings)

query = 'who owns pokemon?'

result = db.similarity_search(query)
print(result[0].page_content)
