# optimize output of LLM so it refers to a large knowledge base outside of its training data
# finetuning is expensive so we use RAGs (especially if it changes a lot)
# we create a vector database and query it like a tool
# vector DB: take data, preprocess, chunk into smaller data, embedding, store in a DB
# can apply similarity based search etc in vector DB
# query is embedded and queried in vector DB, that response is context for an LLM (with prompt) and that is generated as output 

from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
import os
import bs4
from dotenv import load_dotenv
load_dotenv()

## Data ingestion
### from txt
loader = TextLoader("pokemon.txt")
text = loader.load()
print(text)

### from web
loader = WebBaseLoader(web_paths=("https://en.wikipedia.org/wiki/Pok%C3%A9mon",), 
                       bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                           class_=('mw-content-container',)
                       )))
text = loader.load()
print(text)

### from pdf file
loader = PyPDFLoader('pokemon.pdf')
text = loader.load()
print(text)