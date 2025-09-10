# optimize output of LLM so it refers to a large knowledge base outside of its training data
# finetuning is expensive so we use RAGs (especially if it changes a lot)
# we create a vector database and query it like a tool
# vector DB: take data, preprocess, chunk into smaller data, embedding, store in a DB
# can apply similarity based search etc in vector DB
# query is embedded and queried in vector DB, that response is context for an LLM (with prompt) and that is generated as output 
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma, FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(text)
# print(documents[0])


## Vector embeddings and store
embeddings = OllamaEmbeddings(model="llama2")
query = 'who owns pokemon?'

### Chroma database
db = Chroma.from_documents(documents=documents, embedding=embeddings)
result = db.similarity_search(query)
# print(result[0].page_content)

### faiss database
db = FAISS.from_documents(documents=documents, embedding=embeddings)
result = db.similarity_search(query)
# print(result[0].page_content)


## integrate LLM
"""
we take a prompt
We use LLMs for "chain and retrieve"
Chain: creating a document chain which can be passed to llm as prompts
Retrieve: it returns documents in an unstructured query, more general than a vector store
"""
llm = OllamaLLM(model='llama2')
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context. Keep the answers short and simple.
    <context> {context} </context>
    Question: {input}
    """
)
"""
We will use stuff document chain which takes a bunch of documents, formats them into prompts and passes it to LLM
Another helpful chain is sql query chain which can get get information from a sql server instead of documents
"""
### Chain
document_chain = create_stuff_documents_chain(llm, prompt)

### Retrieve
retriever = db.as_retriever()

### Create retriever chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)
result = retrieval_chain.invoke({
    'input': query
})
print(result['answer'])