import os
import uvicorn
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


def chunk_documents(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    documents = text_splitter.split_documents(text)
    return documents


def index_documents(vector_db, chunks):
    vector_db.add_documents(chunks)
    return vector_db


def search_document(vector_db, query):
    return vector_db.similarity_search(query)


def generate_answer(user_query, context_documents, prompt_template, llm):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversational_prompt = ChatPromptTemplate.from_template(prompt_template)
    response_chain = conversational_prompt | llm
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

def init():
    prompt_template = """ You are an expert pokemon analyst. Use the provided context to answer the query. If unsure, ask questions. 
    be concise and give short form answers. do not be goofy. get straight to the point. do not use emojis.
    if it is related to pokemon, use the context provided from the vector DB.

    Query: {user_query},
    Context: {document_context},
    answer:

    """
    pdf_store_path = 'pokemon.pdf'
    embedding_model = OllamaEmbeddings(model="llama2")
    llm = OllamaLLM(model='llama2')
    document_vector_db = InMemoryVectorStore(embedding_model)

    documents = load_pdf(pdf_store_path)
    chunked_documents = chunk_documents(documents)
    document_vector_db = index_documents(document_vector_db, chunked_documents)

    query = 'what pokemons are the starters in generation 1?'
    document = search_document(document_vector_db, query)
    print(generate_answer(query, document, prompt_template, llm))

    return {
        "vector_db": document_vector_db,
        "llm": llm,
        "prompt": prompt_template,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading RAG chain...")
    app.state.rag = init()
    yield


app = FastAPI(
    title="RAG API",
    version="1.0",
    lifespan=lifespan
    )

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str


@app.get("/")
def test():
    return {'test': 'success'}


@app.post("/query")
def query_rag(request: QueryRequest):
    components = app.state.rag
    document = search_document(components["vector_db"], request.query)
    answer = generate_answer(request.query, document, components["prompt"], components["llm"])
    return {"answer": answer}

if __name__ == "__main__":
    
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)