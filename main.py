import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
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

def main():
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





if __name__ == "__main__":
    main()
