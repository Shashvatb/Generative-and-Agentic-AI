import streamlit as sl
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq # cannot use init_chat_model because it doesnt take api key as parameter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
load_dotenv()


@sl.cache_resource
def get_chain(api_key, model_name):
    if not api_key:
        return None
    
    # init model
    model = ChatGroq(model_name=model_name, groq_api_key=api_key, temperature=0.7,
                     streaming=True)
    
    # create prompt
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'you are a helpful assistant. you talk like a gen-z'),
        ('user', '{question}')
    ])
    chain = (
        prompt |
        model |
        StrOutputParser()
    )
    return chain



if __name__ == "__main__":
    # os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    ## Create streamlit page

    # config
    sl.set_page_config(page_title="Conversational Bot using Groq")#, page_icon="") can add an emoji as icon
    # title
    sl.title("LangChain bot with Groq")
    sl.markdown("Talk to this bot powered by llama hosted on groq")


    # side bar
    with sl.sidebar:
        sl.header("Settings")
        # API key
        api_key = sl.text_input("GROQ API key", type="password", help="get api key from groq.com")
        # model selection
        model_name = sl.selectbox("Model", ["llama-3.1-8b-instant", "not a real model"], index=0)
        # clear button
        if sl.button("Clear Chat"):
            sl.session_state.messages = []
            sl.rerun()

    
    # initialize session state
    if 'messages' not in sl.session_state:
        sl.session_state.messages = []

    # get chain
    chain = get_chain(api_key, model_name)
    if not chain:
        sl.warning("enter API key")
    else:
        # display messages
        for message in sl.session_state.messages:
            with sl.chat_message(message['role']):
                sl.write(message['content'])

        # chat input
        if question := sl.chat_input("ask me anything"):
            # add message to session state
            sl.session_state.messages.append({'role': 'user', 'content': 'question'})
            with sl.chat_message('user'):
                sl.write(question)
            # Generate response
            with sl.chat_message("assistant"):
                message_placeholder = sl.empty()
                full_response = ""
                
                try:
                    # Stream response from Groq
                    for chunk in chain.stream({"question": question}):
                        full_response += chunk
                        message_placeholder.markdown(full_response + " ")
                    
                    message_placeholder.markdown(full_response)
                    
                    # Add to history
                    sl.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    sl.error(f"Error: {str(e)}")
    
   