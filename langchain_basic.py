import langchain
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq # legacy
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
load_dotenv()

def get_haiku_text(text):
    return {'haiku': text}


if __name__ == '__main__':
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    ## Simple LLM call with streaming
    #initialize chat model
    model = init_chat_model('groq:llama-3.1-8b-instant') # new method to unify all chat model init
    """
    model = ChatGroq(model='llama-3.1-8b-instant') # legacy way of doing it, can use init_chat_model from now on
    """

    # message creation
    messages = [
        SystemMessage("This is an init message"), # Always passed as the first message
        HumanMessage("What are the top 3 most populous countries"), # input provided by the user
    ]

    # using the model (invoke)
    response = model.invoke(messages)
    print(response) # this gives us the AI message
    # if we want the response content, we will call
    print(response.content)

    # streaming the response
    for chunk in model.stream(messages):
        print(chunk.content, end="\n new chunk \n", flush=True)

    ## Prompts - we can give it an instruction. here we convert the response to german
    messages = [
        SystemMessage("Give me the response in german"),
        HumanMessage("What are the top 3 most populous countries"),
    ]
    response = model.invoke(messages)
    print(response.content)

    ## dynamic prompt template - we can have prompts with multiple rules with this    
    translation_template = ChatPromptTemplate.from_messages([ # can create template from variety of message formats
        ('system', 'translate the text to {language}'),
        ('user', '{text}')
    ])

    language_1 = 'german'
    text = 'hello i am an AI talking to you'

    # using the template
    prompt = translation_template.invoke({
        'language': language_1,
        'text': text

    })
    print(prompt) # this adds the values to the template as a system message

    # using the model with prompt
    translated_result = model.invoke(prompt)
    print(translated_result.content)

    ## building a chain of events
    # event 1
    prompt_create_haiku = ChatPromptTemplate.from_messages([
        ('system', 'you are a creative AI, write a haiku based on a given theme'),
        ('user', 'Theme: {theme}')
    ])

    # event 2
    prompt_judge_haiku = ChatPromptTemplate.from_messages([
        ('system', 'you are a critic and want to judge how good the haiku is. also print the haiku before analysing it'),
        ('user', 'Haiku: \n {haiku}')
    ])

    # create chain
    haiku_create_chain = (
        prompt_create_haiku
        | model 
        | StrOutputParser())
    
    haiku_judge_chain = (
        haiku_create_chain
        | RunnableLambda(get_haiku_text) # this is a simple function but we can have it much more complicated based on our use case
        | prompt_judge_haiku 
        | model 
        | StrOutputParser()
    ) # this would be our final result
    
    # invoking the chain
    result = haiku_judge_chain.invoke({
        'theme': 'summer'
    })
    print('\n\n\n HAIKU \n')
    print(result)
