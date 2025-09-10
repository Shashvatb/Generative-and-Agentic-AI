from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

language_1 = 'german'
text = 'hello i am an AI talking to you'

prompt_create_haiku = ChatPromptTemplate.from_messages([
        ('system', 'you are a creative AI, write a haiku based on a given theme'),
        ('user', 'Theme: {theme}')
    ])

"""
need to install ollama for this

download it:
curl -fsSL https://ollama.com/install.sh | sh

check installation
ollama -v

run a model of choice. it needs to run it here in order to download the model first
ollama run llama-2
"""
llm = OllamaLLM(model='llama2')

haiku_create_chain = prompt_create_haiku | llm | StrOutputParser()

result = haiku_create_chain.invoke({
        'theme': 'summer'
    })

print(result)
