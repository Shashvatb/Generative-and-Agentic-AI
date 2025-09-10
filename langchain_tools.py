from langchain_core.tools import tool, StructuredTool
from typing import Annotated, List
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_community.tools import WikipediaQueryRun 
from langchain_community.utilities import WikipediaAPIWrapper
import asyncio
import os


# basic tool
@tool
def division(a:int, b:int)->int:
    """
    Divides two numbers
    """
    return a/b

# async tool
@tool
async def async_div(a:int, b:int)->int:
    """
    Divides two numbers
    """
    return a/b

# Annotated tool
@tool
def multiply_by_max(a:Annotated[int, "value of a"], b:Annotated[List[int], "value of bs"])-> int:
    """
    multiple a with max of b
    """
    return a * max(b)

# Structured tool functions
def multiply(a:int, b:int)->int:
    """
    multiply two numbers
    """
    return a*b

async def amultiply(a:int, b:int)->int:
    """
    multiply two numbers
    """
    return a*b


async def main():
    print(division.name)
    print(division.description)
    print(division.args)

    print(async_div.name)
    print(async_div.description)
    print(async_div.args)

    print(multiply_by_max.name)
    print(multiply_by_max.description)
    print(multiply_by_max.args)

    # Structured tool
    calc = StructuredTool.from_function(func=multiply, coroutine=amultiply) # coroutine takes async functions
    print(calc.invoke({"a":2, "b":3}))
    print(await calc.ainvoke({"a": 2, "b": 30}))

    # In-built tools
    ## wikipedia integration
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=5000)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
    tool_response = wiki_tool.invoke({"query": "pokemon"})
    print(tool_response)

    ## integrate tool with LLM
    tools = [wiki_tool, division, multiply_by_max]
    llm = init_chat_model('groq:qwen/qwen3-32b')
    llm_with_tools = llm.bind_tools(tools)
    query = "what is 8 divided by 4"
    print()
    response = llm_with_tools.invoke(query)
    print(response)
    # in the response, content is blank. we can see it called a funtion (divison)
    # it uses doctrings to see if this works better and calls it
    # we can see the tool_calls
    print(response.tool_calls)
    messages = [HumanMessage(query)]
    for tool_call in response.tool_calls:
        selected_tool = {'division': division, 'wikipedia':wiki_tool, 'multiply_by_max':multiply_by_max}[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)
    print("\n",messages)

    # now we can pass the messages to the llm with tools
    print("\n",llm_with_tools.invoke(messages))

    # now lets try a complex query which requires llm and tools
    query = "tell me a little about pokemon and tell me what is 10 divided by 5"
    messages = [HumanMessage(query)]
    response = llm_with_tools.invoke(messages)
    print("\n",response)

    print('\n',response.tool_calls)

    for tool_call in response.tool_calls:
        selected_tool = {'division': division, 'wikipedia':wiki_tool, 'multiply_by_max':multiply_by_max}[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)
    print("\n",messages)
    print("\n",llm_with_tools.invoke(messages))



if __name__ == "__main__":
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    asyncio.run(main())