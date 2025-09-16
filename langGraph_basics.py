"""
build stateful, multi-actor applications with LLMs
used to create multi agent workflows
langGraph can have cycles (as opposed to a DAG based solution like langchain)
simplifies development - state management and multi agent coordination
just like the whole family, it is flexible
it is scalable
it has fault tolerance: can handle errors internally (works with coordination)
langGraph creates a graph to execute
"""
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

import os
load_dotenv()

class State(TypedDict):
    messages:Annotated[list, add_messages]


def chatbot(state:State):
    return {"messages":llm.invoke(state['messages'])}

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "LangGraph"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = OllamaLLM(model='llama3.1')

# basic graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

## visualization
graph.get_graph().draw_png("langgraph.png")

## inputs
while True:
    user_input = input("User: ")
    if user_input.lower() == 'quit':
        break
    for event in graph.stream({'messages': ('user', user_input)}):
        for value in event.values():
            print("Assistant: ", value["messages"])


# multi agent graph
## using groq api to use tools
llm = init_chat_model('groq:qwen/qwen3-32b')
## adding tools
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
print(wiki_tool.invoke('tell me about pokemon'))
arxiv_wrapper = ArxivAPIWrapper(top_k_results=5, doc_content_chars_max=300)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
print(arxiv_tool.invoke('attention is all you need'))
tools = [wiki_tool, arxiv_tool]

llm = llm.bind_tools(tools)


## creating graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
tool_node = ToolNode(tools)
graph_builder.add_node('tools', tool_node)
graph_builder.add_conditional_edges('chatbot', tools_condition)
graph_builder.add_edge("tools", "chatbot" )
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

## visualization
graph.get_graph().draw_png("langgraph_with_tools.png")

## inputs
# while True:
#     user_input = input("User: ")
#     if user_input.lower() == 'quit':
#         break
#     for event in graph.stream({'messages': ('user', user_input)}):
#         for value in event.values():
#             print("Assistant: ", value["messages"].content)
while True:
    user_input = input("User: ")
    if user_input.lower() == "quit":
        break

    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            # LLM message
            if "messages" in value and hasattr(value["messages"], "content"):
                print("Assistant:", value["messages"].content, '\n')
            # Tool message
            elif "messages" in value and isinstance(value["messages"], list):
                for msg in value["messages"]:
                    print(f"{msg.name} (tool): {msg.content}", '\n')
            else:
                print("Unknown output:", value, '\n')
