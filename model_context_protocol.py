"""
concept by anthropic
open protocol that standardizes how application provides context to LLMs
the client sends an http(s) request to a server and gets a response. It has many endpoints (or services).       [websites]
Http(s) is the protocol. we can use this irrespective of the language.                                          [websites]
Generative AI - give input to LLM, it will give output based on the training data
Agentic AI - we use multiple tools which are integrated with LLM to help us achieve a specific task

Problem:
if we have to integrate the same set of tools for multiple agents, we need to do it over and over again.
if there is an update to the tool, we have to redo everything.
Solution:
We can use MCP which works as a medium where we can separate out the tools and integrate with our LLM
All the tool providers need to follow protocol and it makes integration much easier
Components:
MCP Host - It can be any IDE, it creates a MCP client (as an extension) which interacts with the MCP server. It just HOSTS the client
MCP Client - API or interactive application
MCP Server - Connected to a tool or service. we can have any number of MCP servers. All these will be managed by service provider
The integration between our application and MCP server will remain the same even if something changes in the backend

The MCP client communicates with MCP server and gets the list of all the services (or tools) available. 
the input and list of tools go to the MCP client which is sent to the LLM
The llm will request the tools needed
The tool is called and the context is sent to the llm and we get the output
"""