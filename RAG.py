# optimize output of LLM so it refers to a large knowledge base outside of its training data
# finetuning is expensive so we use RAGs (especially if it changes a lot)
# we create a vector database and query it like a tool
# vector DB: take data, preprocess, chunk into smaller data, embedding, store in a DB
# can apply similarity based search etc in vector DB
# query is embedded and queried in vector DB, that response is context for an LLM (with prompt) and that is generated as output 