import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from tools import search_tool, retrieve_chunks
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field


tools = ToolNode([search_tool, retrieve_chunks])

hive_api_key = os.environ.get("HIVE_API_KEY")
hive_base_url = os.environ.get("HIVE_BASE_URL", "https://api-cdn.thehive.ai/api/v3")
hive_model = os.environ.get("HIVE_MODEL", "deepseek-ai/deepseek-v4.1-flash")

initial_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that helps the user in retrieving relevant documents based on the query\n"
            "You have access to the following tool: \n"
            "retrieve_chunks that retrieves the relevant documents from the database\n"
            "If you think the query does not require any retrieval, do not retrieve anything\n"
        ),
        (
            "human",
            "{query}"
        )
    ]
)

initial_llm = ChatOpenAI(
    base_url=hive_base_url,
    api_key=hive_api_key,
    model=hive_model,
    temperature=0
).bind_tools([retrieve_chunks])

initial_chain = initial_retrieval_prompt | initial_llm


comparison_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that helps the user in comparing the relevancy of research papers and the query\n"
            "If they are relevant to the query, do not change anything and pass them as is\n"
            "If the research papers given by the user are not relevant to the query, use the search_tool to search for relevant documents on the arxiv archive and ingest them in the database\n"
        ),
        (
            "human",
            "query: {query}\n"
            "research_papers: {documents}"

        )
    ]
)

comparison_llm = ChatOpenAI(
    base_url=hive_base_url,
    api_key=hive_api_key,
    model=hive_model,
    temperature=0
).bind_tools([search_tool])


comparison_chain = comparison_prompt | comparison_llm


response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that gives response based on the context given\n"
            "The context you recieve are a list of relevant research papers based on the query\n"
            "Look at the query and the research papers to formulate appropriate response for the user\n"
            "context: {documents}"
        ),
        (
            "human",
            "query: {query}"
        )
    ]
)

response_llm = ChatOpenAI(
    base_url=hive_base_url,
    api_key=hive_api_key,
    model=hive_model,
)

response_chain = response_prompt | response_llm

class ScopeCheck(BaseModel):
    scope: bool = Field(description = "'ACCEPT' if the query is within the scope else 'DECLINE'")

scope_prompt = ChatPromptTemplate.from_messages(
    [(
        "system",
        "You are a helpful classifier that classifies where the user query falls within the scope of reasoning or not.\n"
        "The scope here is Computer science related research papers present of the arxiv website\n"
        "Only accept the query if the user query is related to the defined scope above\n"
        "if the query is unrelated but it is like a generic one like 'hello', 'how are you', etc, accept it\n"
        "if it is outside the scope, give response as 'False'\n"
        "if the query is within the scope, give response as 'True'\n"
        "Respond ONLY with a JSON object in the format: {{\"scope\": true}} or {{\"scope\": false}}\n"
    ),
    (
        "human",
        "query: {query}"
    )]
)

scope_llm = ChatOpenAI(
    base_url=hive_base_url,
    api_key=hive_api_key,
    model=hive_model,
)

scope_chain = scope_prompt | scope_llm.with_structured_output(ScopeCheck, method="json_mode")