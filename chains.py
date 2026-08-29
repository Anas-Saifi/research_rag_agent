from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from tools import search_tool, retrieve_chunks
from langgraph.prebuilt import ToolNode


tools = ToolNode([search_tool, retrieve_chunks])

initial_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that helps the user in retrieving relevant documents based on the query\n"
            "You have access to the following tool: \n"
            "retrieve_chunks that retrieves the relevant documents from the database\n"
        ),
        (
            "human",
            "{query}"
        )
    ]
)

initial_llm = ChatGoogleGenerativeAI(model = "gemini-3.6-flash", temperature = 0).bind_tools([retrieve_chunks])

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

comparison_llm = ChatGoogleGenerativeAI(model = "gemini-3.6-flash", temperature = 0).bind_tools([search_tool])


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

response_llm = ChatGoogleGenerativeAI(model = "gemini-3.6-flash")

response_chain = response_prompt | response_llm