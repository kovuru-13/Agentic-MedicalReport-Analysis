from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool

from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The search query to look up on the internet.")

class RAGSystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self.agent_executor = None
        # Initialize Groq model for chat
        self.llm = ChatGroq(temperature=0.7, model="openai/gpt-oss-safeguard-20b")
        self.web_search_tool = DuckDuckGoSearchRun()

    def ingest(self, text):
        """
        Ingests text, splits it into chunks, and creates a vector store.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        
        self.vector_store = FAISS.from_texts(chunks, self.embeddings)
        self.setup_agent()

    def setup_agent(self, enable_web_search=False):
        """
        Sets up the RAG agent with retriever and optionally web search tools.
        """
        if not self.vector_store:
            return

        # Create retriever tool
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        retriever_tool = Tool(
            name="medical_report_retriever",
            func=retriever.invoke,
            description="Searches and returns excerpts from the uploaded medical report."
        )

        tools = [retriever_tool]
        
        if enable_web_search:
            # Wrapper function to handle the schema
            def search_func(query: str):
                return self.web_search_tool.invoke(query)

            # Wrap the search tool to ensure correct schema
            search_tool = Tool(
                name="duckduckgo_search",
                func=search_func,
                description="Useful for searching the internet for medical information not present in the report.",
                args_schema=SearchInput
            )
            tools.append(search_tool)

        # Create the agent using langgraph
        self.agent_executor = create_react_agent(self.llm, tools)

    def update_tools(self, enable_web_search):
        """
        Updates the tools available to the agent.
        """
        self.setup_agent(enable_web_search)

    def query(self, question):
        """
        Queries the RAG agent.
        """
        if not self.agent_executor:
            return "Please upload a medical report first."
        
        try:
            # LangGraph agent expects 'messages' key
            response = self.agent_executor.invoke(
                {"messages": [("human", question)]},
                config={"recursion_limit": 50}
            )
            # The response contains the state, we want the last message content
            return response["messages"][-1].content
        except Exception as e:
            return f"An error occurred: {str(e)}"
