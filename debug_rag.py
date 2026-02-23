from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import create_retriever_tool

try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text = "This is a test medical report."
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    llm = ChatGroq(temperature=0.7, model="llama-3.3-70b-versatile")
    web_search_tool = DuckDuckGoSearchRun()
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retriever_tool = create_retriever_tool(
        retriever,
        "medical_report_retriever",
        "Searches and returns excerpts from the uploaded medical report."
    )
    
    print(f"Retriever tool type: {type(retriever_tool)}")
    print(f"Retriever tool func: {retriever_tool.func}")
    
    tools = [retriever_tool, web_search_tool]
    
    print("Creating agent...")
    agent_executor = create_react_agent(llm, tools)
    print("Agent created successfully.")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
