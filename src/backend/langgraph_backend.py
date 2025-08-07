from langgraph.graph import StateGraph,START, END
from typing import TypedDict, Literal, Annotated
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.schema.runnable import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage
import operator
load_dotenv()

class ResponseState(TypedDict):
    user_query: str
    model_response: str
    messages: Annotated[list[BaseMessage], operator.add]

def page_loader(path:str):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    return pages

def extract_doc_and_metadata(pages):
    doc_contents = [doc.page_content for doc in pages]
    metadata = [doc.metadata for doc in pages]
    return doc_contents, metadata

def create_chunks(doc_contents, metadata):
    text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=200,
    chunk_overlap=50,
    )

    chunks = text_splitter.create_documents(doc_contents, metadata)
    return chunks

def add_documents_to_vector_store(chunks)-> None:
    vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory='my_chroma_db',
    collection_name='pdf_data1'
    )
    vector_store.add_documents(chunks)
    return vector_store

def prompt_template() -> PromptTemplate:
    prompt = PromptTemplate(
    template="""
     You are a helpful assistant.

    Use the context below to help answer the question.

    If the context is useful, base your answer on it. If it's not relevant or missing, use your general knowledge to help the user.

    Context:
    {context}

    Question: {question}

    """,
    input_variables = ['context', 'question'],
    
    )
    return prompt

def format_docs(docs):
    formatted = []
    for doc in docs:
        meta = "\n".join([f"{k.capitalize()}: {v}" for k, v in doc.metadata.items()])
        formatted.append(f"{meta}\n\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

def rag_chain(retriever, llm, prompt):
    rag_chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnableLambda(lambda x: x)}
    | prompt
    | llm
    | StrOutputParser()
    )
    return rag_chain

def chat_bot(state:ResponseState):
    messages = state.get("messages", [])

    llm = ChatOpenAI(model="gpt-4o-mini")

    query = state.get('user_query')
    pages = page_loader("resources/lose-fat-get-fittr-the-simple-science-of-staying-healthy-for-life-firstnbsped.pdf")
    doc_contents, metadata = extract_doc_and_metadata(pages)
    chunks = create_chunks(doc_contents, metadata)
    vector_store = add_documents_to_vector_store(chunks)
    prompt = prompt_template()
    retriever = vector_store.as_retriever()
    rag = rag_chain(retriever, llm, prompt)
    response = rag.invoke(query)
    return {'model_response': response}

graph = StateGraph(ResponseState)

graph.add_node("chat_bot", chat_bot)
graph.add_edge(START, "chat_bot")
graph.add_edge("chat_bot", END)


checkpointer = InMemorySaver()
workflow = graph.compile(checkpointer=checkpointer)