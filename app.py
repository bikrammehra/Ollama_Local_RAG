import os
from langchain_chroma import Chroma
import streamlit as st
from typing import List
import tempfile
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders.pdf import PDFPlumberLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Set page config
st.set_page_config(page_title="RAG Chat Assistant", layout="wide")

def initialize_chroma(embeddings):
    """Initialize Chroma with temporary directory"""
    temp_dir = tempfile.mkdtemp()
    vector_store = Chroma(
        persist_directory=temp_dir,
        embedding_function=embeddings
    )
    return vector_store

def process_pdf(uploaded_file) -> List:
    """Process uploaded PDF file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PDFPlumberLoader(tmp_file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Clean up temporary file
    os.unlink(tmp_file_path)
    
    return chunks

def setup_rag_chain(vector_store):
    """Set up the RAG chain"""
    llm = ChatOllama(model="llama3.2")
    
    template = """You are an AI language model assistant. Your task is to generate five different versions \
    of the given user question to retrieve relevant documents from a vector database. By generating multiple \
    perspectives on the user question, your goal is to help the user overcome some of the limitations of the \
    distance-based similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}"""
    
    # template = """You are an AI assistant tasked with helping users find solutions to questions from English books. Given a user query, your goal is to generate five different versions of the question to help retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, you help the user overcome the limitations of distance-based similarity search.
    # Original question: {question}"""

    prompt = ChatPromptTemplate.from_template(template)
    
    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(),
        llm=llm,
        prompt=prompt
    )
    
    response_prompt = ChatPromptTemplate.from_template(
        """Based on the following context and question, provide a comprehensive answer:
        Context: {context}
        Question: {question}
        Answer:"""
    )
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | response_prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain

def main():
    st.title("📚 RAG Chat Assistant")
    
    # File upload section
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                # Initialize embeddings
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                
                # Initialize Chroma
                st.session_state.vector_store = initialize_chroma(embeddings)
                
                # Process each PDF and add to vector store
                all_chunks = []
                for uploaded_file in uploaded_files:
                    chunks = process_pdf(uploaded_file)
                    all_chunks.extend(chunks)
                
                # Add all documents to vector store
                st.session_state.vector_store.add_documents(all_chunks)
                
                # Setup RAG chain
                st.session_state.chain = setup_rag_chain(st.session_state.vector_store)
                st.success("Documents processed successfully!")
    
    # Query section
    st.header("2. Ask Questions")
    
    # Text input
    question = st.text_input("Type your question here:")
    
    if st.button("Get Answer"):
        if not st.session_state.chain:
            st.error("Please process documents first!")
            return
            
        if not question:
            st.warning("Please ask a question!")
            return
            
        with st.spinner("Generating response..."):
            response = st.session_state.chain.invoke(question)
            st.write("Answer:", response)
            # Add to chat history
            st.session_state.chat_history.append({"question": question, "answer": response})
    
    # Display chat history in the sidebar
    st.sidebar.header("Chat History")
    for chat in st.session_state.chat_history:
        st.sidebar.write(f"**Q:** {chat['question']}")
        st.sidebar.write(f"**A:** {chat['answer']}")
        st.sidebar.write("---")

if __name__ == "__main__":
    main()
