# RAG Chat Assistant (💯 Local)

RAG Chat Assistant is a Streamlit application that uses the LangChain library to perform retrieval-augmented generation (RAG) for answering user questions based on uploaded PDF documents.

![image](https://github.com/user-attachments/assets/d9d3da09-37d3-46bc-b6c1-dff7574fa9b5)

## Features

- Upload multiple PDF documents
- Process and split PDF documents into chunks
- Generate multiple perspectives on user questions using LangChain's MultiQueryRetriever
- Retrieve relevant documents from a vector database using Chroma
- Display answers to user questions
- Store chat history in the sidebar

## Prerequisites

- Python 3.8 or higher
- Ollama and the model you want to use https://ollama.com/download
- ollama pull llama3.2 on you terminal to pull the llama3.2 model to use it in our RAG app Similarly you can choose any model from https://ollama.com/search

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/bikrammehra/Ollama_Local_RAG.git
    cd OLLAMA_LOCAL_RAG
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

2. Open your web browser and navigate to `http://localhost:8501` to view the application.

3. Upload your PDF documents, ask questions, and view the chat history in the sidebar.

## Required Libraries

The following libraries are used in this project:

- `streamlit`: For creating the web application.
- `langchain_chroma`: For working with the Chroma vector database.
- `tempfile`: For handling temporary files.
- `typing`: For type hints.
- `os`: For file system operations.
- `langchain_ollama`: For interacting with the Ollama language model.
- `langchain_community.document_loaders.pdf`: For loading PDF documents.
- `langchain_text_splitters.character`: For splitting text into chunks.
- `langchain_core.prompts`: For creating prompts.
- `langchain.retrievers.multi_query`: For using the MultiQueryRetriever.
- `langchain_core.runnables.passthrough`: For passthrough runnables.
- `langchain_core.output_parsers.string`: For parsing output as strings.
