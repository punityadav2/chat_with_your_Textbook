import streamlit as st
import os
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables from.env file
load_dotenv()

def get_vectorstore_from_pdf(pdf_file):
    """
    Processes a PDF file, splits it into chunks, creates embeddings,
    and stores them in a Chroma vector store.
    """
    if "vector_store" in st.session_state and st.session_state.pdf_file_name == pdf_file.name:
        return st.session_state.vector_store

    try:
        # Save the uploaded file to a temporary location
        with open(pdf_file.name, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        # 1. Load the PDF document
        pdf_loader = PyPDFLoader(pdf_file.name)
        documents = pdf_loader.load()

        # 2. Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        document_chunks = text_splitter.split_documents(documents)

        # 3. Create an embedding model (using a local open-source model)
        # This avoids the need for an OpenAI API key for embeddings
        embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        # 4. Create a vector store from the chunks
        vector_store = Chroma.from_documents(document_chunks, embeddings_model)
        
        # Clean up the temporary file
        os.remove(pdf_file.name)

        # Cache the vector store and file name in session state
        st.session_state.vector_store = vector_store
        st.session_state.pdf_file_name = pdf_file.name
        
        return vector_store

    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None


def get_retrieval_chain(groq_api_key, vector_store):
    """
    Creates and returns a retrieval chain for question answering.
    """
    # Initialize the Groq LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="")

    # Create a retriever from the vector store
    retriever = vector_store.as_retriever()

    # Define the prompt template
    RAG_PROMPT_TEMPLATE = """
    You are an expert assistant for question-answering tasks.
    Use the following retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Keep the answer concise and use a maximum of three sentences.

    <context>
    {context}
    </context>

    Question: {input}
    """
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # Create the document processing chain
    stuff_documents_chain = create_stuff_documents_chain(llm, rag_prompt)
    
    # Create the final retrieval chain
    return create_retrieval_chain(retriever, stuff_documents_chain)

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Chat with Your Textbook", page_icon="📖")
st.title("Chat with Your Data Science Textbook 📖")

# --- Sidebar for API Key and PDF Upload ---
with st.sidebar:
    st.header("Settings")
    
    # Get Groq API key from user or.env file
    groq_api_key = st.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password")
    
    # PDF file uploader
    pdf_file = st.file_uploader("Upload your PDF Textbook", type="pdf")

# --- Main Application Logic ---
if not groq_api_key:
    st.info("Please enter your Groq API key in the sidebar to continue.")
elif not pdf_file:
    st.info("Please upload a PDF file to begin.")
else:
    # Initialize session state for chat history if it doesn't exist
    if "chat_history" not in st.session_state or st.session_state.get("pdf_file_name")!= pdf_file.name:
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm ready to answer questions about your textbook."),
        ]
        # Clear old vector store if a new file is uploaded
        if "vector_store" in st.session_state:
            del st.session_state.vector_store

    # Process the PDF and create the RAG chain
    with st.spinner("Processing PDF... This may take a moment."):
        vector_store = get_vectorstore_from_pdf(pdf_file)
    
    if vector_store:
        retrieval_chain = get_retrieval_chain(groq_api_key, vector_store)

        # Display conversation history
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)

        # Get user input from chat box
        user_query = st.chat_input("Ask a question about your textbook...")
        if user_query:
            # Add user message to history and display it
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            with st.chat_message("Human"):
                st.markdown(user_query)

            # Get AI response using the RAG chain and display it
            with st.chat_message("AI"):
                # Use the streaming capability for a better user experience
                response_generator = retrieval_chain.stream({"input": user_query})
                
                # st.write_stream consumes the generator and displays the output in real-time
                full_response = st.write_stream(item['answer'] for item in response_generator if 'answer' in item)

            # Add the complete AI response to the chat history
            st.session_state.chat_history.append(AIMessage(content=full_response))

