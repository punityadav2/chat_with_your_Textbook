📚 Chat with Your Textbook
This is a powerful and intuitive application that allows you to have a conversation with your PDF documents. Built with Streamlit, LangChain, and the lightning-fast Groq LPU™ Inference Engine, this tool transforms any textbook, research paper, or manual into an interactive chatbot.

The application leverages the Retrieval-Augmented Generation (RAG) pattern, enabling a Large Language Model (LLM) to answer questions based on the specific content of your uploaded document.

✨ Features
Interactive Chat Interface: Ask questions in natural language and get answers directly from your document.

PDF Upload: Easily upload any PDF file to start a conversation.

Robust PDF Processing: Supports both text-based and scanned (image-based) PDFs thanks to its built-in Optical Character Recognition (OCR) fallback.

Blazing-Fast Responses: Powered by the Groq LPU™ Inference Engine using the Llama 3 model for near-instant answer generation.

Source Verification: Each answer is accompanied by the exact source text from the document, allowing you to verify the information's accuracy.

Self-Contained & Private: Your document is processed locally, and only the relevant context is sent to the LLM for answering your specific question.

⚙️ How It Works
The application follows a modern Retrieval-Augmented Generation (RAG) pipeline to provide answers grounded in your document's content.

PDF Ingestion & Text Extraction: When you upload a PDF, the system first tries to extract text directly. If it fails (as with a scanned document), it uses the Tesseract OCR engine to convert the pages into text.

Text Chunking: The extracted text is split into smaller, manageable chunks. This is necessary to fit the context into the language model's limits.

Embedding Generation: Each text chunk is converted into a numerical representation (a vector embedding) using a state-of-the-art Hugging Face Sentence Transformer model. This embedding captures the semantic meaning of the text.

Vector Storage: These embeddings are stored in a high-performance, in-memory FAISS vector store. This allows for incredibly fast similarity searches.

Retrieval & Generation:

When you ask a question, your query is also converted into an embedding.

FAISS searches the vector store to find the text chunks with embeddings most similar to your question's embedding.

The original question and the most relevant text chunks are combined into a prompt.

This prompt is sent to the Groq API, where the Llama 3 model generates a coherent, context-aware answer.

▶️ How to Run

streamlit run app.py

Your web browser should automatically open with the application interface.

Using the App:

Enter your Groq API Key in the sidebar.

Upload a PDF document.

Click the "Process Document" button and wait for the indexing to complete.

Ask your questions in the main chat input box!

📂 Project Structure
.
├── app.py                  # Main Streamlit application file      
├── requirements.txt        # List of Python dependencies
├── .env                    # For storing API keys (not included in git)
└── README.md               # This file

🛠️ Technologies Used
Application Framework: Streamlit

LLM Orchestration: LangChain

LLM Inference: Groq (Llama 3 8B)

Embeddings: Hugging Face Sentence Transformers (all-MiniLM-L6-v2)

Vector Store: FAISS (Facebook AI Similarity Search)

PDF Processing: PyPDF, pdf2image

OCR: Tesseract (pytesseract)