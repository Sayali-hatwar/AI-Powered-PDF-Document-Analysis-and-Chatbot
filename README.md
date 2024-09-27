# AI-Powered-PDF-Document-Analysis-and-Chatbot

This project is focused on building an AI-powered document analysis and chatbot using Python. The application extracts text from uploaded PDF files and uses advanced natural language processing techniques to answer user queries based on the content of the PDFs. The core technologies include:

PyPDF2: Extracts text from PDF documents.
Langchain: Manages text splitting, embedding, and question-answering functionality.
Google Generative AI: Provides embeddings and conversational AI (Gemini Pro model) to generate responses.
FAISS: Efficient vector store for text embeddings, enabling similarity search for relevant content.
Streamlit: A web framework used to build a user-friendly interface for document upload, processing, and interacting with the chatbot.
The solution allows users to upload multiple PDF files, processes them by splitting text into manageable chunks, and builds an embedding-based vector store to handle document similarity searches. Users can ask questions based on the document contents, and the chatbot provides contextually accurate responses.

