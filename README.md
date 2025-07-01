# RAG Pipeline with Weaviate, Groq, and FastAPI

![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Weaviate](https://img.shields.io/badge/Weaviate-0C9E73?style=for-the-badge&logo=weaviate&logoColor=white)

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline using Docker. It allows you to upload PDF documents, which are then processed, chunked, and stored as vector embeddings in a Weaviate database. You can then ask questions related to the document content, and the system will retrieve relevant context to generate a precise answer using the Groq API with LLaMA 3.

## 🚀 Core Technologies

| Technology | Role |
| :--- | :--- |
| **FastAPI** | Serves the application through a robust and efficient REST API. |
| **Weaviate** | Acts as the vector database for storing and retrieving text chunks. |
| **Groq & LLaMA 3**| Provides the high-speed, high-quality language model for generating answers. |
| **Sentence Transformers** | Creates the vector embeddings from text chunks. |
| **Docker & Docker Compose** | Containerizes the entire application stack for easy setup and deployment. |
| **PyMuPDF** | Extracts text content from the uploaded PDF documents. |

## ⚙️ How It Works

The application follows a sophisticated, session-based workflow:

1.  **Session Creation**: A unique session is created to isolate documents and queries.
2.  **PDF Upload & Processing**: A PDF is uploaded under a specific session. The system checks for duplicates within that session to prevent reprocessing.
3.  **Text Extraction & Chunking**: The text is extracted page by page. Each page's text is then split into smaller, manageable chunks.
4.  **Vectorization**: Each text chunk is converted into a numerical vector embedding using a Sentence Transformer model.
5.  **Storage**: The chunks, along with their metadata (session ID, filename, page number) and vector embeddings, are stored in the Weaviate vector database.
6.  **Querying (The RAG part)**:
    * When a user asks a question, the question is also converted into a vector.
    * Weaviate performs a similarity search to find the text chunks with vectors most similar to the question's vector.
    * This retrieved context is passed to the Groq API along with the original question.
7.  **Answer Generation**: The LLaMA 3 model uses the provided context to generate a factual, accurate answer, strictly based on the information from the documents.

## 📂 Project Structure
