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

```
.
├── .dockerignore         # Specifies files to ignore in the Docker build context
├── .env                  # Your environment variables (API keys, URLs) - DO NOT COMMIT
├── .gitignore            # Specifies files to ignore for Git
├── Dockerfile            # Instructions to build the FastAPI application image
├── RAGpipline.py         # The core FastAPI application logic
├── docker-compose.yml    # Defines and runs the multi-container application (API + Weaviate)
└── requirements.txt      # Python dependencies
```

## 🏁 Getting Started

Follow these steps to get the project running on your local machine.

### Prerequisites

* **Docker Desktop**: Make sure Docker and Docker Compose are installed and running on your system. You can download it from [docker.com](https://www.docker.com/products/docker-desktop/).

### 1. Clone the Repository

```bash
git clone [https://github.com/Nuwanga-Wijamuni/RAG_Docker.git](https://github.com/Nuwanga-Wijamuni/RAG_Docker.git)
cd RAG_Docker
```

### 2. Set Up Environment Variables

Create a file named `.env` in the root of the project directory and add the following variables.

```env
# .env

# --- Weaviate Configuration ---
# For local Docker setup, use the service name from docker-compose.yml
WEAVIATE_URL="http://weaviate:8080"
# For Weaviate Cloud Service (WCS), use your cluster URL and uncomment the API key
# WEAVIATE_URL="[https://your-wcs-instance.weaviate.network](https://your-wcs-instance.weaviate.network)"
# WEAVIATE_API_KEY="YOUR_WEAVIATE_API_KEY"

# --- Groq Configuration ---
GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

**Important**:
* Replace `YOUR_GROQ_API_KEY` with your actual key from the [Groq Console](https://console.groq.com/keys).
* For this local Docker setup, the `WEAVIATE_URL` **must** be `http://weaviate:8080`.

### 3. Build and Run with Docker Compose

Open your terminal in the project root and run the following command:

```bash
docker-compose up --build
```

This command will:
* Pull the specified Weaviate image.
* Build your FastAPI application's Docker image based on the `Dockerfile`.
* Start both containers and connect them on a shared network.

Your API will be available at `http://localhost:8081`.

## 📖 API Endpoints

The API provides the following endpoints to interact with the RAG system.

| Method | Path | Description |
| :--- | :--- | :--- |
| `POST` | `/sessions` | Creates a new session with a description. |
| `GET` | `/sessions` | Lists all existing sessions. |
| `POST` | `/upload-pdf/{session_id}` | Uploads a PDF to a specific session. |
| `GET` | `/query/{session_id}` | Asks a question within a session's context. |
| `DELETE`| `/documents` | Deletes a specific PDF from a session. |

### Example Usage with `curl`

1.  **Create a session:**
    ```bash
    curl -X POST "http://localhost:8081/sessions" -H "Content-Type: application/json" -d '{"description": "My Research Papers"}'
    ```
    *(Save the `session_id` from the response)*

2.  **Upload a PDF:**
    ```bash
    curl -X POST "http://localhost:8081/upload-pdf/YOUR_SESSION_ID" -F "file=@/path/to/your/document.pdf"
    ```

3.  **Ask a question:**
    ```bash
    curl -G "http://localhost:8081/query/YOUR_SESSION_ID" --data-urlencode "question=What is the main conclusion of the paper?"
    ```

## 🔧 Customization

* **Embedding Model**: You can change the model in `RAGpipline.py` by updating the `SentenceTransformer` line.
* **LLM Model**: To use a different model from Groq (e.g., `mixtral-8x7b-32768`), change the `model` parameter in the `groq_client.chat.completions.create` call.
* **Chunk Size**: Modify the `CHUNK_SIZE` global variable to control how text is split.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

