import os
import uuid
import json
import weaviate
import time
import fitz  # PyMuPDF
import tempfile # For handling temporary uploaded files
from contextlib import asynccontextmanager
from urllib.parse import urlparse
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
from weaviate.classes.config import Configure, Property, DataType # Updated imports
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel # For request body validation
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --- Configuration ---
SESSIONS_FILE = "sessions.json" # For storing session IDs and descriptions
CHUNK_SIZE = 500 # Words per chunk (approximately)

# --- Global Variables for Clients ---
embedding_model = None
groq_client = None
weaviate_client_instance = None
WEAVIATE_CLUSTER_URL = None
WEAVIATE_API_KEY_VALUE = None
GROQ_API_KEY_VALUE = None


def initialize_dependencies():
    """
    Initializes all client dependencies.
    This function is now smart enough to connect to either a local Weaviate instance
    or a Weaviate Cloud Service based on the presence of an API key.
    """
    global embedding_model, groq_client, weaviate_client_instance, WEAVIATE_CLUSTER_URL, WEAVIATE_API_KEY_VALUE, GROQ_API_KEY_VALUE

    print("Loading environment variables...")
    load_dotenv()

    # Load all potential environment variables
    loaded_url = os.getenv('WEAVIATE_URL')
    loaded_api_key = os.getenv('WEAVIATE_API_KEY')
    loaded_groq_key = os.getenv('GROQ_API_KEY')

    # --- MODIFICATION 1: Updated Validation Logic ---
    # We always need a Weaviate URL and a Groq API key.
    if not loaded_url or not loaded_groq_key:
        missing = []
        if not loaded_url: missing.append("WEAVIATE_URL")
        if not loaded_groq_key: missing.append("GROQ_API_KEY")
        error_message = f"CRITICAL ERROR: One or more environment variables are missing: {', '.join(missing)}"
        print(error_message)
        raise EnvironmentError(error_message)

    WEAVIATE_CLUSTER_URL = loaded_url
    WEAVIATE_API_KEY_VALUE = loaded_api_key
    GROQ_API_KEY_VALUE = loaded_groq_key

    print(f"DEBUG PRINT: Weaviate URL is '{WEAVIATE_CLUSTER_URL}'")
    print(f"DEBUG PRINT: Groq Key is present: {'Yes' if loaded_groq_key else 'No'}")
    print(f"DEBUG PRINT: Weaviate API Key is present: {'Yes' if loaded_api_key else 'No'}")


    print("Initializing SentenceTransformer model...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    print("SentenceTransformer model loaded.")

    print("Initializing Groq client...")
    groq_client = Groq(api_key=GROQ_API_KEY_VALUE)
    print("Groq client initialized.")

    print("Initializing Weaviate client...")
    try:
        # --- MODIFICATION 2: Conditional Weaviate Connection ---
        # If an API key is provided, we connect to the Weaviate Cloud Service.
        if WEAVIATE_API_KEY_VALUE:
            print("API Key found, attempting to connect to Weaviate Cloud...")
            weaviate_client_instance = weaviate.connect_to_weaviate_cloud(
                cluster_url=WEAVIATE_CLUSTER_URL,
                auth_credentials=Auth.api_key(WEAVIATE_API_KEY_VALUE),
            )
        # If no API key is found, we connect to a local instance.
        else:
            print("No API Key found, attempting to connect to local Weaviate instance...")
            # We parse the URL to get the host and port for the local connection
            parsed_url = urlparse(WEAVIATE_CLUSTER_URL)
            weaviate_client_instance = weaviate.connect_to_local(
                host=parsed_url.hostname,
                port=parsed_url.port
            )
        
        # This check works for both connection types
        if weaviate_client_instance.is_ready():
            print("✅ Successfully connected to Weaviate.")
        else:
            raise RuntimeError("Weaviate client is not ready. Check URL and network.")
            
    except Exception as e:
        print(f"❌ Failed to initialize Weaviate client: {e}")
        raise RuntimeError(f"Weaviate client initialization failed: {str(e)}") from e

def close_weaviate_connection():
    global weaviate_client_instance
    if weaviate_client_instance:
        print("Closing Weaviate client connection...")
        try:
            weaviate_client_instance.close()
            print("Weaviate client connection closed.")
        except Exception as e:
            print(f"Error closing Weaviate connection: {e}")


# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup: Initializing dependencies...")
    try:
        initialize_dependencies()
        create_or_update_schema()

        try:
            if not os.path.exists(SESSIONS_FILE):
                with open(SESSIONS_FILE, "w") as f:
                    json.dump({}, f)
                print(f"Created empty sessions file: {SESSIONS_FILE}")
            else:
                with open(SESSIONS_FILE, "r+") as f:
                    content = f.read()
                    if not content.strip():
                        f.seek(0)
                        f.truncate()
                        json.dump({}, f)
                        print(f"Initialized empty or blank sessions file: {SESSIONS_FILE}")
                    else:
                        try:
                            f.seek(0)
                            json.load(f)
                        except json.JSONDecodeError:
                            print(f"Warning: {SESSIONS_FILE} contains invalid JSON. Re-initializing as empty.")
                            f.seek(0)
                            f.truncate()
                            json.dump({}, f)
        except IOError as e:
            print(f"Warning: Could not ensure {SESSIONS_FILE} exists, is writable, or initialize it: {e}")

        print("FastAPI application starting with session-based PDF uploads and page-wise chunking.")
        print("Application startup complete.")
        yield
    except Exception as e:
        print(f"An error occurred during application startup: {str(e)}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Startup initialization failed: {str(e)}") from e
    finally:
        print("Application shutdown: Cleaning up resources...")
        close_weaviate_connection()
        print("Application shutdown complete.")


app = FastAPI(lifespan=lifespan)

# --- Session Management ---
class SessionManager:
    @staticmethod
    def load_sessions():
        if os.path.exists(SESSIONS_FILE):
            try:
                with open(SESSIONS_FILE, "r") as f:
                    content = f.read()
                    if not content.strip(): return {}
                    return json.loads(content)
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error loading {SESSIONS_FILE}: {e}. Treating as empty.")
                return {}
        return {}

    @staticmethod
    def save_sessions(sessions_data):
        try:
            with open(SESSIONS_FILE, "w") as f:
                json.dump(sessions_data, f, indent=2)
        except Exception as e:
            print(f"Error saving {SESSIONS_FILE}: {e}")

    @staticmethod
    def session_exists(session_id: str) -> bool:
        sessions = SessionManager.load_sessions()
        return session_id in sessions

# --- Weaviate Schema ---
COLLECTION_NAME = "Document" # Consider making this more specific if you add other types

def create_or_update_schema():
    global weaviate_client_instance
    if not weaviate_client_instance:
        print("Error: Weaviate client not initialized. Cannot create/update schema.")
        raise RuntimeError("Weaviate client not initialized for schema creation.")

    required_props = {
        "session_id": DataType.TEXT,
        "document_uuid": DataType.TEXT,
        "filename": DataType.TEXT,
        "text": DataType.TEXT,
        "chunk_index": DataType.INT, # Overall chunk index for the document
        "page_number": DataType.INT, # Page number of this chunk
        "page_chunk_index": DataType.INT # Index of chunk within its page
    }
    prop_descriptions = {
        "session_id": "ID of the session to which this document chunk belongs",
        "document_uuid": "UUID of the specific document ingestion instance",
        "filename": "Original filename of the PDF",
        "text": "Text content of the chunk",
        "chunk_index": "Overall index of this chunk within the entire document",
        "page_number": "Page number in the original document from which this chunk was extracted",
        "page_chunk_index": "Index of this chunk within its specific page"
    }

    print(f"Checking if collection '{COLLECTION_NAME}' exists...")
    if weaviate_client_instance.collections.exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists. Verifying properties...")
        try:
            collection_config = weaviate_client_instance.collections.get(COLLECTION_NAME).config.get()
            existing_props = {p.name: p.data_type for p in collection_config.properties}
            
            missing_props = []
            type_mismatch_props = []

            for prop_name, expected_type in required_props.items():
                if prop_name not in existing_props:
                    missing_props.append(prop_name)
                elif existing_props[prop_name] != expected_type:
                    type_mismatch_props.append(f"{prop_name} (expected {expected_type}, found {existing_props[prop_name]})")
            
            if missing_props or type_mismatch_props:
                error_messages = []
                if missing_props:
                    error_messages.append(f"Missing properties: {', '.join(missing_props)}")
                if type_mismatch_props:
                    error_messages.append(f"Property type mismatches: {', '.join(type_mismatch_props)}")
                
                full_error_message = (f"Schema for '{COLLECTION_NAME}' is outdated or incorrect. "
                                      f"{'; '.join(error_messages)}. "
                                      "Please manually delete and recreate the collection or update the schema if data preservation is critical.")
                print(f"CRITICAL SCHEMA MISMATCH: {full_error_message}")

            else:
                print(f"All required properties exist with correct types in '{COLLECTION_NAME}'.")

        except Exception as e:
            print(f"Could not fully verify properties of existing collection '{COLLECTION_NAME}': {e}")
        return

    print(f"Creating collection: {COLLECTION_NAME}")
    try:
        properties_to_create = [
            Property(
                name=name,
                data_type=dtype,
                description=prop_descriptions[name]
            ) for name, dtype in required_props.items()
        ]
        weaviate_client_instance.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=properties_to_create
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully with new schema.")
    except Exception as e:
        print(f"Error creating collection '{COLLECTION_NAME}': {e}")
        raise

# --- PDF Processing ---
def pdf_to_page_texts(file_path: str) -> list[dict]:
    """
    Extracts text from each page of a PDF.
    Returns a list of dictionaries, where each dictionary contains:
        'page_number': int (1-based)
        'page_text': str (text content of the page)
    """
    print(f"Processing PDF with PyMuPDF (page-wise): {file_path}")
    pages_data = []
    try:
        doc = fitz.open(file_path)
        for page_num_zero_based, page in enumerate(doc):
            page_num_one_based = page_num_zero_based + 1
            blocks = page.get_text("blocks", sort=True)
            page_text_segments = []
            for block in blocks:
                if block[6] == 0:  # block_type 0 is text
                    page_text_segments.append(block[4]) # block[4] is the text
            
            page_text = "\n".join(page_text_segments).strip()
            
            if page_text:
                pages_data.append({"page_number": page_num_one_based, "page_text": page_text})
            else:
                print(f"Notice: No text extracted from page {page_num_one_based} of {os.path.basename(file_path)}.")
        doc.close()
        if not pages_data:
            print(f"Warning: No text content found in any page of {os.path.basename(file_path)}.")
    except Exception as e:
        print(f"Error reading PDF {os.path.basename(file_path)} with PyMuPDF: {e}")
        if "cannot open" in str(e).lower() or "format error" in str(e).lower() or "no objects found" in str(e).lower():
            print(f"PyMuPDF could not open or process the file. It might be corrupted, not a valid PDF, or empty: {file_path}")
    return pages_data

def chunk_page_text(text: str, chunk_size_words: int) -> list[str]:
    """Chunks text from a single page based on word count."""
    if not text.strip():
        return []
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size_words]) for i in range(0, len(words), chunk_size_words)]
    # print(f"Page text chunked into {len(chunks)} chunks of approximately {chunk_size_words} words each.")
    return chunks

# --- API Endpoints ---

class SessionCreationRequest(BaseModel):
    description: str

@app.post("/sessions", summary="Create a new session ID with a description, ensuring uniqueness in Weaviate")
async def create_session(request: SessionCreationRequest):
    # Ensure global variables are accessible if not passed directly or part of a class
    global weaviate_client_instance, COLLECTION_NAME, SessionManager

    if not weaviate_client_instance or not weaviate_client_instance.is_ready():
        print("Error: Weaviate client not ready during session creation.")
        raise HTTPException(status_code=503, detail="Weaviate client not ready. Cannot validate session ID.")

    max_retries = 10  # Safety break for the ID generation loop
    retries = 0
    new_session_id = ""
    is_globally_unique = False

    print("Attempting to create a new unique session ID...")

    while not is_globally_unique and retries < max_retries:
        potential_session_id = str(uuid.uuid4())
        print(f"Attempt {retries + 1}: Generated potential session ID: {potential_session_id}")

        # 1. Check uniqueness in local SESSIONS_FILE
        if SessionManager.session_exists(potential_session_id):
            print(f"Potential ID {potential_session_id} already exists in local sessions file.")
            retries += 1
            time.sleep(0.05)  # Small delay before trying a new UUID
            continue

        # 2. Check uniqueness in Weaviate collection for the 'session_id' property
        try:
            # Ensure COLLECTION_NAME is correctly defined (e.g., "Document")
            document_collection = weaviate_client_instance.collections.get(COLLECTION_NAME)
            
            # Query Weaviate to see if any object has this session_id
            # Weaviate Python client v4 uses 'Filter' from weaviate.classes.query
            response = document_collection.query.fetch_objects(
                limit=1, # We only need to know if at least one exists
                filters=Filter.by_property("session_id").equal(potential_session_id)
            )

            if len(response.objects) == 0:
                # ID does not exist in Weaviate for this property
                print(f"Potential ID {potential_session_id} is unique in Weaviate collection '{COLLECTION_NAME}'.")
                new_session_id = potential_session_id
                is_globally_unique = True
            else:
                # ID already exists in Weaviate
                print(f"Potential ID {potential_session_id} already exists in Weaviate collection '{COLLECTION_NAME}'.")
                retries += 1
                time.sleep(0.05) # Small delay
        
        except Exception as e:
            print(f"Error checking session ID '{potential_session_id}' in Weaviate: {e}")
            # Depending on the error, you might want to retry or fail immediately.
            # For critical errors (e.g., cannot connect to Weaviate), raising an exception is appropriate.
            # For transient errors, a retry might be okay, but the loop already handles retrying with a new ID.
            # If the collection doesn't exist, it will also raise an error here.
            # The lifespan manager should ensure the collection exists.
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error validating session ID with Weaviate: {str(e)}")

    if not is_globally_unique:
        # This should be extremely rare with UUIDv4 and a reasonable number of retries
        error_msg = "Failed to generate a unique session ID after multiple retries. This might indicate an issue with UUID generation or an unexpectedly high number of collisions."
        print(f"CRITICAL: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    # At this point, new_session_id is unique in both local file and Weaviate
    # Now, load the sessions from file once to add the new one, then save.
    sessions_data = SessionManager.load_sessions()
    sessions_data[new_session_id] = request.description
    SessionManager.save_sessions(sessions_data)

    print(f"Successfully created and validated new session: ID={new_session_id}, Description='{request.description}'")
    return {"session_id": new_session_id, "description": request.description}

@app.get("/sessions", summary="List all session IDs and their descriptions")
async def list_sessions():
    return SessionManager.load_sessions()

@app.get("/sessions/{session_id}", summary="Get description for a specific session ID")
async def get_session_description(session_id: str):
    sessions = SessionManager.load_sessions()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session ID not found.")
    return {"session_id": session_id, "description": sessions[session_id]}

@app.post("/upload-pdf/{session_id}", summary="Upload a PDF for a specific session ID (page-wise chunking)")
async def upload_pdf_for_session(session_id: str, file: UploadFile = File(...)):
    global weaviate_client_instance, embedding_model
    if not weaviate_client_instance or not embedding_model:
        raise HTTPException(status_code=503, detail="Server components (Weaviate/Embedding model) not ready.")
    if not SessionManager.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session ID '{session_id}' not found. Create session first.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

    # --- ADD THIS DUPLICATION CHECK ---
    try:
        collection = weaviate_client_instance.collections.get(COLLECTION_NAME)
        
        where_filter = Filter.all_of([
            Filter.by_property("session_id").equal(session_id),
            Filter.by_property("filename").equal(file.filename)
        ])

        # We only need to know if at least one object exists, so limit=1 is efficient.
        response = collection.query.fetch_objects(filters=where_filter, limit=1)

        if len(response.objects) > 0:
            print(f"Duplicate detected: File '{file.filename}' already exists in session '{session_id}'. Aborting upload.")
            # 409 Conflict is the appropriate HTTP status code for a duplicate resource.
            raise HTTPException(
                status_code=409, 
                detail=f"File '{file.filename}' already exists in this session. Please delete the existing file before uploading a new version."
            )
            
    except HTTPException as http_exc:
        # Re-raise the specific HTTPException we created above.
        raise http_exc
    except Exception as e:
        # Handle other potential errors during the check (e.g., Weaviate connection issue).
        print(f"Error during duplication check for file '{file.filename}' in session '{session_id}': {e}")
        raise HTTPException(status_code=500, detail="Error communicating with the database to check for duplicates.")
    # --- END OF DUPLICATION CHECK ---

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_f:
            content = await file.read()
            temp_f.write(content)
            temp_file_path = temp_f.name
        
        print(f"Processing uploaded PDF '{file.filename}' for session '{session_id}' from temp path '{temp_file_path}'")
        # A new UUID is still generated for each unique upload instance.
        doc_instance_uuid = str(uuid.uuid4())
        
        pages_content = pdf_to_page_texts(temp_file_path)

        if not pages_content:
            raise HTTPException(status_code=400, detail=f"No text could be extracted from any page of '{file.filename}'.")

        collection = weaviate_client_instance.collections.get(COLLECTION_NAME)
        total_chunks_added_for_doc = 0
        overall_chunk_idx_for_doc = 0
        
        print(f"Preparing to batch insert chunks for '{file.filename}' (doc_instance_uuid: {doc_instance_uuid}, session_id: {session_id})...")
        
        with collection.batch.dynamic() as batch_ctx:
            for page_data in pages_content:
                page_num = page_data["page_number"]
                page_text = page_data["page_text"]
                
                if not page_text.strip():
                    print(f"Skipping page {page_num} as it has no text content for '{file.filename}'.")
                    continue

                page_chunks = chunk_page_text(page_text, CHUNK_SIZE)
                
                if not page_chunks:
                    print(f"No chunks generated for page {page_num} of '{file.filename}'.")
                    continue
                
                print(f"Page {page_num}: processing {len(page_chunks)} chunks.")
                for page_chunk_idx, chunk_text_content in enumerate(page_chunks):
                    if not chunk_text_content.strip():
                        print(f"Skipping empty chunk at page {page_num}, page_chunk_idx {page_chunk_idx} for '{file.filename}'.")
                        continue
                    try:
                        vector = embedding_model.encode(chunk_text_content).tolist()
                        batch_ctx.add_object(
                            properties={
                                "session_id": session_id,
                                "document_uuid": doc_instance_uuid,
                                "filename": file.filename,
                                "text": chunk_text_content,
                                "chunk_index": overall_chunk_idx_for_doc,
                                "page_number": page_num,
                                "page_chunk_index": page_chunk_idx
                            },
                            vector=vector
                        )
                        total_chunks_added_for_doc += 1
                        overall_chunk_idx_for_doc += 1
                    except Exception as e_chunk:
                        print(f"Error processing or batching chunk (overall_idx {overall_chunk_idx_for_doc}, page {page_num}, page_chunk {page_chunk_idx}) of '{file.filename}' for session '{session_id}': {e_chunk}. Skipping.")
        
        if total_chunks_added_for_doc > 0:
            print(f"Successfully batched and sent {total_chunks_added_for_doc} chunks for '{file.filename}' (session '{session_id}') to Weaviate.")
        else:
            print(f"No chunks were successfully added to Weaviate for '{file.filename}' in session '{session_id}'.")
            if not pages_content:
                raise HTTPException(status_code=400, detail=f"PDF '{file.filename}' was processed but yielded no text content from any page.")

        return {
            "message": f"PDF '{file.filename}' processed for session '{session_id}'.",
            "session_id": session_id,
            "filename": file.filename,
            "document_instance_uuid": doc_instance_uuid,
            "total_pages_with_text": len(pages_content),
            "total_chunks_added_to_weaviate": total_chunks_added_for_doc
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error uploading PDF for session '{session_id}', file '{file.filename}': {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to upload PDF '{file.filename}': {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e_clean:
                print(f"Error cleaning up temporary file {temp_file_path}: {e_clean}")

@app.get("/query/{session_id}", summary="Query documents within a specific session using a question")
async def query_document_in_session(session_id: str, question: str):
    global weaviate_client_instance, embedding_model, groq_client
    if not weaviate_client_instance or not embedding_model or not groq_client:
        print("Error: One or more clients (Weaviate, Embedding, Groq) not initialized.")
        raise HTTPException(status_code=503, detail="Server components not ready. Please try again shortly.")

    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not SessionManager.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session ID '{session_id}' not found. Create session and upload docs first.")

    try:
        print(f"Received query for session '{session_id}': '{question}'")
        print("Embedding query...")
        query_vector = embedding_model.encode(question).tolist()

        collection = weaviate_client_instance.collections.get(COLLECTION_NAME)
        print(f"Querying Weaviate collection '{COLLECTION_NAME}' for session_id '{session_id}'...")
        
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=5, # You can adjust this for more or less context
            filters=Filter.by_property("session_id").equal(session_id),
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True),
            return_properties=[
                "text", "filename", "document_uuid", 
                "chunk_index", "page_number", "page_chunk_index", "session_id"
            ]
        )

        print(f"Retrieved {len(response.objects)} objects from Weaviate for session '{session_id}'.")
        if not response.objects:
            return {"answer": f"I could not find any relevant information in the documents for session '{session_id}' to answer your question.", "retrieved_context_summary": "No relevant chunks found for this session."}

        context_parts = []
        retrieved_filenames_pages = set() 

        for obj in response.objects:
            props = obj.properties
            filename = props.get('filename', 'N/A')
            page_num = props.get('page_number', 'N/A')
            text_content = props.get('text', '')
            
            # Each context part is clearly marked with its source
            context_parts.append(
                f"<source filename='{filename}' page='{page_num}'>\n{text_content}\n</source>"
            )
            retrieved_filenames_pages.add(f"{filename} (Page {page_num})")
            
            if obj.metadata:
                distance_val = obj.metadata.distance
                distance_str = f"{distance_val:.4f}" if distance_val is not None else "N/A"
                print(f"  - Source: {filename}, Page: {page_num}, Distance: {distance_str}")
        
        context = "\n\n".join(context_parts)
        
        print("Sending context and question to Groq for completion...")
        
        # --- THIS IS THE CRITICAL MODIFICATION ---
        # The new, stricter system prompt
        system_prompt = f"""
You are a highly specialized AI assistant. Your **only** function is to answer the user's question based **exclusively** on the provided context.

**CRITICAL RULES:**
1.  **DO NOT** use any external knowledge, assumptions, or information outside of the text provided in the `<source>` tags.
2.  Your answer **MUST** be derived directly from the content within the provided `<source>` tags.
3.  If the answer is not found in the context, you **MUST** state: "Based on the provided documents, I could not find an answer to this question." Do not try to guess or infer an answer.
4.  When you provide an answer, cite the source filename and page number. For example: "According to `filename.pdf` on page `X`, the value is...".
5.  Synthesize information from multiple sources if they are relevant, but always attribute the information to the correct source file and page.
6.  Do not add any conversational fluff or apologies like "I'm sorry" or "I'm just an AI". Be direct and factual.

**CONTEXT FROM DOCUMENTS:**
{context}
"""

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            model="llama3-70b-8192" 
        )

        answer = chat_completion.choices[0].message.content
        print(f"Groq answer for session '{session_id}': {answer[:300]}...")
        
        summary_sources_list = list(retrieved_filenames_pages)
        if not summary_sources_list:
            summary_display = "N/A"
        else:
            summary_display = ", ".join(summary_sources_list[:3]) 
            if len(summary_sources_list) > 3:
                summary_display += f" and {len(summary_sources_list)-3} other source pages"
        
        return {"answer": answer, "retrieved_context_summary": f"{len(response.objects)} chunks from sources like '{summary_display}' within session '{session_id}'."}

    except weaviate.exceptions.WeaviateQueryException as wqe:
        print(f"Weaviate query error for session '{session_id}': {wqe}")
        raise HTTPException(status_code=503, detail=f"Could not query Weaviate for session '{session_id}': {str(wqe)}")
    except Exception as e:
        print(f"Error during query endpoint call for session '{session_id}': {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error during query for session '{session_id}': {str(e)}")
    
@app.delete("/documents", summary="Delete all document chunks by session_id and filename")
async def delete_documents_by_session_and_filename(session_id: str, filename: str):
    global weaviate_client_instance
    if not weaviate_client_instance:
        print("Error: Weaviate client not initialized during delete request.")
        raise HTTPException(status_code=503, detail="Weaviate client not initialized.")

    if not session_id or not filename:
        raise HTTPException(status_code=400, detail="Both 'session_id' and 'filename' query parameters are required.")

    print(f"Received request to delete documents with session_id='{session_id}' and filename='{filename}'")

    try:
        collection = weaviate_client_instance.collections.get(COLLECTION_NAME)
        
        where_filter = Filter.all_of([
            Filter.by_property("session_id").equal(session_id),
            Filter.by_property("filename").equal(filename)
        ])

        # First, fetch the UUIDs of the objects to be deleted to provide more details if needed
        # This step is optional but can be good for logging or more detailed feedback.
        # For large deletions, this might add overhead.
        # query_response = collection.query.fetch_objects(filters=where_filter, limit=10000) # Adjust limit as needed
        # print(f"Found {len(query_response.objects)} objects matching criteria for deletion.")

        delete_result = collection.data.delete_many(where=where_filter)

        individual_errors = []
        # The structure of delete_result.objects might vary or not exist if all failed at a higher level.
        # Weaviate Python client v4: `delete_result` is a `DeleteManyObject`
        # It has `failed`, `matches`, `successful` attributes.
        # If `failed > 0`, `delete_result.errors` (list of `ErrorObject`) might contain details.

        if delete_result.failed > 0 and hasattr(delete_result, 'errors'):
             for err_obj in delete_result.errors: # Iterate through potential error objects
                individual_errors.append({
                    "message": err_obj.message, # General error message for this failure
                    # err_obj might have more details depending on the error, like affected_object_ids
                })
        elif delete_result.failed > 0:
             individual_errors.append({"message": "Generic failure indicated, but detailed error objects not found in expected 'errors' attribute."})


        response_payload = {
            "message": "Deletion process completed.",
            "session_id_filter": session_id,
            "filename_filter": filename,
            "objects_matched": delete_result.matches,
            "successful_deletions": delete_result.successful,
            "failed_deletions": delete_result.failed,
            "detailed_errors": individual_errors 
        }

        if delete_result.failed > 0:
            print(f"Warning: Deletion operation had {delete_result.failed} failures for session_id='{session_id}', filename='{filename}'. Errors: {individual_errors}")
        
        if delete_result.matches == 0:
            response_payload["message"] = "No documents found matching the specified criteria. Nothing was deleted."
            print(f"No documents matched for deletion with session_id='{session_id}', filename='{filename}'.")
        elif delete_result.successful > 0 :
            print(f"Deletion successful for session_id='{session_id}', filename='{filename}'. Matched: {delete_result.matches}, Deleted: {delete_result.successful}")
            
        return response_payload

    except weaviate.exceptions.WeaviateQueryException as wqe: # Covers connection issues or query syntax errors
        print(f"Weaviate query/connection error during deletion: {wqe}")
        raise HTTPException(status_code=503, detail=f"A Weaviate error occurred during deletion: {str(wqe)}")
    except Exception as e: # Catch-all for other unexpected errors
        print(f"An unexpected error occurred during deletion for session_id='{session_id}', filename='{filename}': {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server...")
    # Ensure you have a .env file in the same directory as this script
    # with WEAVIATE_URL, WEAVIATE_API_KEY, and GROQ_API_KEY
    uvicorn.run("main:app", host="127.127.0.1", port=5009, reload=True)