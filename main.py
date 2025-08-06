import os
import streamlit as st
import pickle
import glob
import requests
import logging
from typing import List, Dict, Any

# --- Logging Setup ---
LOG_FILE = "assistant_app.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def log_and_print(msg):
    logging.info(msg)
    st.session_state.setdefault("log_msgs", []).append(msg)

def show_log():
    if "log_msgs" in st.session_state:
        with st.expander("Show Log"):
            for msg in st.session_state["log_msgs"]:
                st.text(msg)

# --- Constants ---
PROJECTS_DIR = os.path.abspath("projects")
RAW_DIR = "raw"
EMBEDDINGS_FILE = "embeddings.pkl"
CHUNK_SIZE = 20
CHUNK_OVERLAP = 5
TOP_K = 5

# --- Embedding Model ---
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDING_MODEL_AVAILABLE = True
    log_and_print("sentence-transformers loaded successfully.")
except Exception as e:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDING_MODEL_AVAILABLE = False
    log_and_print(f"sentence-transformers not available: {e}")

def ensure_project_dirs(project_name: str):
    raw_path = os.path.join(PROJECTS_DIR, project_name, RAW_DIR)
    os.makedirs(raw_path, exist_ok=True)
    log_and_print(f"Ensured directories for project: {project_name}")
    return raw_path

def list_projects() -> List[str]:
    if not os.path.exists(PROJECTS_DIR):
        return []
    return [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]

def save_uploaded_files(project: str, uploaded_files: List[Any]):
    raw_path = ensure_project_dirs(project)
    for file in uploaded_files:
        file_path = os.path.join(raw_path, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        log_and_print(f"Saved uploaded file: {file_path}")

def get_project_files(project: str) -> List[str]:
    raw_path = os.path.join(PROJECTS_DIR, project, RAW_DIR)
    return glob.glob(os.path.join(raw_path, "*"))

def chunk_code(code: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    lines = code.splitlines()
    chunks = []
    i = 0
    while i < len(lines):
        chunk_lines = lines[i:i+chunk_size]
        chunk_text = "\n".join(chunk_lines)
        chunks.append({
            "text": chunk_text,
            "start_line": i+1,
            "end_line": i+len(chunk_lines)
        })
        if i + chunk_size >= len(lines):
            break
        i += chunk_size - overlap
    return chunks

def load_embedding_model():
    if EMBEDDING_MODEL_AVAILABLE:
        try:
            model = SentenceTransformer('./models/all-MiniLM-L6-v2')
            _ = model.encode(["warmup"], convert_to_tensor=True, show_progress_bar=False)
            log_and_print("Loaded sentence-transformers embedding model.")
            return model, None
        except Exception as e:
            log_and_print(f"Could not load embedding model: {e}. Falling back to TF-IDF.")
    log_and_print("Using TF-IDF vectorizer for embeddings.")
    return None, TfidfVectorizer()

def compute_embeddings(chunks: List[str], model, tfidf_vectorizer=None):
    if model:
        log_and_print("Computing SBERT embeddings for code chunks.")
        return model.encode(chunks)
    else:
        log_and_print("Computing TF-IDF embeddings for code chunks.")
        tfidf_matrix = tfidf_vectorizer.fit_transform(chunks)
        return tfidf_matrix

def save_embeddings(project: str, data: Dict):
    emb_path = os.path.join(PROJECTS_DIR, project, EMBEDDINGS_FILE)
    with open(emb_path, "wb") as f:
        pickle.dump(data, f)
    log_and_print(f"Saved embeddings to {emb_path}")

def load_embeddings(project: str) -> Dict:
    emb_path = os.path.join(PROJECTS_DIR, project, EMBEDDINGS_FILE)
    if os.path.exists(emb_path):
        with open(emb_path, "rb") as f:
            log_and_print(f"Loaded embeddings from {emb_path}")
            return pickle.load(f)
    return {}

def files_changed(project: str, file_list: List[str]) -> bool:
    emb_data = load_embeddings(project)
    prev_files = emb_data.get("files", [])
    prev_mtimes = emb_data.get("mtimes", {})
    for f in file_list:
        if f not in prev_files or os.path.getmtime(f) != prev_mtimes.get(f, 0):
            log_and_print(f"File changed or new: {f}")
            return True
    return False

def get_code_chunks_and_metadata(file_list: List[str]) -> (List[str], List[Dict]):
    all_chunks = []
    metadata = []
    for file_path in file_list:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        chunks = chunk_code(code)
        for chunk in chunks:
            all_chunks.append(chunk["text"])
            metadata.append({
                "file": os.path.basename(file_path),
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "text": chunk["text"],
                "file_path": file_path
            })
    log_and_print(f"Chunked {len(file_list)} files into {len(all_chunks)} code chunks.")
    return all_chunks, metadata

def get_top_chunks(question: str, chunks: List[str], embeddings, model, tfidf_vectorizer, top_k=TOP_K):
    if model:
        q_emb = model.encode([question])
        sims = cosine_similarity(q_emb, embeddings)[0]
    else:
        q_vec = tfidf_vectorizer.transform([question])
        sims = cosine_similarity(q_vec, embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    log_and_print(f"Selected top {top_k} code chunks for the query.")
    return top_indices

def build_prompt(selected_chunks: List[Dict], question: str) -> str:
    prompt = (
        "You are a helpful AI code assistant. Based on the following code, answer the user's question:\n\n"
    )
    for chunk in selected_chunks:
        prompt += f"File: {chunk['file']} (lines {chunk['start_line']}-{chunk['end_line']}):\n{chunk['text']}\n\n"
    prompt += f"Question: {question}\nAnswer:\n"
    log_and_print("Built prompt for LLM.")
    return prompt

import json

def query_ollama(prompt: str, model: str = "codellama"):
    log_and_print("Sending prompt to Ollama LLM.")
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=120,
            stream=True  # Enable streaming response
        )
        response.raise_for_status()
        # Try to handle streaming JSONL (one JSON per line)
        content = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
                # Ollama streaming: each line has a "message" with "content"
                if "message" in data and "content" in data["message"]:
                    content += data["message"]["content"]
                # Some models use "response"
                elif "response" in data:
                    content += data["response"]
                elif "content" in data:
                    content += data["content"]
            except Exception as e:
                log_and_print(f"Error parsing Ollama stream line: {e} | Line: {line}")
        if content.strip():
            log_and_print("Received answer from Ollama (streaming).")
            return content
        else:
            # Fallback: try to parse as a single JSON object
            try:
                data = response.json()
                if "message" in data and "content" in data["message"]:
                    return data["message"]["content"]
                elif "response" in data:
                    return data["response"]
                elif "content" in data:
                    return data["content"]
                else:
                    return str(data)
            except Exception as e:
                log_and_print(f"Error parsing Ollama fallback JSON: {e}")
                return f"Error parsing Ollama response: {e}\nRaw response: {response.text}"
    except Exception as e:
        log_and_print(f"Error communicating with Ollama: {e}")
        return f"Error communicating with Ollama: {e}"


# --- Streamlit UI ---

st.set_page_config(page_title="Offline AI Code Assistant", layout="wide")
st.title("🧑‍💻 Offline AI Coding Assistant")

# Project selection/creation
st.sidebar.header("Project Management")
projects = list_projects()
project = st.sidebar.selectbox("Select Project", [""] + projects)
new_project = st.sidebar.text_input("Or create new project", "")

if new_project:
    project = new_project.strip()
    if project and project not in projects:
        ensure_project_dirs(project)
        st.sidebar.success(f"Project '{project}' created.")
        log_and_print(f"Created new project: {project}")

if not project:
    st.info("Please select or create a project to continue.")
    show_log()
    st.stop()

# File upload
st.subheader(f"Project: {project}")
uploaded_files = st.file_uploader(
    "Upload code files (Python, Java, C++, etc.)",
    type=["py", "java", "cpp", "c", "js", "ts", "go", "rb", "rs", "cs", "php", "swift", "kt", "scala", "m", "h", "hpp", "html", "css", "json", "xml", "sh", "bat", "pl", "r", "jl", "sql", "md", "txt"],
    accept_multiple_files=True,
    key="file_uploader"
)
if uploaded_files:
    save_uploaded_files(project, uploaded_files)
    st.success(f"Uploaded {len(uploaded_files)} files.")
    log_and_print(f"Uploaded {len(uploaded_files)} files to project {project}.")

# List files in project
file_list = get_project_files(project)
if file_list:
    st.markdown("**Files in project:**")
    for f in file_list:
        st.code(os.path.basename(f))
else:
    st.info("No files uploaded yet.")

# Embedding and caching
model, tfidf_vectorizer = load_embedding_model()
need_embedding = files_changed(project, file_list) or not load_embeddings(project)
if need_embedding and file_list:
    st.info("Generating embeddings for code files. This may take a moment...")
    code_chunks, metadata = get_code_chunks_and_metadata(file_list)
    embeddings = compute_embeddings(code_chunks, model, tfidf_vectorizer)
    emb_data = {
        "files": file_list,
        "mtimes": {f: os.path.getmtime(f) for f in file_list},
        "chunks": code_chunks,
        "metadata": metadata,
        "embeddings": embeddings,
        "model_type": "sbert" if model else "tfidf"
    }
    save_embeddings(project, emb_data)
    st.success("Embeddings updated and cached.")
else:
    emb_data = load_embeddings(project)
    code_chunks = emb_data.get("chunks", [])
    metadata = emb_data.get("metadata", [])
    embeddings = emb_data.get("embeddings", None)
    if emb_data.get("model_type") == "sbert":
        model, tfidf_vectorizer = load_embedding_model()
    else:
        model, tfidf_vectorizer = None, load_embedding_model()[1]

# Question answering
st.subheader("Ask a question about your code")
question = st.text_area("Enter your question", height=80)
if st.button("Get Answer", disabled=not code_chunks or not question.strip()):
    if not code_chunks:
        st.warning("No code files found for this project.")
        log_and_print("No code files found for this project.")
    else:
        top_indices = get_top_chunks(
            question, code_chunks, embeddings, model, tfidf_vectorizer, top_k=TOP_K
        )
        selected_chunks = [metadata[i] for i in top_indices]
        prompt = build_prompt(selected_chunks, question)
        with st.spinner("Querying Ollama LLM..."):
            answer = query_ollama(prompt)
        st.markdown("**Top Relevant Code Chunks:**")
        for chunk in selected_chunks:
            with st.expander(f"{chunk['file']} (lines {chunk['start_line']}-{chunk['end_line']})"):
                st.code(chunk['text'])
        st.markdown("**Answer:**")
        st.write(answer)
        st.markdown("---")
        st.markdown("**Prompt sent to LLM:**")
        st.code(prompt)
        log_and_print("Displayed answer and prompt to user.")

show_log()


