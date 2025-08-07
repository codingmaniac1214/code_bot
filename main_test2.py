import os
import streamlit as st
import pickle
import glob
import requests
import logging
from typing import List, Dict, Any
import fnmatch
import re
import subprocess
import platform


# Function to style code terms (backtick-wrapped) with color and background
CODE_TERM_STYLE = "color: #d63384; background: #f3e6f9; font-family: monospace; padding: 2px 4px; border-radius: 4px;"
def style_code_terms(text):
   return re.sub(
       r'`([^`]+)`',
       rf'<span style="{CODE_TERM_STYLE}">\1</span>',
       text
   )

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
    matches = []
    for root, _, filenames in os.walk(raw_path):
        for ext in ('.js', '.java', '.cc'):
            for filename in filenames:
                if filename.lower().endswith(ext):
                    matches.append(os.path.join(root, filename))
    return matches




# --- Recursive Folder Browsing ---
def get_files_recursively(folder_path: str, extensions=(".js", ".java", ".cc")) -> List[str]:
   matches = []
   for root, dirnames, filenames in os.walk(folder_path):
       for ext in extensions:
           for filename in fnmatch.filter(filenames, f"*{ext}"):
               matches.append(os.path.join(root, filename))
   return matches


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


def get_code_chunks_and_metadata(file_list: List[str], chunk_size: int = 20, overlap: int = 5) -> (List[str], List[Dict]):
   all_chunks = []
   metadata = []
   for file_path in file_list:
       if not os.path.isfile(file_path):
           continue  # Skip directories or non-files
       with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
           code = f.read()
       chunks = chunk_code(code, chunk_size=chunk_size, overlap=overlap)
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


import os
import streamlit as st
import pickle
import glob
import requests
import logging
from typing import List, Dict, Any
import fnmatch
import re
import subprocess
import platform
import zipfile
import io

# --- Recursive Folder Upload via Zip ---
ALLOWED_EXTENSIONS = ('.js', '.java', '.cc')
st.sidebar.markdown("---")
st.sidebar.subheader("Add Folder with Subfolders")
zip_file = st.sidebar.file_uploader(
    "Upload a .zip file containing your folder (includes subfolders and .js, .java, .cc files):",
    type=["zip"],
    accept_multiple_files=False,
    key="zip_folder_uploader"
)

if zip_file:
    raw_path = ensure_project_dirs(project)
    files_added = 0
    try:
        # Create a temporary directory to extract the zip
        temp_zip_path = os.path.join(PROJECTS_DIR, project, "temp_zip")
        os.makedirs(temp_zip_path, exist_ok=True)
        
        # Save the uploaded zip file
        zip_path = os.path.join(temp_zip_path, zip_file.name)
        with open(zip_path, "wb") as f:
            f.write(zip_file.getbuffer())
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_path)
        
        # Get all extracted files recursively
        extracted_files = get_files_recursively(raw_path, extensions=ALLOWED_EXTENSIONS)
        if extracted_files:
            files_added = len(extracted_files)
            for file_path in extracted_files:
                log_and_print(f"Added {file_path} to project {project}.")
            st.sidebar.success(f"Added {files_added} .js, .java, or .cc files from the uploaded zip.")
        else:
            st.sidebar.warning("No .js, .java, or .cc files found in the uploaded zip.")
        
        # Clean up temporary zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)
            log_and_print(f"Cleaned up temporary zip file: {zip_path}")
        
    except Exception as e:
        st.sidebar.error(f"Error processing zip file: {e}")
        log_and_print(f"Error processing zip file: {e}")
    finally:
        # Clean up temporary directory if empty
        if os.path.exists(temp_zip_path) and not os.listdir(temp_zip_path):
            os.rmdir(temp_zip_path)
            log_and_print(f"Removed empty temporary directory: {temp_zip_path}")

# --- Sidebar File Tree with Nested Expanders (only .js, .java, .cc files) ---


def show_file_tree(base_path, rel_path=""):
   abs_path = os.path.join(base_path, rel_path)
   entries = sorted(os.listdir(abs_path))
   dirs = [e for e in entries if os.path.isdir(os.path.join(abs_path, e))]
   # Only show files with allowed extensions
   files = [e for e in entries if os.path.isfile(os.path.join(abs_path, e)) and e.lower().endswith(ALLOWED_EXTENSIONS)]
   # Only show folders that contain allowed files or subfolders with allowed files
   for d in dirs:
       subfolder_path = os.path.join(abs_path, d)
       # Recursively check if subfolder contains allowed files
       has_allowed = False
       for root, _, filenames in os.walk(subfolder_path):
           if any(f.lower().endswith(ALLOWED_EXTENSIONS) for f in filenames):
               has_allowed = True
               break
       if has_allowed:
           with st.expander(f"{d}/", expanded=False):
               show_file_tree(base_path, os.path.join(rel_path, d))
   for f in files:
       file_abs_path = os.path.join(abs_path, f)
       file_key = f"filebtn_{rel_path}_{f}"
       if st.button(f, key=file_key):
           try:
               if platform.system() == "Windows":
                   subprocess.Popen(["notepad", file_abs_path])
               elif platform.system() == "Darwin":
                   subprocess.Popen(["open", file_abs_path])
               else:
                   subprocess.Popen(["xdg-open", file_abs_path])
           except Exception as e:
               st.sidebar.error(f"Failed to open file: {e}")


# Replace the sidebar file list expander with the file tree
file_list = get_project_files(project)
raw_path = os.path.join(PROJECTS_DIR, project, RAW_DIR)
with st.sidebar.expander("Show Project Files", expanded=False):
   if os.path.exists(raw_path) and os.listdir(raw_path):
       show_file_tree(raw_path)
   else:
       st.info("No files uploaded yet.")


# --- Sidebar Controls for Retrieval ---
st.sidebar.markdown("---")
st.sidebar.subheader("Retrieval Settings")
num_top_chunks = st.sidebar.slider("Number of Top Chunks", min_value=1, max_value=30, value=10)
chunk_size = 20  # Fixed value
overlap_size = 5  # Fixed value


# Embedding and caching
model, tfidf_vectorizer = load_embedding_model()
need_embedding = files_changed(project, file_list) or not load_embeddings(project)


if need_embedding and file_list:
   st.info("Generating embeddings for code files. This may take a moment...")
   code_chunks, metadata = get_code_chunks_and_metadata(file_list, chunk_size=chunk_size, overlap=overlap_size)
   embeddings = compute_embeddings(code_chunks, model, tfidf_vectorizer)
   emb_data = {
       "files": file_list,
       "mtimes": {f: os.path.getmtime(f) for f in file_list},
       "chunks": code_chunks,
       "metadata": metadata,
       "embeddings": embeddings,
       "model_type": "sbert" if model else "tfidf",
       "chunk_size": chunk_size,
       "overlap_size": overlap_size
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


# --- Chat-like Query System with Context Retention ---
st.subheader("Ask a question about your code (Chat Mode)")
if "chat_history" not in st.session_state:
   st.session_state["chat_history"] = []


user_input = st.text_area("Enter your question or follow-up", height=80, key="chat_input")
show_latest = False
latest_turn = None
if st.button("Send", disabled=not code_chunks or not user_input.strip()):
   # Build context from chat history
   chat_context = ""
   for turn in st.session_state["chat_history"][-5:]:  # last 5 turns
       chat_context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
   chat_context += f"User: {user_input}\nAssistant:"
   # Use top chunks for the latest question, as per slider
   top_indices = get_top_chunks(
       user_input, code_chunks, embeddings, model, tfidf_vectorizer, top_k=num_top_chunks
   )
   selected_chunks = [metadata[i] for i in top_indices[:num_top_chunks]]
   prompt = build_prompt(selected_chunks, chat_context)
   with st.spinner("Querying Ollama LLM..."):
       answer = query_ollama(prompt)
   latest_turn = {"user": user_input, "assistant": answer, "chunks": selected_chunks, "num_chunks": num_top_chunks}
   st.session_state["chat_history"].append(latest_turn)
   show_latest = True


# Show the latest answer above chat history if available
if (show_latest or (st.session_state.get("chat_history") and not user_input.strip())) and st.session_state["chat_history"]:
   latest_turn = st.session_state["chat_history"][-1]
   st.markdown("**Latest Answer:**")
   st.markdown(f"**You:** {latest_turn['user']}")
   styled_answer = style_code_terms(latest_turn['assistant'])
   with st.container():
       st.markdown(
           f"""
           <div style='background-color:#f6f8fa; border-radius:8px; padding:16px; margin-bottom:8px;'>
               <strong>Assistant:</strong><br>
               <span style='font-size: 1.05em;'>{styled_answer.replace(chr(10), '<br>')}</span>
           </div>
           """,
           unsafe_allow_html=True
       )
   if "chunks" in latest_turn and latest_turn["chunks"]:
       with st.expander(f"Show {latest_turn.get('num_chunks', len(latest_turn['chunks']))} Chunks Used for This Answer"):
           for chunk in latest_turn["chunks"][:latest_turn.get('num_chunks', len(latest_turn['chunks']))]:
               st.markdown(f"**{chunk['file']} (lines {chunk['start_line']}-{chunk['end_line']}):**")
               st.code(chunk['text'])
   st.markdown("---")


# Display chat history in reverse order (most recent at the top)
if st.session_state["chat_history"]:
   st.markdown("**Chat History:**")
   for i, turn in enumerate(reversed(st.session_state["chat_history"])):
       st.markdown(f"**You:** {turn['user']}")
       styled_answer = style_code_terms(turn['assistant'])
       with st.container():
           st.markdown(
               f"""
               <div style='background-color:#f6f8fa; border-radius:8px; padding:16px; margin-bottom:8px;'>
                   <strong>Assistant:</strong><br>
                   <span style='font-size: 1.05em;'>{styled_answer.replace(chr(10), '<br>')}</span>
               </div>
               """,
               unsafe_allow_html=True
           )
       if "chunks" in turn and turn["chunks"]:
           with st.expander(f"Show {turn.get('num_chunks', len(turn['chunks']))} Chunks Used for This Answer"):
               for chunk in turn["chunks"][:turn.get('num_chunks', len(turn['chunks']))]:
                   st.markdown(f"**{chunk['file']} (lines {chunk['start_line']}-{chunk['end_line']}):**")
                   st.code(chunk['text'])
       st.markdown("---")


show_log()







