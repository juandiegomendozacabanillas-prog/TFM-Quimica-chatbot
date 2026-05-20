import streamlit as st
import os
import time
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, PromptTemplate
from llama_index.core.settings import Settings
# ATENCIÓN: Cambiamos al conector estándar de OpenAI que no añade prefijos fantasma
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- 1. CARGA DE CONFIGURACIÓN ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()
STORAGE_DIR = "./storage"

clave_bruta = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if clave_bruta:
    clave_gemini = clave_bruta.strip()
else:
    clave_gemini = None

# --- 2. CONFIGURACIÓN DE LOS "MOTORES" DE IA (Embedding y LLM) ---
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

if clave_gemini:
    # Usamos la clase OpenAI estricta apuntando al endpoint compatible de Gemini.
    # Al ser la clase genérica, enviará el nombre del modelo intacto sin añadir "models/".
    Settings.llm = OpenAI(
        model="gemini-1.5-flash",
        api_key=clave_gemini,
        api_base="https://generativelanguage.googleapis.com/v1beta/openai",
        temperature=0.7
    )
else:
    st.error("⚠️ Error: No se ha detectado la clave API (GEMINI_API_KEY) en los Secrets.")
    st.stop()

# --- 3. DEFINICIÓN DE LA PERSONALIDAD (Prompt) ---
# ... (El resto de tu código del prompt y la interfaz de abajo se queda EXACTAMENTE igual)
