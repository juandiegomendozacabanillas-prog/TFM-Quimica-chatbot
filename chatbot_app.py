import streamlit as st
import os
import time
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, PromptTemplate
from llama_index.core.settings import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- 1. CARGA DE CONFIGURACIÓN ---
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Desactiva el aviso de paralelismo
load_dotenv()
STORAGE_DIR = "./storage"

# Prioridad 1: Secrets de Streamlit (Nube) | Prioridad 2: Archivo .env (Local)
clave_gemini = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

# Sincroniza la clave con el entorno global del servidor de forma segura
if clave_gemini:
    os.environ["GEMINI_API_KEY"] = clave_gemini

# --- 2. CONFIGURACIÓN DE LOS "MOTORES" DE IA (Embedding y LLM) ---
# Búsqueda local con HuggingFace (Vectores)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Bloque de Diagnóstico Seguro para comprobar la API Key en la pantalla de Streamlit
if clave_gemini:
    clave_limpia = clave_gemini.strip()
    # Mostramos datos de control en la app para ver qué está leyendo el servidor
    st.info(f"🔍 [INFO DE CONTROL] Longitud de la clave detectada: {len(clave_limpia)} caracteres.")
    st.info(f"🔍 [INFO DE CONTROL] La clave empieza por: '{clave_limpia[:6]}' y termina por: '{clave_limpia[-4:]}'")
    
    import google.generativeai as genai
    genai.configure(api_key=clave_limpia)
    os.environ["GEMINI_API_KEY"] = clave_limpia
    
    Settings.llm = Gemini(
        model="models/gemini-1.5-flash",
        api_key=clave_limpia
    )
else:
    st.error("⚠️ Error: No se ha detectado la clave API (GEMINI_API_KEY) en los Secrets.")
    st.stop()

# --- 3. DEFINICIÓN DE LA PERSONALIDAD (Prompt) ---
template = (
    "Eres un profesor experto en Química del Bachillerato Internacional (BI).\n"
    "Tu objetivo es ayudar al alumno de forma pedagógica, clara y motivadora.\n\n"
    "REGLAS:\n"
    "1. Usa EXCLUSIVAMENTE los apuntes proporcionados para responder.\n"
    "2. Al citar la fuente, indica siempre el TEMA y la PÁGINA en castellano.\n"
    "   FORMATO: '(Fuente: Tema X - [Nombre], Página Y)'.\n"
    "3. Si la información no está en los apuntes, admítelo con amabilidad.\n"
    "4. Estructura la respuesta con negritas y listas para facilitar la lectura.\n"
    "5. Termina siempre con una pregunta de seguimiento para el alumno.\n\n"
    "CONTEXTO DE APUNTES:\n{context_str}\n\n"
    "PREGUNTA DEL ALUMNO: {query_str}\n\n"
    "RESPUESTA DEL PROFESOR:"
)
qa_prompt_tmpl = PromptTemplate(template)

# --- 4. FUNCIÓN PARA CARGAR EL ÍNDICE GUARDADO ---
@st.cache_resource(show_spinner="Analizando apuntes de Química...")
def cargar_asistente():
    if not os.path.exists(STORAGE_DIR):
        st.error("Error: No encuentro la carpeta 'storage'. Verifica que esté subida a GitHub.")
        return None
    
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    indice = load_index_from_storage(storage_context)
    
    engine = indice.as_query_engine(
        text_qa_template=qa_prompt_tmpl,
        similarity_top_k=2,
        streaming=False
    )
    return engine

# --- 5. INTERFAZ VISUAL ---
st.set_page_config(page_title="Asistente de Química BI", page_icon="🧪")
st.title("🧪 Profesor-Asistente de Química (JuandiBot)")

query_engine = cargar_asistente()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy JuandiBot. He analizado tus apuntes de Química. ¿Qué tema quieres que exploremos hoy?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- 6. LÓGICA DE CONSULTA ---
if prompt := st.chat_input("Escribe tu duda..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if query_engine:
        with st.chat_message("assistant"):
            try:
                response = query_engine.query(prompt)
                texto_final = response.response
                
                def generador_lento():
                    for palabra in texto_final.split(" "):
                        yield palabra + " "
                        time.sleep(0.02)

                full_response = st.write_stream(generador_lento())
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    st.error("⚠️ Cuota diaria de la API agotada (Máximo 20 interacciones).")
                elif "index out of range" in error_msg.lower():
                    st.error("⚠️ El asistente no encontró suficiente información en los apuntes para responder a eso.")
                else:
                    st.error(f"Error de conexión: {error_msg}")
