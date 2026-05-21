import streamlit as st
import os
import time
import requests  # Usamos la librería estándar de peticiones web
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, PromptTemplate
from llama_index.core.settings import Settings
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

# --- 2. CONFIGURACIÓN DEL MOTOR DE EMBEDDING (Para buscar en los apuntes) ---
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None  # Desactivamos el LLM de LlamaIndex para evitar conflictos

# --- 3. DEFINICIÓN DEL PROMPT ---
PROMPT_SISTEMA = (
    "Eres un profesor experto en Química del Bachillerato Internacional (BI).\n"
    "Tu objetivo es ayudar al alumno de forma pedagógica, clara y motivadora.\n\n"
    "REGLAS:\n"
    "1. Usa EXCLUSIVAMENTE los apuntes proporcionados en el contexto para responder.\n"
    "2. Al citar la fuente, indica siempre el TEMA y la PÁGINA en castellano.\n"
    "   FORMATO: '(Fuente: Tema X - [Nombre], Página Y)'.\n"
    "3. Si la información no está en los apuntes, admítelo con amabilidad.\n"
    "4. Estructura la respuesta con negritas y listas para facilitar la lectura.\n"
    "5. Termina siempre con una pregunta de seguimiento para el alumno.\n"
)

# --- 4. FUNCIÓN PARA CARGAR EL RETRIEVER DEL RAG ---
@st.cache_resource(show_spinner="Analizando apuntes de Química...")
def cargar_retriever():
    if not os.path.exists(STORAGE_DIR):
        st.error("Error: No encuentro la carpeta 'storage'. Verifica que esté subida a GitHub.")
        return None
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    indice = load_index_from_storage(storage_context)
    return indice.as_retriever(similarity_top_k=2)

# --- 5. INTERFAZ VISUAL ---
st.set_page_config(page_title="Asistente de Química BI", page_icon="🧪")
st.title("🧪 Profesor-Asistente de Química (JuandiBot)")

retriever = cargar_retriever()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy JuandiBot. He analizado tus apuntes de Química. ¿Qué tema quieres que exploremos hoy?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- 6. LÓGICA DE CONSULTA POR PETICIÓN DIRECTA (Inmune a bugs) ---
if prompt := st.chat_input("Escribe tu duda..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not clave_gemini:
        st.error("⚠️ Error: No se ha detectado la clave API en los Secrets.")
        st.stop()

    if retriever:
        with st.chat_message("assistant"):
            try:
                # 1. Recuperamos los fragmentos de tus apuntes en GitHub
                nodos_recuperados = retriever.retrieve(prompt)
                contexto_apuntes = "\n\n".join([nodo.get_content() for nodo in nodos_recuperados])
                
                # 2. Construimos el mensaje final combinando el rol y tus apuntes
                texto_instrucciones = f"{PROMPT_SISTEMA}\n\nCONTEXTO DE LOS APUNTES:\n{contexto_apuntes}"
                
                # 3. Llamada directa vía REST API a la versión estable de Google
                url_api = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash-latest:generateContent?key={clave_gemini}"
                
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [{
                        "parts": [
                            {"text": f"{texto_instrucciones}\n\nPREGUNTA DEL ALUMNO: {prompt}\n\nRESPUESTA DEL PROFESOR:"}
                        ]
                    }]
                }
                
                response = requests.post(url_api, json=payload, headers=headers)
                datos_respuesta = response.json()
                
                # 4. Procesamos la respuesta del JSON de Google
                if response.status_code == 200:
                    texto_final = datos_respuesta['candidates'][0]['content']['parts'][0]['text']
                    
                    # Efecto de escritura lenta en la interfaz
                    def generador_lento():
                        for palabra in texto_final.split(" "):
                            yield palabra + " "
                            time.sleep(0.02)

                    full_response = st.write_stream(generador_lento())
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    mensaje_error = datos_respuesta.get('error', {}).get('message', 'Error desconocido')
                    st.error(f"⚠️ Error de la API de Google ({response.status_code}): {mensaje_error}")
                
            except Exception as e:
                st.error(f"Error de procesamiento: {str(e)}")
