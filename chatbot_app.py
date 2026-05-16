import streamlit as st
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Desactiva el aviso de paralelismo
import time
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, PromptTemplate
from llama_index.core.settings import Settings
from llama_index.llms.gemini import Gemini 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # Para velocidad local

# --- 1. CARGA DE CONFIGURACIÓN ---
load_dotenv()
STORAGE_DIR = "./storage"
## clave_gemini = os.getenv("GEMINI_API_KEY")

# Prioridad 1: Secrets de Streamlit (Nube) | Prioridad 2: Archivo .env (Local)
clave_gemini = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not clave_gemini:
    st.error("⚠️ Error: No se ha configurado la clave API de Gemini.")
    st.stop()

# --- 2. CONFIGURACIÓN DE LOS "MOTORES" DE IA --- MOTOR DE BÚSQUEDA (Embedding)
# Búsqueda local con HuggingFace (Gratuito y rápido)
# Es un modelo de HuggingFace que se descarga en el Mac. 
# Convierte los apuntes en vectores (números) para poder buscar en ellos.
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Motor de respuesta Gemini 2.5 Flash (Estable)
if clave_gemini:
    Settings.llm = Gemini(
        model="gemini-1.5-flash", 
        api_key=clave_gemini
        # transport="rest" 
    )
else:
st.error("⚠️ La clave API (GEMINI_API_KEY) está vacía o no se lee correctamente.")
st.stop()

# --- 3. DEFINICIÓN DE LA PERSONALIDAD (Prompt) , Aquñi definimos cómo se comporta el profesor---
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
@st.cache_resource(show_spinner=True)
def cargar_asistente():
    if not os.path.exists(STORAGE_DIR):
        st.error("Error: No encuentro la carpeta 'storage'. Ejecuta 'cargar_material.py' primero.")
        return None
    
    # Cargamos el índice que creamos previamente
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    indice = load_index_from_storage(storage_context)
    
    # Creamos el motor de consulta con nuestra plantilla de profesor
    # Usamos similarity_top_k=2 para no saturar la cuota de la API (evitar error 429)
    engine = indice.as_query_engine(
        text_qa_template=qa_prompt_tmpl,
        similarity_top_k=2,
        streaming=False
    )
    return engine

# --- 5. INTERFAZ VISUAL BÁSICA ---
st.set_page_config(page_title="Asistente de Química BI", page_icon="🧪")
st.title("🧪 Profesor-Asistente de Química (JuandiBot)")

# IMPORTANTE: Guardamos el motor en la variable 'query_engine'
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
               # 1. Obtenemos la respuesta completa (sin streaming de API, para evitar el error 400)
                response = query_engine.query(prompt)
                texto_final = response.response
                
                # 2. GENERADOR ARTIFICIAL: Para mantener el efecto visual que querías
                def generador_lento():
                    # Dividimos por espacios para animar palabra por palabra
                    for palabra in texto_final.split(" "):
                        yield palabra + " "
                        time.sleep(0.03) # Velocidad del efecto

                # 3. MOSTRAR CON ANIMACIÓN
                full_response = st.write_stream(generador_lento())
                
                # 4. GUARDADO
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                # Capturamos el error de 'index out of range' o cuota
                if "429" in str(e):
                    st.error("⚠️ Cuota agotada. Espera 15 segundos.")
                elif "index out of range" in str(e).lower():
                    st.error("⚠️ El asistente no encontró suficiente información en ese apartado. Prueba a preguntar de otra forma.")
                else:
                    st.error(f"Error: {e}")
