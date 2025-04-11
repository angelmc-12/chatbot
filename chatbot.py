import streamlit as st
from google import genai
from google.genai import types
from langchain.document_loaders import PyPDFLoader
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry
from kaggle_secrets import UserSecretsClient
import chromadb

# Solicitar la clave API de Google al usuario
GOOGLE_API_KEY = st.text_input("Ingresa tu clave API de Google:", type="password")

# Si el usuario no ha ingresado una clave, no continúa el código
if not GOOGLE_API_KEY:
    st.warning("Por favor, ingresa tu clave API de Google para continuar.")
else:
    # Inicializa el cliente de Google GenAI
    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Cargar el documento PDF
    @st.cache_resource
    def cargar_documento():
        # Descargar el archivo PDF (solo si no está previamente descargado)
        !wget https://raw.githubusercontent.com/angelmc-12/myfirstrepo/master/beca_pronabec.pdf
        
        loader = PyPDFLoader('beca_pronabec.pdf')
        data = loader.load()

        def limpiar_texto(texto):
            # Eliminar encabezados o pies de página conocidos
            texto = re.sub(r"Resolución Directoral Ejecutiva \nNº 022-2025-MINEDU/VMGI-PRONABEC \n \nLima, 05 de marzo de 2025", "", texto)
            texto = re.sub(r'Esta es una copia autenticada imprimible de un documento electrónico archivado por el PRONABEC, aplicando lo dispuesto por el Art. 25 del D.S. 070-2013- PCM y la Tercera Disposición Complementaria\nFinal del D.S. 026-2016-PCM.\nSu autenticidad e integridad pueden ser contrastadas a través de la siguiente dirección web: "https://mitramite.pronabec.gob.pe/verifica" e ingresar clave: GCAFFCFC código seguridad: 113', "", texto, flags=re.DOTALL)
            return texto.strip()

        data_limpio = []
        for d in data:
            texto_limpio = limpiar_texto(d.page_content)
            doc_limpio = d
            doc_limpio.page_content = texto_limpio
            data_limpio.append(doc_limpio)

        paginas_a_eliminar = [4,5,6,7]

        data_filtrado = [d for i, d in enumerate(data_limpio) if i not in paginas_a_eliminar]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=300,
            length_function=len
        )

        documents = text_splitter.split_documents(data_filtrado)
        return documents

    documents = cargar_documento()

    # Define una función para la clase de embebido
    is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

    class GeminiEmbeddingFunction(EmbeddingFunction):
        document_mode = True

        @retry.Retry(predicate=is_retriable)
        def __call__(self, input: Documents) -> Embeddings:
            if self.document_mode:
                embedding_task = "retrieval_document"
            else:
                embedding_task = "retrieval_query"

            response = client.models.embed_content(
                model="models/text-embedding-004",
                contents=input,
                config=types.EmbedContentConfig(task_type=embedding_task),
            )
            return [e.values for e in response.embeddings]

    # Inicializa la base de datos de Chroma
    DB_NAME = "googlecardb"
    embed_fn = GeminiEmbeddingFunction()
    embed_fn.document_mode = True

    chroma_client = chromadb.Client()
    db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

    db.add(documents=[doc.page_content for doc in documents], ids=[str(i) for i in range(len(documents))])

    # Cambia a modo de consulta cuando generes embeddings.
    embed_fn.document_mode = False

    # Solicitar la pregunta del usuario
    query = st.text_input("Haz una pregunta sobre las becas:", "")

    if query:
        # Realiza la consulta a la base de datos de Chroma
        result = db.query(query_texts=[query], n_results=5)
        [all_passages] = result["documents"]

        # Formatear la consulta para el modelo de GenAI
        query_oneline = query.replace("\n", " ")

        prompt = f"""Eres un asistente amable y claro que ayuda a las personas a entender mejor las becas disponibles. Usa el texto de referencia incluido más abajo para responder la pregunta de forma completa y útil. 
        Explica los conceptos de manera sencilla, sin usar jerga técnica, y mantén un tono amigable y cercano. Si el texto no tiene relación con la pregunta, puedes ignorarlo. Responde en español.

        PREGUNTA: {query_oneline}
        """

        # Agregar los pasajes recuperados al prompt
        for passage in all_passages:
            passage_oneline = passage.replace("\n", " ")
            prompt += f"PASSAGE: {passage_oneline}\n"

        # Generar la respuesta usando Google GenAI
        answer = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        # Mostrar la respuesta en la interfaz de Streamlit
        st.markdown(f"**Respuesta:** {answer.text}")
