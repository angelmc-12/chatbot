!pip uninstall -qqy jupyterlab kfp  # Remove unused conflicting packages
!pip install -qU "google-genai==1.7.0" "chromadb==0.6.3"
!pip install -U -q langchain-community

from google import genai
from google.genai import types

from IPython.display import Markdown

genai.__version__

from kaggle_secrets import UserSecretsClient

GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)

for m in client.models.list():
    if "embedContent" in m.supported_actions:
        print(m.name)

!wget https://raw.githubusercontent.com/angelmc-12/myfirstrepo/master/beca_pronabec.pdf

from langchain.document_loaders import PyPDFLoader
import re

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

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=300,
    length_function = len
)

documents = text_splitter.split_documents(data_filtrado)
print(f"Generamos {len(documents)} fragmentos")

from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry

from google.genai import types


# Define a helper to retry when per-minute quota is reached.
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})


class GeminiEmbeddingFunction(EmbeddingFunction):
    # Specify whether to generate embeddings for documents, or queries
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
            config=types.EmbedContentConfig(
                task_type=embedding_task,
            ),
        )
        return [e.values for e in response.embeddings]

import chromadb

DB_NAME = "googlecardb"

embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = True

chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

db.add(documents=[doc.page_content for doc in documents], ids=[str(i) for i in range(len(documents))])

# Switch to query mode when generating embeddings.
embed_fn.document_mode = False

# Search the Chroma DB using the specified query.
query = "Cuantas becas existen y cuales son las instituciones elegibles?"

result = db.query(query_texts=[query], n_results=5)
[all_passages] = result["documents"]

query_oneline = query.replace("\n", " ")

prompt = f"""Eres un asistente amable y claro que ayuda a las personas a entender mejor las becas disponibles. Usa el texto de referencia incluido más abajo para responder la pregunta de forma completa y útil. 
Explica los conceptos de manera sencilla, sin usar jerga técnica, y mantén un tono amigable y cercano. Si el texto no tiene relación con la pregunta, puedes ignorarlo. Responde en español.

PREGUNTA: {query_oneline}
"""


# Add the retrieved documents to the prompt.
for passage in all_passages:
    passage_oneline = passage.replace("\n", " ")
    prompt += f"PASSAGE: {passage_oneline}\n"

answer = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt)

Markdown(answer.text)