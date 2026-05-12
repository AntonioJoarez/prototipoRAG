from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Carregar o PDF
print("Carregando PDF...")
loader = PyMuPDFLoader("data/apostila.pdf")
documents = loader.load()
print(f"{len(documents)} páginas carregadas.")

# 2. Dividir em chunks
print("Dividindo em chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"{len(chunks)} chunks criados.")

# 3. Gerar embeddings e salvar no ChromaDB
print("Gerando embeddings (pode demorar alguns minutos)...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="vectorstore"
)

print("Pronto! Vectorstore salvo em ./vectorstore")