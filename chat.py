from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Carregar o vectorstore salvo
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. Configurar o LLM
llm = OllamaLLM(model="llama3.2:3b")

# 3. Prompt
prompt = PromptTemplate.from_template("""
Você é um assistente que responde SOMENTE com base no contexto de programação de computadores.
NUNCA use conhecimento próprio. Se a informação não estiver no contexto abaixo, 
responda EXATAMENTE: "Não encontrei essa informação na apostila."

Contexto:
{context}n

Pergunta: {question}
Resposta:
""")

# 4. Chain
def formatar_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | formatar_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# TESTE TEMPORÁRIO
# pergunta_teste = "quantos ponteiros são necessários para manipular uma fila?"
# docs = retriever.invoke(pergunta_teste)
# print("\n--- CHUNKS ENCONTRADOS ---")
# for i, doc in enumerate(docs):
#     print(f"\nChunk {i+1} (página {doc.metadata.get('page', '?')}):")
#     print(doc.page_content)
# print("--- FIM DOS CHUNKS ---\n")




# 5. Loop de chat
print("Chat pronto! Digite 'sair' para encerrar.\n")
while True:
    pergunta = input("Você: ")
    if pergunta.lower() == "sair":
        break
    resposta = chain.invoke(pergunta)
    print(f"\nAssistente: {resposta}\n")