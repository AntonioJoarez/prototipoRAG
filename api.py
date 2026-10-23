from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import json
import os

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

llm = OllamaLLM(model="llama3.2:3b", temperature=0.1)

prompt = PromptTemplate.from_template("""
Você é um tutor de programação. Responda na lingua portuguesa, com base no contexto da apostila fornecida.
Se a resposta estiver no contexto: responda de forma clara e didática,
podendo usar exemplos de código quando útil.
Se o contexto tiver informação parcial: use APENAS o que está no contextoe
deixe claro o que não foi coberto. Se a informação não estiver no contexto:
responda exatamente "Não sei como ajudá-lo(a). Procure o seu professor."
e sugira ao aluno consultar o professor ou outra fonte. Nunca invente definições,
exemplos ou comportamentos de linguagens que não constem no contexto.

Contexto:
{context}

Pergunta: {question}
Resposta:
""")

def formatar_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | formatar_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

app = FastAPI()

class Pergunta(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(os.path.dirname(__file__), "static", "index.html"), encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
async def chat(body: Pergunta):
    async def gerar():
        async for chunk in chain.astream(body.question):
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gerar(), media_type="text/event-stream")

app.mount("/static", StaticFiles(directory="static"), name="static")
