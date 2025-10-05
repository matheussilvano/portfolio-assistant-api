# main.py
import os
import re
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# --- Novas importações para Streaming e CORS ---
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# --- Configuração Inicial ---
ASSISTANT_ID = os.getenv("ASSISTANT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not ASSISTANT_ID or not OPENAI_API_KEY:
    raise ValueError("As variáveis de ambiente OPENAI_API_KEY e ASSISTANT_ID devem ser definidas no arquivo .env")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(
    title="API do Assistente de Carreira",
    description="Uma API para interagir com o bot de portfólio do Matheus Silvano.",
    version="1.0.0"
)

# --- Bloco de configuração do CORS ---
origins = [
    "https://matheussilvano.github.io",
    "http://127.0.0.1:5500",
    "http://localhost",
    "http://localhost:8080",
    "null",
    "https://www.matheussilvano.dev",
    "https://matheus-silvano.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Modelos de Dados (Pydantic) ---
class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        description="A pergunta a ser feita ao assistente.",
        examples=["Qual a sua experiência com projetos de IA?"]
    )
    thread_id: str | None = Field(
        default=None,
        description="O ID da conversa (thread) para manter o contexto. Se for nulo, uma nova conversa será criada."
    )

class AnswerResponse(BaseModel):
    answer: str = Field(
        ...,
        description="A resposta gerada pelo assistente."
    )
    thread_id: str = Field(
        ...,
        description="O ID da conversa, para ser usado em perguntas de acompanhamento."
    )


# --- Endpoint da API com Streaming ---
@app.post("/ask",
          summary="Envia uma pergunta ao assistente via streaming",
          description="Recebe uma pergunta e um ID de conversa opcional. Retorna a resposta do assistente em tempo real (streaming).")
async def ask_assistant_streaming(request: QuestionRequest):
    thread_id = request.thread_id
    
    try:
        if thread_id is None:
            thread = client.beta.threads.create()
            thread_id = thread.id
        
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=request.question
        )

        async def stream_generator():
            # Usar 'stream=True' em vez de 'create_and_poll'
            with client.beta.threads.runs.stream(
                thread_id=thread_id,
                assistant_id=ASSISTANT_ID,
            ) as stream:
                # Envia o thread_id como o primeiro evento
                initial_data = {"event": "thread_id", "data": thread_id}
                yield f"data: {json.dumps(initial_data)}\n\n"

                # Itera sobre os eventos de streaming
                for event in stream:
                    # Verifica se há um delta de texto no evento
                    if event.event == 'thread.message.delta':
                        if event.data.delta.content:
                            text_chunk = event.data.delta.content[0].text.value
                            # Limpa anotações em tempo real e envia o pedaço de texto
                            cleaned_chunk = re.sub(r'【.*?】', '', text_chunk).strip()
                            if cleaned_chunk:
                                chunk_data = {"event": "text_chunk", "data": cleaned_chunk}
                                yield f"data: {json.dumps(chunk_data)}\n\n"
        
        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ocorreu um erro interno: {str(e)}"
        )

@app.get("/", include_in_schema=False)
def root():
    return {"message": "API do Assistente de Carreira está funcionando. Acesse /docs para a documentação interativa."}