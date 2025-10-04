# main.py
import os
import re
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# --- Nova importação para o CORS ---
from fastapi.middleware.cors import CORSMiddleware

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

# --- Bloco de configuração do CORS (AQUI ESTÁ A CORREÇÃO) ---
# Define de quais origens (sites) a API pode receber requisições.
origins = [
    "https://matheussilvano.github.io", # Permite o seu portfólio online
    "http://127.0.0.1:5500",           # Permite testes locais (se você usar o Live Server do VS Code)
    "http://localhost",
    "http://localhost:8080",
    "null" # Permite requisições sem origem (útil para ferramentas como Postman)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Adicionado OPTIONS para ser explícito
    allow_headers=["*"], # Mantido como curinga, pois é geralmente suficiente
)
# --- Fim do bloco de correção ---


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


# --- Endpoints da API ---
@app.post("/ask",
          response_model=AnswerResponse,
          summary="Envia uma pergunta ao assistente",
          description="Recebe uma pergunta e um ID de conversa opcional. Retorna a resposta do assistente e o ID da conversa para manter o contexto.")
async def ask_assistant(request: QuestionRequest):
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
        
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID,
        )
        
        if run.status == 'completed':
            messages = client.beta.threads.messages.list(
                thread_id=thread_id,
                run_id=run.id
            )
            assistant_message = messages.data[0].content[0].text.value
            
            cleaned_answer = re.sub(r'【.*?】', '', assistant_message).strip()
            
            return AnswerResponse(answer=cleaned_answer, thread_id=thread_id)
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"O assistente não conseguiu processar a requisição. Status: {run.status}"
            )
            
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ocorreu um erro interno: {str(e)}"
        )

@app.get("/", include_in_schema=False)
def root():
    return {"message": "API do Assistente de Carreira está funcionando. Acesse /docs para a documentação interativa."}