# main.py
import os
import re  # Importa a biblioteca de expressões regulares
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# --- Configuração Inicial ---
# Pega as configurações essenciais do ambiente
ASSISTANT_ID = os.getenv("ASSISTANT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validação das variáveis de ambiente
if not ASSISTANT_ID or not OPENAI_API_KEY:
    raise ValueError("As variáveis de ambiente OPENAI_API_KEY e ASSISTANT_ID devem ser definidas no arquivo .env")

# Inicializa o cliente da OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Inicializa a aplicação FastAPI
app = FastAPI(
    title="API do Assistente de Carreira",
    description="Uma API para interagir com o bot de portfólio do Matheus Silvano.",
    version="1.0.0"
)


# --- Modelos de Dados (Pydantic) ---
# Define a estrutura da requisição que a API vai receber
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

# Define a estrutura da resposta que a API vai enviar
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
    """
    Processa uma pergunta para o assistente da OpenAI.
    - Cria uma nova thread (conversa) se nenhum `thread_id` for fornecido.
    - Adiciona a pergunta do usuário à thread.
    - Executa o assistente e aguarda a resposta.
    - Limpa a resposta de quaisquer citações e a retorna.
    """
    thread_id = request.thread_id
    
    try:
        # Se nenhum thread_id for fornecido, cria uma nova conversa
        if thread_id is None:
            thread = client.beta.threads.create()
            thread_id = thread.id
        
        # Adiciona a mensagem do usuário à conversa
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=request.question
        )
        
        # Executa o "Run" e aguarda a conclusão usando o método 'create_and_poll'
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID,
        )
        
        # Verifica se o Run foi concluído com sucesso
        if run.status == 'completed':
            messages = client.beta.threads.messages.list(
                thread_id=thread_id,
                run_id=run.id
            )
            # A resposta mais recente do assistente é sempre a primeira da lista
            assistant_message = messages.data[0].content[0].text.value
            
            # --- Bloco de Limpeza da Resposta ---
            # Remove qualquer padrão como 【...†source】 ou similar usando regex
            # e remove espaços em branco extras no início ou fim.
            cleaned_answer = re.sub(r'【.*?】', '', assistant_message).strip()
            
            # Retorna a resposta JÁ LIMPA
            return AnswerResponse(answer=cleaned_answer, thread_id=thread_id)
        else:
            # Caso o Run falhe ou tenha outro status
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"O assistente não conseguiu processar a requisição. Status: {run.status}"
            )
            
    except Exception as e:
        # Captura de erros genéricos (ex: API da OpenAI fora do ar)
        print(f"Ocorreu um erro inesperado: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ocorreu um erro interno: {str(e)}"
        )

@app.get("/", include_in_schema=False)
def root():
    """ Rota raiz para verificar se a API está no ar. """
    return {"message": "API do Assistente de Carreira está funcionando. Acesse /docs para a documentação interativa."}