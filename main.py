# main.py
import os
import re
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import OpenAI

# ------------------------------------------------------------------
# Env & OpenAI
# ------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID   = os.getenv("ASSISTANT_ID")  # usado no /ask

if not OPENAI_API_KEY:
    raise ValueError("Defina OPENAI_API_KEY no .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------------------------------------------
# FastAPI + CORS
# ------------------------------------------------------------------
app = FastAPI(
    title="Atlas.AI API",
    description="Admin + consultas (local e Assistants API) em um único arquivo.",
    version="1.0.0",
)

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:5500",
    "https://matheussilvano.github.io",
    "https://www.matheussilvano.dev",
    "https://matheus-silvano.vercel.app/",
    "matheussilvano.dev",
    "null",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"],
)

# ------------------------------------------------------------------
# Tipos
# ------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., examples=["Qual a sua experiência com projetos de IA?"])
    thread_id: Optional[str] = Field(
        default=None,
        description="Thread para manter contexto. Se vazio, cria nova."
    )

class ConsultaAtlasRequest(BaseModel):
    pergunta: str = Field(..., examples=["Quem tem experiência com IA?"])

# ------------------------------------------------------------------
# Local mode (equipe.json) — como era antes
# ------------------------------------------------------------------
EQUIPE_JSON_PATH = Path(__file__).parent / "equipe.json"
try:
    equipe_data = json.loads(EQUIPE_JSON_PATH.read_text(encoding="utf-8"))
except Exception:
    equipe_data = {}

def gerar_resposta(pergunta: str) -> str:
    """
    Consulta local: injeta o JSON no prompt e usa chat.completions.
    """
    if not equipe_data:
        raise RuntimeError("equipe.json não encontrado ou vazio ao tentar responder localmente.")

    contexto = (
        "Você é um assistente que conhece todos os membros da equipe Atlas.AI.\n"
        "Abaixo estão os dados da equipe em formato JSON:\n\n"
        f"{json.dumps(equipe_data, ensure_ascii=False, indent=2)}\n\n"
        "Com base nisso, responda à pergunta do usuário. "
        "Se for sobre quem tem experiência em algo ou quem é mais indicado para uma vaga, "
        "diga o(s) nome(s) e justifique de forma breve e clara, explicando o porquê da escolha. "
        "Sempre use informações reais do JSON. "
        "Responda em português e de forma natural.\n\n"
        f"Pergunta: {pergunta}"
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um assistente inteligente especializado em perfis de equipe."},
            {"role": "user", "content": contexto},
        ],
        temperature=0.7,
    )
    return completion.choices[0].message.content

# ------------------------------------------------------------------
# Endpoint 1: /consulta-atlas (LOCAL, sem streaming)
# ------------------------------------------------------------------
@app.post(
    "/consulta-atlas",
    summary="Consulta local à Atlas.AI (equipe.json)",
    description="Lê equipe.json e responde via chat.completions (sem streaming)."
)
async def consulta_atlas_local(req: ConsultaAtlasRequest):
    try:
        answer = gerar_resposta(req.pergunta)
        return {"answer": answer, "mode": "local-json"}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao gerar resposta local: {e}")

# ------------------------------------------------------------------
# Endpoint 2: /ask (Assistants API com streaming SSE)
# ------------------------------------------------------------------
@app.post(
    "/ask",
    summary="Pergunta ao Assistente (Assistants API) com streaming",
    description="Mantém threads e faz streaming de chunks via SSE."
)
async def ask_assistant_streaming(req: AskRequest):
    if not ASSISTANT_ID:
        raise HTTPException(status_code=400, detail="ASSISTANT_ID não definido no .env para usar /ask.")

    thread_id = req.thread_id
    try:
        if thread_id is None:
            thread = client.beta.threads.create()
            thread_id = thread.id

        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=req.question
        )

        async def stream_generator():
            # Manda o thread_id primeiro (para o front-end guardar)
            yield "data: " + json.dumps({"event": "thread_id", "data": thread_id}) + "\n\n"

            # Stream do run
            with client.beta.threads.runs.stream(
                thread_id=thread_id,
                assistant_id=ASSISTANT_ID,
            ) as stream:
                for event in stream:
                    if event.event == "thread.message.delta":
                        if event.data.delta.content:
                            text_chunk = event.data.delta.content[0].text.value
                            cleaned = re.sub(r'【.*?】', '', text_chunk)
                            if cleaned:
                                yield "data: " + json.dumps({"event": "text_chunk", "data": cleaned}) + "\n\n"

                    elif event.event == "thread.run.requires_action":
                        # Se seu Assistant chamar tools personalizadas, trate aqui.
                        run_id = event.data.id
                        tool_calls = event.data.required_action.submit_tool_outputs.tool_calls
                        tool_outputs = []

                        # Exemplo: navegar no front
                        for call in tool_calls:
                            if call.function.name == "navigateToSection":
                                args = json.loads(call.function.arguments)
                                yield "data: " + json.dumps({"event": "tool_call", "data": {"name": "navigateToSection", "arguments": args}}) + "\n\n"
                                tool_outputs.append({
                                    "tool_call_id": call.id,
                                    "output": json.dumps({"success": True})
                                })

                        if tool_outputs:
                            with client.beta.threads.runs.submit_tool_outputs_stream(
                                thread_id=thread_id,
                                run_id=run_id,
                                tool_outputs=tool_outputs,
                            ) as follow_stream:
                                for part in follow_stream:
                                    if part.event == "thread.message.delta":
                                        if part.data.delta.content:
                                            text_chunk = part.data.delta.content[0].text.value
                                            cleaned = re.sub(r'【.*?】', '', text_chunk)
                                            if cleaned:
                                                yield "data: " + json.dumps({"event": "text_chunk", "data": cleaned}) + "\n\n"

        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Encoding": "identity",
        }
        return StreamingResponse(stream_generator(), headers=headers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no /ask: {e}")

# ------------------------------------------------------------------
# Root
# ------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return {"message": "Atlas.AI API ativa. Veja /docs."}
