from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .models import LLM

app = FastAPI()
model = LLM()

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class GenerateRequest(BaseModel):
    messages: list[ChatMessage]
    max_context_tokens: int
    max_new_tokens: int = 2048

@app.get("/")
def read_root():
    return {"message": "Welcome to the LLM API Server!"}

@app.post("/generate")
def generate_text(request: GenerateRequest):
    messages = [message.model_dump() for message in request.messages]
    used_tokens = len(
        model.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    if used_tokens > request.max_context_tokens:
        raise HTTPException(status_code=400, detail="Chat context token limit exceeded")

    generated_text = model.generate(messages, request.max_new_tokens)
    return {"generated_text": generated_text, "used_tokens": used_tokens}
