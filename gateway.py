"""OpenAI-compatible gateway wrapping an abliterated model + Policy Gateway.

Run:  uvicorn gateway:app --host 0.0.0.0 --port 8000
Then: curl http://localhost:8000/v1/chat/completions -H 'Authorization: Bearer demo' \
      -d '{"model":"abliterated","messages":[{"role":"user","content":"hi"}]}'
"""
from __future__ import annotations
import os, json, time, uuid, httpx
from typing import List, Optional, AsyncIterator
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from policy_engine import default_gateway, PolicyGateway

app = FastAPI(title="Abliterated Mini Gateway")
GATEWAY: PolicyGateway = default_gateway()

ABLIT_KEY = os.getenv("ABLIT_KEY", "")
ABLIT_URL = "https://api.abliteration.ai/v1/chat/completions"
DEFAULT_MODEL = "abliterated-model"  # base model (cheapest, multimodal)
LARGE_MODEL = "abliterated-model-large-v2"  # GLM-5.3 (frontier)


class Msg(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    model: str = "abliterated"
    messages: List[Msg]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 512
    policy_ids: Optional[List[str]] = None


def _last_user(req: ChatReq) -> str:
    for m in reversed(req.messages):
        if m.role == "user":
            return m.content
    return ""


async def _backend_stream(prompt: str, temperature: float, max_tokens: int, model: str) -> AsyncIterator[str]:
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", ABLIT_URL, headers={
            "Authorization": f"Bearer {ABLIT_KEY}",
            "Content-Type": "application/json",
        }, json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "reasoning_effort": "low",
        }) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                        if delta:
                            yield delta
                    except Exception:
                        pass


async def _backend_full(prompt: str, temperature: float, max_tokens: int, model: str) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(ABLIT_URL, headers={
            "Authorization": f"Bearer {ABLIT_KEY}",
            "Content-Type": "application/json",
        }, json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_effort": "low",
        })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""


@app.post("/v1/chat/completions")
async def chat(req: ChatReq, authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(401, "missing auth")
    user_text = _last_user(req)

    # 1. Policy check on request
    req_decisions = GATEWAY.evaluate(user_text, phase="request", policy_ids=req.policy_ids)
    # If any refuse & block, short-circuit
    if any(d.outcome == "refuse" and d.matched for d in req_decisions):
        refusal = GATEWAY.apply(user_text, req_decisions)
        body = {"id": "chatcmpl-" + uuid.uuid4().hex[:8], "object": "chat.completion",
                "created": int(time.time()), "model": req.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": refusal},
                             "finish_reason": "content_filter"}]}
        return JSONResponse(body)

    # rewrite/redact the user text if needed
    user_text = GATEWAY.apply(user_text, req_decisions)

    if req.stream:
        async def gen():
            cid = "chatcmpl-" + uuid.uuid4().hex[:8]
            async for tok in _backend_stream(user_text, req.temperature, req.max_tokens, req.model or DEFAULT_MODEL):
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                         "model": req.model, "choices": [{"index": 0, "delta": {"content": tok},
                                                          "finish_reason": None}]}
                yield "data: " + json.dumps(chunk) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    content = await _backend_full(user_text, req.temperature, req.max_tokens, req.model or DEFAULT_MODEL)
    # 2. Policy check on response
    resp_decisions = GATEWAY.evaluate(content, phase="response", policy_ids=req.policy_ids)
    content = GATEWAY.apply(content, resp_decisions)
    body = {"id": "chatcmpl-" + uuid.uuid4().hex[:8], "object": "chat.completion",
            "created": int(time.time()), "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(content.split()),
                      "total_tokens": len(content.split())}}
    return JSONResponse(body)


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [
        {"id": "abliterated-model", "object": "model", "owned_by": "abliteration.ai"},
        {"id": "abliterated-model-large", "object": "model", "owned_by": "abliteration.ai"},
        {"id": "abliterated-model-large-v2", "object": "model", "owned_by": "abliteration.ai"},
    ]}


@app.get("/health")
def health():
    return {"status": "ok", "policies": len(GATEWAY.policies)}
