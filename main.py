"""
soul.py v2.0 Demo — RAG + RLM Hybrid
Shows automatic query routing: FOCUSED → RAG, EXHAUSTIVE → RLM
Three modes: Auto, RAG only, RLM only
"""
import os, sys, time, uuid, tempfile
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request, Cookie
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, ".")
from hybrid_agent import HybridAgent

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

sessions: dict = {}
SESSION_TTL = 1800

def make_agent(mem_path: str, soul_path: str) -> HybridAgent:
    return HybridAgent(
        soul_path=soul_path,
        memory_path=mem_path,
        mode="auto",  # overridden per request
        collection_name=f"soul_v2_{uuid.uuid4().hex[:8]}",
        qdrant_url=os.environ.get("QDRANT_URL",""),
        qdrant_api_key=os.environ.get("QDRANT_API_KEY",""),
        azure_embedding_endpoint=os.environ.get("AZURE_EMBEDDING_ENDPOINT",""),
        azure_embedding_key=os.environ.get("AZURE_EMBEDDING_KEY",""),
        k=4,
        rlm_chunk_size=8,
    )

DEFAULT_SOUL = (
    "You are a helpful persistent AI assistant running on soul.py v2.0. "
    "Be concise and direct. Acknowledge what you remember naturally."
)

def get_or_create_session(session_id: str) -> dict:
    now = time.time()
    stale = [k for k,v in sessions.items() if now - v["last_active"] > SESSION_TTL]
    for k in stale:
        try: os.unlink(sessions[k]["mem_path"])
        except: pass
        try: os.unlink(sessions[k]["soul_path"])
        except: pass
        del sessions[k]

    if session_id not in sessions:
        tmp_mem  = tempfile.NamedTemporaryFile(suffix=".md", delete=False)
        tmp_soul = tempfile.NamedTemporaryFile(suffix=".md", delete=False)
        tmp_mem.write(b"# MEMORY.md\n"); tmp_mem.close()
        tmp_soul.write(DEFAULT_SOUL.encode()); tmp_soul.close()
        agent = make_agent(tmp_mem.name, tmp_soul.name)
        sessions[session_id] = {
            "agent": agent,
            "mem_path": tmp_mem.name,
            "soul_path": tmp_soul.name,
            "last_active": now,
            "message_count": 0,
            "session_count": 1,
            "history": [],
        }
    sessions[session_id]["last_active"] = now
    return sessions[session_id]

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(open("index.html").read())

@app.post("/ask")
async def ask(request: Request, session_id: str = Cookie(default=None)):
    try:
        body = await request.json()
        question = body.get("question","").strip()
        force_mode = body.get("mode","auto")  # "auto" | "rag" | "rlm"
        if not question: return JSONResponse({"error":"empty"},status_code=400)
        if not session_id: session_id = str(uuid.uuid4())

        session = get_or_create_session(session_id)
        agent = session["agent"]
        agent.mode = force_mode  # override mode per request
        agent._history = session["history"]  # restore history

        result = agent.ask(question, remember=True)

        session["history"] = agent._history
        session["message_count"] += 1

        memory_text = Path(session["mem_path"]).read_text()

        response = JSONResponse({
            "answer":        result["answer"],
            "route":         result["route"],
            "router_ms":     result["router_ms"],
            "retrieval_ms":  result["retrieval_ms"],
            "total_ms":      result["total_ms"],
            "rag_context":   result["rag_context"] or "",
            "rlm_meta":      result["rlm_meta"],
            "memory":        memory_text,
            "message_count": session["message_count"],
            "session_count": session["session_count"],
            "total_memories": agent._rag.count(),
            "mode_used":     result["route"],
        })
        response.set_cookie("session_id", session_id, max_age=SESSION_TTL, samesite="lax")
        return response
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)

@app.post("/new-session")
async def new_session(session_id: str = Cookie(default=None)):
    if session_id and session_id in sessions:
        sessions[session_id]["history"] = []
        sessions[session_id]["agent"]._history = []
        sessions[session_id]["session_count"] += 1
        memory_text = Path(sessions[session_id]["mem_path"]).read_text()
        return JSONResponse({
            "ok": True,
            "memory": memory_text,
            "session_count": sessions[session_id]["session_count"],
            "total_memories": sessions[session_id]["agent"]._rag.count(),
        })
    return JSONResponse({"ok": False}, status_code=400)

@app.post("/reset")
async def reset(session_id: str = Cookie(default=None)):
    if session_id and session_id in sessions:
        try: os.unlink(sessions[session_id]["mem_path"])
        except: pass
        del sessions[session_id]
    return JSONResponse({"ok": True})

@app.get("/health")
async def health():
    return {
        "ok": True,
        "sessions": len(sessions),
        "key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "qdrant_set": bool(os.environ.get("QDRANT_URL")),
        "embedding_set": bool(os.environ.get("AZURE_EMBEDDING_ENDPOINT")),
        "version": "2.0"
    }
