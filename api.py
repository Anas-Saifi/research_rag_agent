import os
import threading
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

print("api.py: importing", flush=True)

app = FastAPI(title="Research Paper Assistant API", version="1.0.0")

# Configure CORS origins
default_origins = [
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "https://research-rag-agent.vercel.app",
]

allowed_origins_raw = os.environ.get("ALLOWED_ORIGINS", "")
if allowed_origins_raw.strip() == "*":
    allow_origins = ["*"]
elif allowed_origins_raw.strip():
    allow_origins = [origin.strip() for origin in allowed_origins_raw.split(",") if origin.strip()]
    for origin in default_origins:
        if origin not in allow_origins:
            allow_origins.append(origin)
else:
    allow_origins = default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True if "*" not in allow_origins else False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Lazy graph loading -----------------------------------------------------
# Importing nodes pulls in langchain/langgraph/pinecone, which is slow on a
# small instance. Doing it at import time delays uvicorn from opening its port,
# and Render reports "No open ports detected". So the graph is loaded in a
# background thread after the server is up.
_graph = None
_graph_lock = threading.Lock()


def get_graph():
    global _graph
    with _graph_lock:
        if _graph is None:
            t0 = time.time()
            print("api.py: loading graph...", flush=True)
            from nodes import graph

            _graph = graph
            print(f"api.py: graph ready in {time.time() - t0:.1f}s", flush=True)
    return _graph


@app.on_event("startup")
def warm_up():
    threading.Thread(target=get_graph, daemon=True).start()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str


@app.get("/")
def root():
    return {
        "name": "Research Paper Assistant API",
        "status": "online",
        "health": "/health",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def run_query(body: QueryRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    try:
        result = get_graph().invoke({"query": body.query, "searched": 0})

        # LangChain sometimes returns a list of content blocks instead of a string
        llm_response = result["llm_response"]
        if isinstance(llm_response, list):
            # Extract the text content from the blocks
            llm_response = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in llm_response
            )

        return QueryResponse(response=str(llm_response))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


print("api.py: ready", flush=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)