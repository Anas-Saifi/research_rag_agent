from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from nodes import graph

app = FastAPI(title="Research Paper Assistant API", version="1.0.0")

# Allow the Vite dev server and any local origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def run_query(body: QueryRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    try:
        result = graph.invoke({"query": body.query, "searched": 0})
        
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


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
