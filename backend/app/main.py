import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .predict import router
from .utils import set_seed

app = FastAPI(title="Electronix AI Sentiment API", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*","http://localhost:5173" ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# deterministic seeds for CPU-only runs
set_seed(42)

app.include_router(router, prefix="")

@app.get("/")
async def health_check():
    return {"status": "ok"}


