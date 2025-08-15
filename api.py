from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Buggy API",
    version="1.0.0",
    description="Example API with a bug",
    servers=[
        {
            "url": "http://localhost:8086",
            "description": "Local Server"
        },
    ]
)

class A(BaseModel):
    a: int

class B(BaseModel):
    b: str

@app.get("/health")
def health():
    a: A = B(b="hello")
    print(a.a)
    return {"status": "ok"}
