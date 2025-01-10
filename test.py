from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import uvicorn

app = FastAPI()

class MatrixRequest(BaseModel):
    matrix: list[list[float]]


@app.post("/calculate")
async def calculate(request: MatrixRequest):
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
