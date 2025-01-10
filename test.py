from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import uvicorn

app = FastAPI()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class MatrixRequest(BaseModel):
    matrix: list[list[float]]

def matrix_multiply_with_numpy(M, X, B):
    result = np.dot(M, X) + B
    return result

def matrix_multiply_without_numpy(M, X, B):
    result = [[sum(M[i][k] * X[k][j] for k in range(len(X))) + B[i][j] for j in range(len(X[0]))] for i in range(len(M))]
    return result

@app.post("/calculate")
async def calculate(request: MatrixRequest):
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
