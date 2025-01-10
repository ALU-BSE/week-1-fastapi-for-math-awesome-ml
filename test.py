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
    result = []
    for i in range(len(M)):
        row = []
        for j in range(len(X[0])):
            sum_product = sum(M[i][k] * X[k][j] for k in range(len(X)))
            row.append(sum_product + B[i])
        result.append(row)
    return result

@app.post("/calculate")
async def calculate(request: MatrixRequest):
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
