from fastapi import FastAPI,HTTPException
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
    rows_M = len(M)
    cols_X = len(X[0])
    result = [[0 for _ in range(cols_X)] for _ in range(rows_M)]
    
    for i in range(rows_M):
        for j in range(cols_X):
            sum_product = 0
            for k in range(len(X)):
                sum_product += M[i][k] * X[k][j]
            # Add bias
            result[i][j] = sum_product + B[i][j]
    return result

@app.post("/calculate")
async def calculate(request: MatrixRequest):
    M = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ])
    
    B = np.array([
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5]
    ])

    X = np.array(request.matrix)

    try:
    
        result_np = matrix_multiply_with_numpy(M, X, B)

        result_no_np = matrix_multiply_without_numpy(M.tolist(), X.tolist(), B.tolist())

        sigmoid_result = sigmoid(result_np)

        return {
            "matrix_multiplication": result_np.tolist(),
            "non_numpy_multiplication": result_no_np,  
            "sigmoid_output": sigmoid_result.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
