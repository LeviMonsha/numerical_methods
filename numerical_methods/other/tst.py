import numpy as np

def simple_iteration(A, b, x0, epsilon=1e-10, max_iter=1000):
    n = len(b)
    x = x0.copy()

    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)

    T = np.linalg.inv(D) @ (L + U)
    C = np.linalg.inv(D) @ b

    for k in range(max_iter):
        x_new = T @ x + C
        if np.linalg.norm(x_new - x) < epsilon:
            print("Converged in", k + 1, "iterations")
            return x_new
        x = x_new

    print("Did not converge after", max_iter, "iterations")
    return x


A = np.array([[1, 1, 1],
                  [1, -1, 0],
                  [0, -1, 4]])
b = np.array([54, 8, 85])
x0 = np.zeros_like(b)

solution = simple_iteration(A, b, x0)
print("Solution:", solution)


def method_Gauss(mat: np.array) -> list[float]:
    mat = mat.copy()
    row_count = len(mat)
    for i in range(row_count):
        # Приведение матрицы к верхнетреугольному виду
        for j in range(i+1, row_count):
            div = mat[j][i] / mat[i][i]
            #mat[i] = [mat[i][t] - mat[i][t] * div for t in range(row_count+1)]
            mat[j][i:-1] = mat[j][i:-1] - div * mat[i][i:-1]
            #mat[j] = [mat[j][t] - mat[i][t] for t in range(row_count+1)]
            mat[j][i:-1] = mat[j][i:-1] - mat[i][i:-1]

            mat[j][-1] = mat[j][-1] - div * mat[i][-1]
            print(mat)

    # Решение уравнения с обратным ходом
    res = [0] * row_count
    res[row_count - 1] = mat[row_count - 1][-1] / mat[row_count - 1][row_count - 1]

    for i in range(row_count-2, -1, -1):
        res[i] = (mat[i][-1] - np.dot(mat[i][i + 1:-1], res[i + 1:])) / mat[i][i]
    return res