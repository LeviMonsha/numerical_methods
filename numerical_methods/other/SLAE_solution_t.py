import unittest

# вывод матрицы в консоль
def display_matrix(mat: list[list[float]], text: str = "матрица") -> None:
    print(f"~~~ {text} ~~~")
    for row in mat:
        print(" ".join("{:4}".format(el) for el in row))

# вывод решения СЛАУ в консоль
def display_results(res_list: list[float]) -> None:
    x, y, z = [res_ for res_ in res_list]
    print(f"~~~ результат ~~~\n"
          f"x = {x}\n"
          f"y = {y}\n"
          f"z = {z}")

# создание копии матрицы
def copy_matrix(mat: list[list[float]]) -> list[list[float]]:
    c_mat = []
    for row in mat:
        row_copy = row[:]
        c_mat.append(row_copy)
    return c_mat

# элементарное преобразование
def conversion_elementary(matrix: list[list[float]]) -> list[list[float]]:
    row_count = len(matrix)
    for k in range(row_count):
        for i in range(k + 1, row_count):
            if matrix[k][k] == 0:
                raise ZeroDivisionError("Деление на ноль невозможно")
            else:
                div = matrix[i][k] / matrix[k][k]
            for j in range(k, row_count):
                matrix[i][j] -= div * matrix[k][j]
            matrix[i][-1] -= div * matrix[k][-1]
    return matrix

# метод Гаусса
def method_Gauss(matrix: list[list[float]]) -> list[float]:
    matrix_copy = copy_matrix(matrix)
    row_count = len(matrix_copy)

    print("Выполнение элементарных преобразований над матрицей")
    matrix_copy = conversion_elementary(matrix_copy)
    display_matrix(matrix_copy, "треугольный вид матрицы")

    # решаем обратным ходом
    res = [0.0] * row_count
    for i in range(row_count - 1, -1, -1):
        res[i] = matrix_copy[i][-1]
        for j in range(i + 1, row_count):
            res[i] -= matrix_copy[i][j] * res[j]
        res[i] /= matrix_copy[i][i]

    return res

# элементарное преобразование МПИ
def conversion_simple_iter(matrix: list[list[float]]) -> list[list[float]]:
    row_count = len(matrix)
    column_count = len(matrix[0])

    d_mas = [0.0] * (column_count-1)
    for i in range(row_count):
        if matrix[i][i] != 0:
            d_mas[i] = 1 / matrix[i][i]
        else:
            raise ZeroDivisionError("Деление на ноль невозможно")
    for i in range(row_count):
        for j in range(column_count - 1):
            matrix[i][j] = matrix[i][j] * d_mas[i]
    print(matrix)
    print("simpIter")


    # for k in range(row_count):
    #     x = matrix[k][k]
    #     for i in range(column_count - 1):
    #         if k != i:
    #             if matrix[k][i] != 0:
    #                 matrix[k][i] = -(matrix[k][i] / x)
    #     if matrix[k][-1] != 0:
    #         matrix[k][-1] = matrix[k][-1] / x
    #     matrix[k][k] = matrix[k][-1]
    # print(matrix)
    return matrix

# проверка на сходимость матрицы
def matrix_convergence(matrix: list[list[float]]) -> bool:
    # сумма всех столбцов
    bl = True
    for i in range(len(matrix)):
        sum = 0.0
        for j in range(len(matrix[i]) - 1):
            if i != j:
                sum += abs(matrix[i][j])
        if sum >= abs(matrix[i][i]):
            bl = False
            return bl
    return bl

def other_iter(matrix, resent):
    row_count = len(matrix)
    column_count = len(matrix[0])
    c = [0.0]*row_count
    for i in range(row_count):
        for j in range(column_count-1):
            if i != j:
                c[i] += matrix[i][j] * resent[j]
            else:
                c[i] += matrix[i][j]
        c[i] = c[i] * matrix[i][-1]
    return c

# Метод простой итерации
def method_simple_iteration(matrix: list[list[float]],
                            MAX_ITER = 30,
                            accuracy = 1e-3) \
        -> list[float] | None:
    matrix_copy = copy_matrix(matrix)

    matrix_copy = conversion_simple_iter(matrix_copy)

    if matrix_convergence(matrix_copy):
        print("Сходимость осуществляется")
    else:
        print("Сходимость не осуществляется")

    free = [matrix[0][-1], matrix[1][-1], matrix[2][-1]]
    q = other_iter(matrix, free)
    for i in range(MAX_ITER):
        p = q
        q = other_iter(matrix, p)
        print(f"Итерация n = {i}\n{q}")
        s = [0.0]*len(p)
        for i in range(len(p)):
            s[i] = abs(p[i] - q[i])
        if max(s) <= accuracy:
            return q

    raise NotImplementedError(f"Прошло {MAX_ITER} итераций")

if __name__ == '__main__':
    matrix = [[1, 1, 1, 54],
            [1, -1, 0, 8],
            [0, -1, 4, 85]]

    display_matrix(matrix)

    # вызов метода Гаусса
    print("~~~~~~~~~~~~~~~\nМетод Гаусса\n~~~~~~~~~~~~~~~")
    res_list_Gauss = method_Gauss(matrix)
    display_results(res_list_Gauss)

    # вызов метода простой итерации
    print("~~~~~~~~~~~~~~~\nМетод простой итерации\n~~~~~~~~~~~~~~~")
    res_list_iter = method_simple_iteration(matrix)
    display_results(res_list_iter)

    # # res + y + z = 54
    # # res - 4 = y + 4
    # # 4 * (z - 17) = y + 17
    # #
    # # z = 24
    # # y = 11
    # # x = 19


# class UnitTest_Matrix(unittest.TestCase):
#     def test_Gauss(self):
#         self.assertRaises(TypeError, method_Gauss, [['a', 1, 1, 1], [1, 1, 'b', 2], [2, 3, 4, 'c']])
#         self.assertRaises(ZeroDivisionError, method_Gauss, [[1, 1, 0, 1], [1, 1, 1, 1], [1, 2, 0, 10]])
#         self.assertEqual(method_Gauss([[1, 1, 1, 54], [1, -1, 0, 8], [0, -1, 4, 85]]), [19.0, 11.0, 24.0])
#
#     def test_simple_iteration(self):
#         self.assertRaises(TypeError, method_simple_iteration, [['a', 1, 1, 1], [1, 1, 'b', 2], [2, 3, 4, 'c']])
#         self.assertRaises(ZeroDivisionError, method_simple_iteration, [[1, 1, 0, 1], [1, 1, 1, 1], [1, 2, 0, 10]])
#         self.assertRaises(NotImplementedError, method_simple_iteration, [[1, 1, 1, 54], [1, -1, 0, 8], [0, -1, 4, 85]])
#         self.assertEqual(method_simple_iteration([[-1, 0, 0, -49], [0, 1, 0, 41], [0, 0, 2, 40]]), [49.0, 41.0, 20.0])