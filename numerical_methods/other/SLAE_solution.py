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

# проверка на правильность размерности матрицы
def check_matrix(mat: list[list[float]]) -> bool:
    return len(mat) != len(mat[0]) - 1

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

# проверка на сходимость матрицы
def matrix_norm(matrix: list[list[float]]) -> bool:
    # сумма всех столбцов
    sum = 0.0
    for i in range(len(matrix)):
        for j in range(len(matrix[i]) - 1):
            if i != j:
                sum += matrix[i][j]
    return sum <= 1

# Метод простой итерации
def method_simple_iteration(matrix: list[list[float]],
                            MAX_ITER = 30,
                            accuracy = 1e-3) \
        -> list[float] | None:
    matrix_copy = copy_matrix(matrix)
    row_count = len(matrix_copy)
    column_count = len(matrix_copy[0])

    if matrix_norm(matrix_copy):
        print("Сходимость осуществляется")
    else:
        raise NotImplementedError("Сходимость не осуществляется")

    d_mas = [0.0] * row_count
    for i in range(row_count):
        d_mas[i] = [0.0] * row_count
        for j in range(column_count):
            if i != j:
                d_mas[i].append(-matrix_copy[i][j] / matrix_copy[i][i])

    cur = [0.0] * (row_count)
    for i in range(MAX_ITER):
        tmp = [0.0] * (row_count)
        for j in range(row_count):
            sum = 0.0
            for u in range(row_count):
                sum += d_mas[j][u] * cur[u]
            tmp[j] = sum + d_mas[j][-1]
        iteration = list(map(lambda x: -x, cur))
        print(f"Итерация n = {i}\n{iteration}")
        if all([abs(cur[u] - tmp[u]) < accuracy for u in range(row_count)]):
            return iteration
        cur = tmp

    raise NotImplementedError(f"Прошло {MAX_ITER} итераций")

if __name__ == '__main__':
    matrix = [[1, 1, 1, 54],
            [1, -1, 0, 8],
            [0, -1, 4, 85]]

    if check_matrix(matrix):
        print("Неверная размерность матрицы")
    else:
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


class UnitTest_Matrix(unittest.TestCase):
    def test_Gauss(self):
        self.assertRaises(TypeError, method_Gauss, [['a', 1, 1, 1], [1, 1, 'b', 2], [2, 3, 4, 'c']])
        self.assertRaises(ZeroDivisionError, method_Gauss, [[1, 1, 0, 1], [1, 1, 1, 1], [1, 2, 0, 10]])
        self.assertEqual(method_Gauss([[1, 1, 1, 54], [1, -1, 0, 8], [0, -1, 4, 85]]), [19.0, 11.0, 24.0])

    def test_simple_iteration(self):
        self.assertRaises(TypeError, method_simple_iteration, [['a', 1, 1, 1], [1, 1, 'b', 2], [2, 3, 4, 'c']])
        self.assertRaises(NotImplementedError, method_simple_iteration, [[1, 1, 0, 1], [1, 1, 1, 1], [1, 2, 0, 10]])
        self.assertRaises(NotImplementedError, method_simple_iteration, [[1, 1, 1, 54], [1, -1, 0, 8], [0, -1, 4, 85]])
        self.assertEqual(method_simple_iteration([[-1, 0, 0, -49], [0, 1, 0, 41], [0, 0, 2, 40]]), [49.0, 41.0, 20.0])

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

# метод Гаусса
def method_Gauss(matrix: list[list[float]]) -> list[float]:
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

    matrix_copy = copy_matrix(matrix)
    row_count = len(matrix_copy)

    # print("Выполнение элементарных преобразований над матрицей")
    matrix_copy = conversion_elementary(matrix_copy)
    # display_matrix(matrix_copy, "треугольный вид матрицы")

    # решаем обратным ходом
    res = [0.0] * row_count
    for i in range(row_count - 1, -1, -1):
        res[i] = matrix_copy[i][-1]
        for j in range(i + 1, row_count):
            res[i] -= matrix_copy[i][j] * res[j]
        res[i] /= matrix_copy[i][i]

    return res

# проверка на сходимость матрицы
def matrix_norm(matrix: list[list[float]]) -> bool:
    # сумма всех столбцов
    sum = 0.0
    for i in range(len(matrix)):
        for j in range(len(matrix[i]) - 1):
            if i != j:
                sum += matrix[i][j]
    return sum <= 1

# Метод простой итерации
def method_simple_iteration(matrix: list[list[float]],
                            MAX_ITER=30,
                            accuracy=1e-3) \
        -> list[float] | None:
    # создание копии матрицы
    def copy_matrix(mat: list[list[float]]) -> list[list[float]]:
        c_mat = []
        for row in mat:
            row_copy = row[:]
            c_mat.append(row_copy)
        return c_mat

    matrix_copy = copy_matrix(matrix)
    row_count = len(matrix_copy)

    if matrix_norm(matrix_copy):
        print("Сходимость осуществляется")
    else:
        raise NotImplementedError("Сходимость не осуществляется")

    for i in range(len(matrix)):
        sum = 0.0
        for j in range(len(matrix[i]) - 1):
            if j != i:
                sum += abs(matrix[i][j])
        if sum >= abs(matrix[i][i]):
            print("Условие на сходимость не выполняется")
            break

    x = [0.0] * row_count
    for i in range(row_count):
        x[i] = []
        for j in range(len(matrix[0])):
            if i != j:
                x[i].append(-matrix[i][j] / matrix[i][i])
            else:
                x[i].append(0.0)

    current = [0.0] * row_count
    for i in range(MAX_ITER):
        tmp = [0.0] * row_count
        for j in range(row_count):
            sum = 0.0
            for k in range(row_count):
                sum += x[j][k] * current[k]
            tmp[j] = sum + x[j][-1]
        if all([abs(current[k] - tmp[k]) < accuracy for k in range(row_count)]):
            return list(map(lambda x: -x, current))

        current = tmp

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

class UnitTest_Matrix(unittest.TestCase):
    def test_Gauss(self):
        self.assertRaises(TypeError, method_Gauss, [['a', 1, 1, 1], [1, 1, 'b', 2], [2, 3, 4, 'c']])
        self.assertRaises(ZeroDivisionError, method_Gauss, [[1, 1, 0, 1], [1, 1, 1, 1], [1, 2, 0, 10]])
        self.assertEqual(method_Gauss([[1, 1, 1, 54], [1, -1, 0, 8], [0, -1, 4, 85]]), [19.0, 11.0, 24.0])

    def test_simple_iteration(self):
        self.assertRaises(TypeError, method_simple_iteration, [['a', 1, 1, 1], [1, 1, 'b', 2], [2, 3, 4, 'c']])
        self.assertRaises(NotImplementedError, method_simple_iteration,
                          [[1, 1, 0, 1], [1, 1, 1, 1], [1, 2, 0, 10]])
        self.assertRaises(NotImplementedError, method_simple_iteration,
                          [[1, 1, 1, 54], [1, -1, 0, 8], [0, -1, 4, 85]])
        self.assertEqual(method_simple_iteration([[-1, 0, 0, -49], [0, 1, 0, 41], [0, 0, 2, 40]]),
                         [49.0, 41.0, 20.0])