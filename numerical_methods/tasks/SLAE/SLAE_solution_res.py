import unittest
from copy import deepcopy

import numpy as np


def display_matrix(matrix):
    '''вывод матрицы в консоль'''
    print("Исходная система")
    for row in matrix:
        print(" ".join("{:4}".format(el) for el in row))

def method_Gauss(matrix):
    '''Решение систем линейных алгебраических уравнений
        методом Гаусса'''

    def conversion_elementary(mat):
        ''' элементарное преобразование '''
        row_count = len(mat)
        for k in range(row_count):
            d = mat[k][k]
            if int(d) == 0:
                for i in range(k + 1, row_count):
                    if int(d) == 0:
                        print("Деление на ноль невозможно")
                        return None
                    for j in range(k, row_count):
                        mat[i][j], mat[k][j] = mat[k][j], mat[i][j]
                    mat[i][-1], mat[k][-1] = mat[k][-1], mat[i][-1]
                    d = mat[k][k]

            for i in range(row_count):
                mat[k][i] = mat[k][i] / d
            mat[k][-1] = mat[k][-1] / d

            for i in range(k + 1, row_count):
                mat[i][-1] = mat[i][-1] - mat[k][-1] * mat[i][k]
                t = mat[i][k]
                for j in range(row_count):
                    mat[i][j] = mat[i][j] - mat[k][j] * t
        return mat

    def conversion_reverse(mat):
        ''' обратный ход метода Гаусса'''
        row_count = len(mat)
        res = [0] * row_count
        res[-1] = mat[-1][-1] / mat[-1][-2]
        for i in range(row_count - 2, -1, -1):
            c = mat[i][-1]
            for j in range(i + 1, row_count):
                c = c - mat[i][j] * res[j]
            res[i] = c / mat[i][i]
        return res

    mat_copy = deepcopy(matrix)
    conversion = conversion_elementary(mat_copy)
    if conversion == None:
        return "Решений нет"
    res = conversion_reverse(conversion)
    return np.round(res, 5)

def method_simple_iteration(matrix,
                            max_iter = 100,
                            accuracy = 1e-3):
    '''Решение систем линейных алгебраических уравнений
            методом простой итерации'''

    def matrix_norm(mat):
        ''' проверка на сходимость матрицы '''
        row_count = len(mat)
        for i in range(row_count):
            sum = 0
            for j in range(row_count - 1):
                if i != j:
                    sum += abs(mat[i][j])
            if sum >= abs(mat[i][i]):
                return False
        return True

    def conversion(mat):
        ''' преобразования от вида AX=B к X=CX+F '''
        row_count = len(mat)
        for k in range(row_count):
            d = mat[k][k]
            for i in range(row_count):
                if i != k and mat[k][i] != 0:
                    mat[k][i] = -mat[k][i] / d
            if mat[k][-1] != 0:
                mat[k][-1] = mat[k][-1] / d
            mat[k][k] = mat[k][-1]
        return mat

    def iteration_method(mat):
        row_count = len(mat)
        t = [0] * row_count
        for i in range(row_count):
            for j in range(row_count):
                if i != j:
                    t[i] = t[i] + mat[i][j] * mat[j][-1]
                else:
                    t[i] = t[i] + mat[i][j]
        return t

    mat_copy = deepcopy(matrix)
    convergence = True
    if not matrix_norm(mat_copy):
        convergence = False

    res = conversion(mat_copy)
    cur_iter = iteration_method(res)
    for n_iter in range(max_iter):
        prev_iter = cur_iter
        for u in range(len(res)):
            res[u][-1] = prev_iter[u]
        cur_iter = iteration_method(res)
        l_iter = len(cur_iter)
        delta_vec = [0] * l_iter
        for i in range(l_iter):
            delta_vec[i] = abs(cur_iter[i] - prev_iter[i])
        if max(delta_vec) < accuracy:
            return np.round(cur_iter, 5), convergence
    return "Решений нет", convergence


if __name__ == '__main__':
    matrix = [[1, 1, 1, 54],
            [1, -1, 0, 8],
            [0, -1, 4, 85]]

    display_matrix(matrix)

    print("Решение системы линейных уравнений с помощью:")
    print("метода Гаусса")
    res_list_Gauss = method_Gauss(matrix)
    print(res_list_Gauss)

    print("метода простой итерации")
    res_list_iter, convergence = method_simple_iteration(matrix)
    if not convergence:
        print("Матрица не сходится")
    print(res_list_iter)

    a = [row[:-1] for row in matrix]
    b = [row[-1] for row in matrix]
    correct_result = np.round(np.linalg.solve(a, b), 5)
    print(f"\nПоиск решения системы линейных уравнений, \n"
          f"используя функцию solve "
          f"из библиотеки numpy: {correct_result}")


# class UnitTest_SLAE(unittest.TestCase):
#     mat1 = [[1, 1, 1, 54], [1, -1, 0, 8], [0, -1, 4, 85]]
#     mat2 = [[2, 1, 1, 2], [1, -1, 0, -2], [3, -1, 2, 2]]
#     def test_Gauss(self):
#         self.assertTrue(np.allclose(method_Gauss(self.mat1), [19., 11., 24.]))
#         self.assertTrue(np.allclose(method_Gauss(self.mat2), [-1.,  1.,  3.]))
#
#     def test_simple_iteration(self):
#         self.assertEqual(method_simple_iteration(self.mat1), ("Решений нет", False))
#         self.assertTrue(method_simple_iteration(self.mat2), ([-1.00013,  1.00017,  2.99964], False))