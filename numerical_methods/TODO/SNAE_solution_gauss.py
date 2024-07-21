import unittest

import math

import numpy as np
from scipy.optimize import fsolve


def system(arg):
    """Исходная система"""
    x, y = arg
    r = [0] * 2
    r[0] = math.sin(y + 1) - x - 1.2
    r[1] = 2 * y + math.cos(x) - 2
    return r


def method_steepest_descent(sys, arg=None,
                            accuracy=1e-3, max_iter=100):
    """
    Решение системы нелинейных уравнений с помощью
        метода наискорейшего градиентного спуска
    """

    def target_function(sys_f, arg_vect):
        """ метод для поиска целевой функции """
        return sum([func ** 2 for func in sys_f(arg_vect)])

    def derivative1(sys_f, arg_vect, h_step, i):
        """ метод для поиска производной """
        e = [0] * len(arg_vect)
        e[i] = 1
        dif_f = target_function(sys_f, [arg_vect[index] - e[index] * h_step for index in range(len(e))])
        sum_f = target_function(sys_f, [arg_vect[index] + e[index] * h_step for index in range(len(e))])
        derivative = (sum_f - dif_f) / (2 * h_step)
        return derivative

    def grad(sys_f, arg_vect, h_step):
        """ метод для поиска градиента """
        u_grad = [0] * 2
        for i in range(len(arg_vect)):
            e = [0] * 2
            e[i] = 1
            dtarg = derivative1(sys_f, arg_vect, h_step, i)
            u_grad[i] += dtarg * e[i]
        return u_grad

    if arg is None:
        arg = [0, 0]

    iteration_count = 0

    while iteration_count < max_iter:
        target = target_function(sys, arg)
        h = 0.001
        while h < 1:
            u = grad(sys, arg, h)
            arg_min = arg[0] - h * u[0], arg[1] - h * u[1]
            h += 0.001
            if target_function(sys, arg_min) < target_function(sys, arg):
                arg = arg_min

        target_cur = target_function(sys, arg)

        if abs(target - target_cur) < accuracy:
            break

        iteration_count += 1

    return [round(argi, 5) for argi in arg]


def method_Newton(sys, arg=None,
                  accuracy=1e-3, max_iter=100):
    """
    Решение системы нелинейных уравнений с помощью
        метода Ньютона
    """

    def Jacobian_matrix(sys_f, arg_vect,
                        accuracy_jacobian=1e-3):
        """ метод для поиска матрицы Якоби """
        n = len(arg_vect)
        A_matrix = [[]] * n
        for i in range(n):
            row = [0] * n
            for j in range(n):
                x_accuracy = arg_vect.copy()
                x_accuracy[j] += accuracy_jacobian
                sum_f = sys_f(x_accuracy)
                dif_f = sys_f(arg_vect)
                derivative = (sum_f[i] - dif_f[i]) / accuracy_jacobian
                row[j] = derivative
            A_matrix[i] = row
        return A_matrix

    if arg is None:
        arg = method_steepest_descent(sys)

    iteration_count = 1

    while iteration_count < max_iter:
        L = np.array(sys(arg))
        A = np.array(Jacobian_matrix(sys, arg))
        delta_arg = np.linalg.solve(-A, L)
        arg = delta_arg + arg

        if all(abs(d_arg) < accuracy for d_arg in sys(arg)):
            break

        iteration_count += 1

    return [round(argi, 5) for argi in arg]


if __name__ == "__main__":
    print(f"Исходная система:\n"
          f"| sin(y + 1) - x = 1,2\n"
          f"| 2 * y + cos(x) = 2")

    print("Решение системы нелинейных уравнений с помощью:")
    print("метода наискорейшего градиентного спуска")
    res1 = method_steepest_descent(system)
    print(res1)

    print("метода Ньютона")
    res2 = method_Newton(system)
    print(res2)

    correct_result = fsolve(system, [0, 0])
    print(f"\nПоиск решения системы нелинейных уравнений, \n"
          f"используя функцию fsolve "
          f"из библиотеки scipy: {correct_result}")

# class UnitTest_SNAE(unittest.TestCase):
#     def test_method_steepest_descent(self):
#         self.assertEqual(method_steepest_descent(system), [-0.20164, 0.51015])
#
#     def test_method_Newton(self):
#         self.assertEqual(method_Newton(system), [-0.20184, 0.51015])
