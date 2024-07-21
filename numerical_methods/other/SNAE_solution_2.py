import unittest

import math
from scipy.optimize import fsolve


def system(arg):
    """Исходная система"""
    x, y = arg
    r = [0] * 2
    r[0] = math.sin(y + 1) - x - 1.2
    r[1] = 2 * y + math.cos(x) - 2
    return r


def method_steepest_descent(sys, arg=None, h=0.21,
                            accuracy=1e-3, max_iter=100):
    """
    Решение системы нелинейных уравнений с помощью
        метода наискорейшего градиентного спуска
    """

    def target_function(sys_f, arg_vect):
        return sum([func ** 2 for func in sys_f(arg_vect)])

    def derivative1(sys_f, arg_vect, h_step, i):
        e = [0] * len(arg_vect)
        e[i] = 1
        dif_f = target_function(sys_f, [arg_vect[index] - e[index] * h_step for index in range(len(e))])
        sum_f = target_function(sys_f, [arg_vect[index] + e[index] * h_step for index in range(len(e))])
        derivative = (sum_f - dif_f) / (2 * h_step)
        return derivative

    def grad(sys_f, arg_vect, h_step):
        u_grad = [0] * 2
        for i in range(len(arg_vect)):
            e = [0] * 2
            e[i] = 1
            dtarg = derivative1(sys_f, arg_vect, h_step, i)
            u_grad[i] += dtarg * e[i]
        return u_grad

    if arg is None:
        arg = [0, 0]

    if h <= 0:
        raise ValueError("Шаг не может быть меньше либо равен 0")

    arg_cur = [0] * 2
    target = 0
    iteration_count = 0

    while iteration_count < max_iter:
        target_cur = target_function(sys, arg)
        u = grad(sys, arg, h)
        arg_cur = [arg[i] - h * u[i] for i in range(len(u))]

        if abs(target - target_cur) < accuracy:
            break

        if sys(arg_cur) >= sys(arg):
            h *= 0.5

        arg = arg_cur
        target = target_cur
        iteration_count += 1

    return [round(arg, 3) for arg in arg_cur]


def method_Newton(sys, arg=None,
                  accuracy=1e-3, max_iter=100):
    """
    Решение системы нелинейных уравнений с помощью
        метода Ньютона
    """

    def inverse_matrix(matrix):
        """ Поиск обратной мартицы
        для матрицы размером 2x2 """
        determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        if determinant == 0:
            raise ValueError("Матрица вырожденная")
        inv_det = 1 / determinant
        inv_matrix = [[matrix[1][1] * inv_det, -matrix[0][1] * inv_det],
                      [-matrix[1][0] * inv_det, matrix[0][0] * inv_det]]
        return inv_matrix

    def Jacobian_matrix(sys_f, arg_vect, accuracy_jacobian=1e-3):
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
        L = sys(arg)
        A = Jacobian_matrix(sys, arg)
        inv_A = inverse_matrix(A)
        arg_cur = [arg[i] - (inv_A[i][0] * L[0] + inv_A[i][1] * L[1])
                   for i in range(len(L))]
        delta_arg = [arg_cur[i] - arg[i] for i in range(len(arg))]

        if all([abs(delta) < accuracy for delta in delta_arg]):
            break

        arg = [delta_arg[i] + arg[i] for i in range(len(arg))]
        iteration_count += 1

    return [round(arg, 3) for arg in arg]


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

# def system1(arg):
#     x, y = arg
#     r = [0] * 2
#     r[0] = math.cos(x - 1) + y - 0.8
#     r[1] = x - math.cos(y) - 2
#     return r
#
# class UnitTest_SNAE(unittest.TestCase):
#     def test_method_steepest_descent(self):
#         self.assertEqual(method_steepest_descent(system, h=0.5), [-0.25, 0.517])
#         self.assertEqual(method_steepest_descent(system, h=0.21), [-0.203, 0.531])
#         self.assertEqual(method_steepest_descent(system1, h=0.5), [2.643, 0.863])
#
#     def test_method_Newton(self):
#         self.assertEqual(method_Newton(system), [-0.202, 0.51])
#         self.assertEqual(method_Newton(system), [-0.202, 0.51])
#         self.assertEqual(method_Newton(system1), [2.644, 0.873])
