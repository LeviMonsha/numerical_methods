import math
import unittest

from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

# Создание сетки точек для построения графика
x = np.linspace(-50, 50, 100)
y = np.linspace(-50, 50, 100)
X, Y = np.meshgrid(x, y)


def r1(x, y):
    return np.sin(y + 1) - x - 1.2


def r2(x, y):
    return 2 * y + np.cos(x) - 2


# Вычисление значений функций r1 и r2 для каждой точки сетки
Z1 = r1(X, Y)
Z2 = r2(X, Y)
# Построение графика для r1
plt.figure(figsize=(10, 6))


def system(arg):
    '''Исходная система'''
    x, y = arg
    r = [0] * 2
    r[0] = math.sin(y + 1) - x - 1.2
    r[1] = 2 * y + math.cos(x) - 2
    return r

def method_steepest_descent(sys, arg_vect=None, h=0.5,
                            accuracy=1e-3, MAX_ITER=100):
    '''
    Решение системы нелинейных уравнений с помощью
        метода наискорейшего градиентного спуска
    '''

    # ~~~~~~~~~~~~~~~~~~
    if arg_vect is None:
        arg_vect = [0, 0]

    def target_func(sys, arg_vect):
        target = 0
        for func in sys(arg_vect):
            target += func ** 2
        return target

    def derivative1(sys, arg_vect, h, i):
        e = [0.0] * len(arg_vect)
        e[i] = 1
        dif_f = target_func(sys, [arg_vect[index] - e[index] * h for index in range(len(e))])
        sum_f = target_func(sys, [arg_vect[index] + e[index] * h for index in range(len(e))])
        der = (sum_f - dif_f) / (2 * h)
        return der

    def grad(sys, arg_vect, h):
        u = [0.0] * 2
        for i in range(len(arg_vect)):
            e = [0.0] * 2
            e[i] = 1
            dtarg = derivative1(sys, arg_vect, h, i)
            u[i] += dtarg * e[i]
        return u

    # ~~~~~~~~~~~~~~~~~~

    if h <= 0: return None

    arg_vect_cur = [0, 0]
    target = 0
    iteration_count = 0

    while iteration_count < MAX_ITER:

        target_cur = target_func(sys, arg_vect)
        u = grad(sys, arg_vect, h)
        arg_vect_cur = [arg_vect[i] - h * u[i] for i in range(len(u))]

        plt.scatter(arg_vect_cur[0], arg_vect_cur[1], c="red")

        if abs(target - target_cur) < accuracy:
            break

        if sys(arg_vect_cur) >= sys(arg_vect):
            h *= 0.5

        arg_vect = arg_vect_cur
        target = target_cur
        iteration_count += 1

    return [round(arg, 3) for arg in arg_vect_cur]


def method_Newton(sys, arg_vect=None,
                  accuracy=1e-3, MAX_ITER=100):
    '''
    Решение системы нелинейных уравнений с помощью
        метода Ньютона
    '''

    # ~~~~~~~~~~~~~~~~~~
    if arg_vect is None:
        arg_vect = [0, 0]

    def inv_A_matrix(A):
        det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        if det == 0:
            raise ValueError("Matrix is singular")
        inv_det = 1 / det
        A_inv = [[A[1][1] * inv_det, -A[0][1] * inv_det],
                 [-A[1][0] * inv_det, A[0][0] * inv_det]]
        return A_inv

    def jacobian_matrix(sys, arg_vect, accuracy=1e-3):
        n = len(arg_vect)
        J = []
        for i in range(n):
            row = []
            for j in range(n):
                x_plus_h = arg_vect.copy()
                x_plus_h[j] += accuracy
                F_plus_h = sys(x_plus_h)
                F_minus = sys(arg_vect)
                partial_derivative = (F_plus_h[i] - F_minus[i]) / accuracy
                row.append(partial_derivative)
            J.append(row)
        return J

    def matrix_vector_product(X_vect, A, L):
        return [X_vect[0] - 0.1 * (A[0][0] * L[0] + A[0][1] * L[1]),
                X_vect[1] - 0.1 * (A[1][0] * L[0] + A[1][1] * L[1])]

    # ~~~~~~~~~~~~~~~~~~

    iteration_count = 1
    X_vect_cur = arg_vect

    while iteration_count < MAX_ITER:
        X_vect = X_vect_cur

        L = sys(arg_vect)
        A = jacobian_matrix(sys, arg_vect)
        inv_A = inv_A_matrix(A)
        X_vect_cur = matrix_vector_product(X_vect_cur, inv_A, L)

        plt.scatter(arg_vect[0], arg_vect[1], c="blue")

        delta_X_vect = [X_vect_cur[i] - X_vect[i] for i in range(len(X_vect))]

        if abs((delta_X_vect[1] + delta_X_vect[0]) / 2) < accuracy:
            break

        arg_vect = [delta_X_vect[i] + arg_vect[i] for i in range(len(arg_vect))]
        iteration_count += 1

    return [round(arg, 3) for arg in arg_vect]


def plotsystem():
    plt.contour(X, Y, Z1, levels=[0], colors='b', linestyles='-')
    plt.contour(X, Y, Z2, levels=[0], colors='r', linestyles='-')
    plt.title('r1(x, y) = 0')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # arg = [0] * 2
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

    x = fsolve(system, [0, 0])
    print(f"Проверенный ответ: {x}")

    plotsystem()


def system1(arg):
    x, y = arg
    r = [0] * 2
    r[0] = math.cos(x - 1) + y - 0.8
    r[1] = x - math.cos(y) - 2
    return r

# class UnitTest_SNAE(unittest.TestCase):
#     def test_method_steepest_descent(self):
#         self.assertEqual(method_steepest_descent(system), [-0.25, 0.517])
#         self.assertEqual(method_steepest_descent(system1), [2.643, 0.863])
# #
#     def test_method_Newton(self):
#         self.assertEqual(method_Newton(system), [-0.173, 0.463])
#         self.assertEqual(method_Newton(system1), [2.635, 0.861])
