import math
import unittest

from sympy import sign
from scipy.integrate import fixed_quad

def function(x):
    return 2 * x**2 - 2 + x**0.5

def diff_n(x, n, h = 1e-3):
    if n == 0:
        return function(x)
    else:
        return (diff_n(x + h, n - 1, h) - diff_n(x - h, n - 1, h)) / (2 * h)

def quadrature_formula_rectangles(func, segm, interval_c, mode=1):
    """
    Решение интеграла с помощью
        квадратурной формулы прямоугольников

    mode нужен для выбора метода прямоугольников:
         1 - левосторонний
         2 - правосторонний
         3 - средний
    """

    a, b = segm
    h = (b - a) / interval_c
    Q = 0

    if mode == 2:
        x = a + h
        b += h
    else:
        x = a

    while(x < b):
        if mode == 3:
            y = (func(x) + func(x + h)) / 2
        else:
            y = func(x)
        Q += y * h
        x += h

    Ea = a + 0.01
    Eb = b - 0.01
    f_der2 = diff_n(Ea, 2)
    M2 = abs(f_der2)
    sg = sign(f_der2)

    while Ea < Eb:
        Ea += 0.01
        f_der2 = diff_n(Ea, 2)
        if abs(f_der2) > M2:
            M2 = abs(f_der2)
            sg = sign(f_der2)

    M2 *= sg
    R = (h**2 * (b - a) * M2) / 24

    return [round(Q, 3), round(R, 5)]


def quadrature_formula_trapezoids(func, segm, interval_c):
    """
    Решение интеграла с помощью
        квадратурной формулы трапеций
    """

    a, b = segm
    h = (b - a) / interval_c
    x = a
    Q = 0

    while (x < b):
        yi1 = func(x)
        yi2 = func(x + h)
        Q += (yi1 + yi2) * h / 2
        x += h

    Ea = a + 0.01
    Eb = b - 0.01
    f_der2 = diff_n(Ea, 2)
    M2 = abs(f_der2)
    sg = sign(f_der2)

    while Ea < Eb:
        Ea += 0.01
        f_der2 = diff_n(Ea, 2)
        if abs(f_der2) > M2:
            M2 = abs(f_der2)
            sg = sign(f_der2)

    M2 *= sg
    R = -(h**2 * (b - a) * M2) / 12

    return [round(Q, 3), round(R, 5)]


def quadrature_formula_parabola(func, segm, interval_c):
    """
    Решение интеграла с помощью
        квадратурной формулы парабол (Симпсона)
    """

    a, b = segm
    h = (b - a) / interval_c
    x = a + h
    Q = 0
    i = 1

    while (x < b):
        Q += (4 * func(x) if i % 2 == 1
              else 2 * func(x))
        x += h
        i += 1

    Q = (h / 3) * (func(a) + Q + func(b))

    Ea = a + 0.01
    Eb = b - 0.01
    f_der4 = diff_n(Ea, 4)
    M2 = abs(f_der4)
    sg = sign(f_der4)

    while Ea < Eb:
        Ea += 0.01
        f_der4 = diff_n(Ea, 4)
        if abs(f_der4) > M2:
            M2 = abs(f_der4)
            sg = sign(f_der4)

    M2 *= sg
    R = -(h**4 * (b - a) * M2) / 180

    return [round(Q, 3), round(R, 5)]


if __name__ == "__main__":
    print(f"Исходный интеграл:\n"
          f"5/2|(2x^2 - 2 + Vx)dx")

    segment = [2, 5]
    interval_count = 6

    print("Решение интегралов с помощью:")
    print("квадратурной формулы прямоугольников (левосторонний)")
    res1 = quadrature_formula_rectangles(function, segment, interval_count, 1)
    print(res1[0])
    print(f"Погрешность метода: {res1[1]}")

    print("квадратурной формулы трапеций")
    res2 = quadrature_formula_trapezoids(function, segment, interval_count)
    print(res2[0])
    print(f"Погрешность метода: {res2[1]}")

    print("квадратурной формулы парабол")
    res3 = quadrature_formula_parabola(function, segment, interval_count)
    print(res3[0])
    print(f"Погрешность метода: {res3[1]}")

    correct_result = fixed_quad(function, segment[0], segment[1], n=interval_count)[0]
    print(f"\nПоиск решения интеграла, \n"
          f"используя функцию fixed_quad \n"
          f"из библиотеки scipy: {correct_result}")

    
# class UnitTest_Integral(unittest.TestCase):
#     def test_quadrature_formula_rectangles(self):
#         self.assertEqual(quadrature_formula_rectangles(function, [2, 5], 6), [67.11, 0.12430])
#
#     def test_quadrature_formula_trapezoids(self):
#         self.assertEqual(quadrature_formula_trapezoids(function, [2, 5], 6), [77.815, -0.24860])
#
#     def test_quadrature_formula_parabola(self):
#         self.assertEqual(quadrature_formula_parabola(function, [2, 5], 6), [77.568, 0.00009])