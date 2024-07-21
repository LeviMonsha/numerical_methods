import math
import unittest

from scipy.integrate import fixed_quad

def integral(x):
    return 2 * x**2 - 2 + x**0.5

def quadrature_formula_rectangles(integral_f, segm, interval_c, mode=1):
    """
    Решение интеграла с помощью
        квадратурной формулы прямоугольников
    """

    h = (segm[1] - segm[0]) / interval_c
    res = 0

    if mode == 2:
        x, x_end = segm[0] + h, segm[1] + h
    else:
        x, x_end = segm

    while(x < x_end):
        if mode == 3:
            y = (integral_f(x) + integral_f(x+h)) / 2
        else:
            y = integral_f(x)
        res += y * h

        x += h

    return round(res, 5)


def quadrature_formula_trapezoids(integral_f, segm, interval_c):
    """
    Решение интеграла с помощью
        квадратурной формулы трапеций
    """

    h = (segm[1] - segm[0]) / interval_c
    x = segm[0]
    x_end = segm[1]
    res = 0

    while (x < x_end):
        yi1 = integral_f(x)
        yi2 = integral_f(x + h)
        res += (yi1 + yi2) * h / 2

        x += h

    return round(res, 5)


def quadrature_formula_parabola(integral_f, segm, interval_c):
    """
    Решение интеграла с помощью
        квадратурной формулы парабол (Симпсона)
    """

    if interval_c % 2 != 0: interval_c *= 2
    h = (segm[1] - segm[0]) / interval_c
    x = segm[0]
    x_end = segm[1]
    res = 0
    i = 0

    while (x < x_end):
        y = integral_f(x)
        k = 2 + 2 * (i % 2)
        res += (k * y) * h / 3

        x += h
        i += 1

    return round(res, 5)


if __name__ == "__main__":
    print(f"Исходная система:\n"
          f"5/2|(2x^2 - 2 + Vx)dx")

    segment = [2, 5]
    interval_count = 6

    print("Решение интегралов с помощью:")
    print("квадратурной формулы прямоугольников (левосторонний)")
    res1 = quadrature_formula_rectangles(integral, segment, interval_count, 1)
    print(res1)

    print("квадратурной формулы трапеций")
    res2 = quadrature_formula_trapezoids(integral, segment, interval_count)
    print(res2)

    print("квадратурной формулы парабол")
    res2 = quadrature_formula_parabola(integral, segment, interval_count)
    print(res2)

    correct_result = fixed_quad(integral, segment[0], segment[1], n=interval_count)[0]
    print(f"\nПоиск решения интеграла, \n"
          f"используя функцию fixed_quad \n"
          f"из библиотеки scipy: {correct_result}")



    
# class UnitTest_Integral(unittest.TestCase):
#     def test_quadrature_formula_rectangles(self):
#         self.assertEqual(quadrature_formula_rectangles(integral, [2, 5], 6), 67.10978)
#
#     def test_quadrature_formula_trapezoids(self):
#         self.assertEqual(quadrature_formula_trapezoids(integral, [2, 5], 6), 77.81524)
#
#     def test_quadrature_formula_parabola(self):
#         self.assertEqual(quadrature_formula_parabola(integral, [2, 5], 6), 70.43095)