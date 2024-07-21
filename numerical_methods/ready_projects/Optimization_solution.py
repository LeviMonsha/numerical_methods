import math
import unittest

import scipy.optimize
import scipy.optimize as sc


# y = x**4 + 8 * x**3 - 6 * x**2 - 72 * x + 90
# [1.5, 2]

def f(x):
    """Исходная функция"""
    return x ** 4 + 8 * x ** 3 - 6 * x ** 2 - 72 * x + 90

result_arr = [float] * 3
iterc_arr = [int] * 3

def second_way_dichotomy(func, segment, h = 1e-3,
                         accuracy = 1e-3, MAX_ITER = 100):
    """
    Поиск минимума функции с помощью
        метода второй способ дихотомии
    """
    c = None
    a, b = segment
    iteration_count = 1
    while iteration_count < MAX_ITER:
        c = (a + b) / 2
        if abs(b - a) < accuracy:
            break
        u = c + h
        v = c - h
        if func(u) < func(v):
            a = u
        else:
            b = v
        iteration_count += 1
    result_arr[0] = c
    iterc_arr[0] = iteration_count
    return round(c, 3)

def golden_ratio(func, segment,
                 accuracy = 1e-3, MAX_ITER = 100):
    """
    Поиск минимума функции с помощью
        метода золотого сечения
    """
    GOLDEN_RATIO = 1.618
    a, b = segment
    x = None
    iteration_count = 1
    while iteration_count < MAX_ITER:
        if abs(b - a) < accuracy:
            x = (a + b) / 2
            break
        u = a + (b - a) / GOLDEN_RATIO
        v = b - (b - a) / GOLDEN_RATIO
        if func(u) < func(v):
            a = v
        else:
            b = u
        iteration_count += 1
    result_arr[1] = x
    iterc_arr[1] = iteration_count
    return round(x, 3)

def quadratic_interpolation(func, segment, A,
                            h=1e-3, E=1e-3, MAX_ITER=100):
    """
    Поиск минимума функции с помощью
        метода квадратичной интерполяции
    """
    x = None
    a, b = segment
    t = None
    f_a = func(A)
    b = A + h
    f_b = func(b)
    if f_a < f_b:
        c = A - h
    else:
        c = A + 2 * h
    f_c = func(c)

    iteration_count = 1
    while MAX_ITER > iteration_count:
        t = find_new_point(func, A, b, c)
        f_t = func(t)
        if abs(func(A) - f_t) < E:
            x = min(max(t, a), b)
            break
        else:
            points = [A, b, c, t]
            max_f = max(f_t, func(A), func(b), f_c)
            for point in points:
                if func(point) == max_f:
                    if abs(point - t) <= h:
                        max_point = max([(_point, abs(t - _point)) for _point in points], key=lambda x: x[1])
                        points.remove(max_point[0])
                    else:
                        points.remove(point)
                    break
            points = sorted(points)
            A, b, c = points
            iteration_count += 1
    result_arr[2] = x
    iterc_arr[2] = iteration_count
    return round(x, 3)

def find_new_point(func, x1, x2, x3):
    y1, y2, y3 = func(x1), func(x2), func(x3)

    delta = (x1 - x2) * (x2 - x3) * (x3 - x1)
    a = ((x3 - x2) * y1 + (x1 - x3) * y2 + (x2 - x1) * y3) / delta
    b = ((x2**2 - x3**2) * y1 + (x3**2 - x1**2) * y2 + (x1**2 - x2**2) * y3) / delta
    c = (x2 * x3 * (x3 - x2) * y1 + x3 * x1 * (x1 - x3) * y2 + x1 * x2 * (x2 - x1) * y3) / delta
    if a == 0:
        raise ZeroDivisionError("Деление на ноль")
    x_min = -b / (2 * a)

    return x_min

def check_accuracy(res_arr):
    """Сравнение точности"""
    min_main = sc.minimize_scalar(f, bounds=(1.5, 2), method="bounded")
    correct_result = round(min_main.x, 5)
    print(f"Поиск минимума, используя функцию minimize_scalar "
          f"из библиотеки scipy: {round(correct_result, 5)}")
    print(f"Решение для методов:\n"
          f"второй способ дихотомии: {round(res_arr[0], 5)}\n"
          f"золотого сечения: {round(res_arr[1], 5)}\n"
          f"квадратичной интерполяции: {round(res_arr[2], 5)}")
    print(f"Их погрешность:\n"
          f"второй способ дихотомии: {round(correct_result - res_arr[0], 5)}\n"
          f"золотого сечения: {round(correct_result - res_arr[1], 5)}\n"
          f"квадратичной интерполяции: {round(correct_result - res_arr[2], 5)}")

def check_convergence(count_iter):
    """Сравнение сходимости"""
    print(f"Количество итераций для методов:\n"
          f"второй способ дихотомии: {count_iter[0]}\n"
          f"золотого сечения: {count_iter[1]}\n"
          f"квадратичной интерполяции: {count_iter[2]}")

if __name__ == "__main__":
    print("Исходная функция: x^4 + 8 * x^3 - 6 * x^2 - 72 * x + 90")
    a, b = [1.5, 2]
    print(f"Область определения функции: [{a}, {b}]")

    print("Поиск минимума функции с помощью:")

    print("Второго способа дихотомии")
    res1 = second_way_dichotomy(f, [a, b])
    print(res1)

    print("Золотого сечения")
    res2 = golden_ratio(f, [a, b])
    print(res2)

    print("Квадратичной интерполяции")
    res3 = quadratic_interpolation(f, [a, b], a)
    print(res3)

    print(f"~~~~~~~~~~~~~~~~~\n"
          f"Сравнение точности методов\n"
          f"~~~~~~~~~~~~~~~~~")
    check_accuracy(result_arr)
    print(f"~~~~~~~~~~~~~~~~~\n"
          f"Сравнение сходимости методов\n"
          f"~~~~~~~~~~~~~~~~~")
    check_convergence(iterc_arr)

# def f1(x):
#     return x ** 3 / 3 - 5 * x - x * math.log(x)
#
# class UnitTest_Optimisation(unittest.TestCase):
#     def test_method_second_way_dichotomy(self):
#         self.assertEqual(second_way_dichotomy(f, [1.5, 2]), 1.731)
#         self.assertEqual(second_way_dichotomy(f1, [1.5, 2]), 2.0)
#
#     def test_method_golden_ratio(self):
#         self.assertEqual(golden_ratio(f, [1.5, 2]), 1.691)
#         self.assertEqual(golden_ratio(f1, [1.5, 2]), 2.0)
#
#     def test_method_quadratic_interpolation_r(self):
#         self.assertEqual(quadratic_interpolation(f, [1.5, 2], 1.5), 1.732)
#         self.assertEqual(quadratic_interpolation(f1, [1.5, 2], 1.5), 2.0)