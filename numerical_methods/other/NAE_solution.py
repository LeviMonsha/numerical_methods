import unittest

import numpy as np
import math

import matplotlib.pyplot as plt

# f(x) = arccos(x) - V(1 - 0.3*x^3)

# def derivative2(x: float) -> float:
#     return -x / (math.sqrt(1 - x**2) * (1 - x**2)) + (360 * x - 27 * x**4) / (4 * math.sqrt(100 - 30 * x**3) * (10 - 3 * x**3))

def display_results(x, n) -> None:
    """вывод результата в консоль"""
    print(f"x{n} = {x}")

# Отделение корней на отрезке [a,b]
def Dx(i : float):
    """область определения функции проверяет значение"""
    return (i >= -1.0 and i <= 1.0) and (1 - 0.3 * i**3 >= 0)
def D_ab(a : float, b : float) -> tuple[float, float]:
    """область определения функции всего диапазона"""
    min_v = a
    max_v = b
    while True:
        min_v = round(min_v, 1)
        max_v = round(max_v, 1)
        if (Dx(min_v) and Dx(max_v)):
            a = round(min_v, 2)
            b = round(max_v, 2)
            return (a, b)
        if not Dx(min_v):
            min_v += 0.1
        if not Dx(max_v):
            max_v -= 0.1

def separation_roots(a : float, b : float, n : int) -> list[tuple[float, float]]:
    """выделение n диапазонов"""
    segment_length = (b - a) / n
    segments = []
    for i in range(n):
        start = a + i * segment_length
        end = a + (i + 1) * segment_length
        segments.append((start, end))
    return segments

def f(x: float) -> float:
    """исходная функция"""
    return math.acos(x) - math.sqrt(1 - 0.3 * x**3)

def derivative1(func, x: float, accuracy = 1e-3) -> float:
    """первая производная"""
    return (func(x + accuracy) - func(x)) / accuracy

# def f1(x: float) -> float:
#     """первая часть исходной функции"""
#     return math.acos(x)
#
# def f2(x: float) -> float:
#     """вторая часть исходной функции"""
#     return math.sqrt(1 - 0.3 * x**3)

# class Graph:
#     """класс для добавления и отрисовки графиков"""
#     def __init__(self):
#         self.number_gr = 1
#
#     def create_new_graph(self, func, color: str) -> None:
#         """создание графика"""
#         # определение отрезка построения графика
#         x = np.linspace(-10, 10, 100)
#         y = [func(n) if Dx(n) else None for n in x]
#         plt.plot(x, y, color=color,
#                  label=f"f{self.number_gr}(x)")
#         self.number_gr += 1
#
#     def update(self) -> None:
#         """отрисовка содержимого окна"""
#         plt.xlabel("x")  # подпись у горизонтальной оси х
#         plt.ylabel("y")  # подпись у вертикальной оси y
#         plt.legend()  # отображение легенды
#         plt.grid(True)  # отображение сетки на графике
#
#         plt.show()  # показать график

# Уточнение корней методом половинного деления
def method_half_division(func, a : float, b : float,
                         accuracy = 1e-3, MAX_ITER = 100) -> float or None:
    """решение нелинейного алгебраического уравнения методом половинного деления"""
    # проверка на разные знаки
    if func(a) * func(b) >= 0:
        return None

    iteration_count = 1
    while iteration_count < MAX_ITER:
        c = (a + b) / 2
        a_val = func(a)
        c_val = func(c)
        if abs(c_val) < accuracy:
            return round(c, 3)
        if a_val * c_val < 0:
            b = c
        else:
            a = c
    return None

# Уточнение корней методом хорд
def method_chord(func, a : float, b : float,
                 accuracy = 1e-3, MAX_ITER = 100) -> float or None:
    """решение нелинейного алгебраического уравнения методом хорд"""
    # проверка на разные знаки
    if func(a) * func(b) >= 0:
        return None

    if abs(derivative1(func, a)) < abs(func(a)) / (b - a):
        c1 = a
    else:
        c1 = b

    iteration_count = 1
    while iteration_count < MAX_ITER:
        a_val = func(a)
        b_val = func(b)
        c2 = a - (a_val*(a-b)) / (a_val - b_val)
        c_val = func(c2)
        if abs(c1 - c2) < accuracy:
            return round(c2, 3)
        if (a_val < 0 and c_val > 0) or (a_val > 0 and c_val < 0):
            b = c2
        else:
            a = c2
        c1 = c2
    return None

# Уточнение корней методом касательной
def method_tangent(func, a : float, b : float,
                   accuracy = 1e-3, MAX_ITER = 100) -> float or None:
    """решение нелинейного алгебраического уравнения методом касательных (Ньютона)"""
    # проверка на разные знаки
    if func(a) * func(b) >= 0:
        return None

    if abs(derivative1(func, a)) < abs(func(a)) / (b - a):
        c1 = a
    else:
        c1 = b

    iteration_count = 1
    while iteration_count < MAX_ITER:
        if abs(func(c1)) < accuracy:
            return round(c1, 3)
        c1 = c1 - func(c1) / derivative1(func, c1)
    return None

def iterate_method(func, equation, segments : list[tuple[float, float]]) -> list[float]:
    """вызов методов и их обработка"""
    root = None
    hv_root = 0
    result = []
    for a_segm, b_segm in segments:
        root_pr = root
        root = func(equation, a_segm, b_segm)
        if root != root_pr and not root is None:
            display_results(root, hv_root + 1)
            result.append(root)
            hv_root += 1
    if hv_root == 0:
        print("корней нет")
    return result


if __name__ == '__main__':
    # graph = Graph()
    # # создание графиков
    # graph.create_new_graph(f1, "red")
    # graph.create_new_graph(f2, "blue")
    # # обновление рисунка
    # graph.update()

    print("Исходное уравнение: acos(x) - sqrt(1 - 0.3 * x**3)")
    a, b = D_ab(-100, 100)
    print(f"Область определения функции: [{a}, {b}]")

    segments = separation_roots(a, b, 100)

    print("Решение нелинейных алгебраических уравнений:")

    print("методом половинного деления")
    print(f"~~~ результат ~~~")
    res1 = iterate_method(method_half_division, f, segments)

    print("методом хорд")
    print(f"~~~ результат ~~~")
    res2 = iterate_method(method_chord, f, segments)

    print("методом касательных")
    print(f"~~~ результат ~~~")
    res3 = iterate_method(method_tangent, f, segments)

def f2(x: float) -> float:
    """исходная функция"""
    return x**2 - 2*x - 5

class UnitTest_NAU(unittest.TestCase):
    def test_method_half_division(self):
        self.assertEqual(iterate_method(method_chord, f2, separation_roots(-100.0, 100.0, 100)), [-1.449, 3.449])
        self.assertRaises(ValueError, iterate_method, method_chord, f, separation_roots(-100.0, 100.0, 100))
        self.assertEqual(iterate_method(method_chord, f, separation_roots(-1.0, 1.0, 100)), [0.563])

    def test_method_chord(self):
        self.assertEqual(iterate_method(method_chord, f2, separation_roots(-100.0, 100.0, 100)), [-1.449, 3.449])
        self.assertRaises(ValueError, iterate_method, method_chord, f, separation_roots(-100.0, 100.0, 100))
        self.assertEqual(iterate_method(method_chord, f, separation_roots(-1.0, 1.0, 100)), [0.563])

    def test_method_tangent(self):
        self.assertEqual(iterate_method(method_tangent, f2, separation_roots(-100.0, 100.0, 100)), [-1.449, 3.449])
        self.assertRaises(ValueError, iterate_method, method_tangent, f, separation_roots(-100.0, 100.0, 100))
        self.assertEqual(iterate_method(method_tangent, f, separation_roots(-1.0, 1.0, 100)), [0.563])