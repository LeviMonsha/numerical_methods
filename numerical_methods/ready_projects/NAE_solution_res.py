import math

# f(x) = arccos(x) - V(1 - 0.3*x^3)

# def derivative2(x: float) -> float:
#     return -x / (math.sqrt(1 - x**2) * (1 - x**2)) + (360 * x - 27 * x**4) / (4 * math.sqrt(100 - 30 * x**3) * (10 - 3 * x**3))

# def display_results(arr : [float]) -> None:
#     """вывод результата в консоль"""
#     if len(arr) == 0:
#         print("корней нет")
#         return
#     for i in range(0, len(arr)):
#         print(f"x{i + 1} = {arr[i]}")

# Отделение корней на отрезке [a,b]
def Dx(i):
    """область определения функции проверяет значение"""
    return (i >= -1.0 and i <= 1.0) and (1 - 0.3 * i**3 >= 0)
def D_ab(a, b):
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

def separation_roots(a, b, n):
    """выделение n диапазонов"""
    segment_length = (b - a) / n
    segments = []
    for i in range(n):
        start = a + i * segment_length
        end = a + (i + 1) * segment_length
        segments.append((start, end))
    return segments

def f(x):
    """исходная функция"""
    return math.acos(x) - math.sqrt(1 - 0.3 * x**3)

def derivative1(func, x, accuracy = 1e-3):
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
def method_half_division(func, segments,
                         accuracy = 1e-3, MAX_ITER = 100):
    """решение нелинейного алгебраического уравнения методом половинного деления"""

    root_count = 0
    for a_segm, b_segm in segments:
        # проверка на разные знаки
        if func(a_segm) * func(b_segm) > 0:
            continue

        iteration_count = 1
        while iteration_count < MAX_ITER:
            c = (a_segm + b_segm) / 2
            a_val = func(a_segm)
            c_val = func(c)
            if abs(c_val) < accuracy:
                root_count += 1
                print(f"x{root_count} = {round(c, 3)}")
                break
            if a_val * c_val > 0:
                a_segm = c
            else:
                b_segm = c
            iteration_count += 1

# Уточнение корней методом хорд
def method_chord(func, segments,
                 accuracy = 1e-3, MAX_ITER = 100):
    """решение нелинейного алгебраического уравнения методом хорд"""
    a_gl = segments[0][1]
    b_gl = segments[-1][0]

    if abs(derivative1(func, a_gl)) < abs(func(a_gl)) / (b_gl - a_gl):
        c1 = a_gl
    else:
        c1 = b_gl

    root_count = 0
    for a_segm, b_segm in segments:
        # проверка на разные знаки
        if func(a_segm) * func(b_segm) > 0:
            continue

        iteration_count = 1
        while iteration_count < MAX_ITER:
            a_val = func(a_segm)
            b_val = func(b_segm)
            c2 = a_segm - (a_val * (a_segm - b_segm)) / (a_val - b_val)
            c_val = func(c2)
            if abs(c1 - c2) < accuracy:
                root_count += 1
                print(f"x{root_count} = {round(c2, 3)}")
                break
            if a_val * c_val > 0:
                a_segm = c2
            else:
                b_segm = c2
            c1 = c2
            iteration_count += 1

# Уточнение корней методом касательной
def method_tangent(func, segments,
                   accuracy = 1e-3, MAX_ITER = 100):
    """решение нелинейного алгебраического уравнения методом касательных (Ньютона)"""

    root_count = 0
    for a_segm, b_segm in segments:
        # проверка на разные знаки
        if func(a_segm) * func(b_segm) > 0:
            continue

        if abs(derivative1(func, a_segm)) < abs(func(a_segm)) / (b_segm - a_segm):
            c = a_segm
        else:
            c = b_segm

        iteration_count = 1
        while iteration_count < MAX_ITER:
            c_prev = c
            c = c - func(c) / derivative1(func, c)
            if abs(c - c_prev) < accuracy:
                root_count += 1
                print(f"x{root_count} = {round(c, 3)}")
                break
            iteration_count += 1
    if root_count == 0:
        print("корней нет")

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
    res1 = method_half_division(f, segments)

    print("методом хорд")
    print(f"~~~ результат ~~~")
    res2 = method_chord(f, segments)

    print("методом касательных")
    print(f"~~~ результат ~~~")
    res3 = method_tangent(f, segments)

def f2(x: float) -> float:
    """исходная функция"""
    return x**2 - 2*x - 5

# class UnitTest_NAU(unittest.TestCase):
#     def test_method_half_division(self):
#         self.assertEqual(method_chord(f2, separation_roots(-100.0, 100.0, 100)), [-1.449, 3.449])
#         self.assertRaises(ValueError, method_chord, f, separation_roots(-100.0, 100.0, 100))
#         self.assertEqual(method_chord(f, separation_roots(-1.0, 1.0, 100)), [0.562])
#
#     def test_method_chord(self):
#         self.assertEqual(method_chord(f2, separation_roots(-100.0, 100.0, 100)), [-1.449, 3.449])
#         self.assertRaises(ValueError, method_chord, f, separation_roots(-100.0, 100.0, 100))
#         self.assertEqual(method_chord(f, separation_roots(-1.0, 1.0, 100)), [0.563])
#
#     def test_method_tangent(self):
#         self.assertEqual(method_tangent(f2, separation_roots(-100.0, 100.0, 100)), [-1.449, 3.449])
#         self.assertRaises(ValueError, method_tangent, f, separation_roots(-100.0, 100.0, 100))
#         self.assertEqual(method_tangent(f, separation_roots(-1.0, 1.0, 100)), [0.563])