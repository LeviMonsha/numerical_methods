import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize

# ОПРЕДЕЛЕНИЕ МИНИМАЛЬНЫХ ЗНАЧЕНИЙ С ПОМОЩЬЮ
# БИБЛИОТЕЧНЫХ ФУНКЦИЙ

def f1(x):
    # определение функции для минимизации
    return (x-5)*(x+3)*(x+4)*(x-10)

def f(x):
    # определение функции для минимизации
    return -20 * x**2 + 7000 * x - 300000

def var1():
    """Поиск минимума одномерной функции с помощью minimize_scalar()"""

    # Общий вид функции minimize_scalar() следующий:
    # scipy.optimize.minimize_scalar(fun, bracket=None, bounds=None, args=(),
    # method=None, tol=None, options=None)

    a, b = -5, 10
    result = optimize.minimize_scalar(f, bounds=(a, b), method="bounded") # находим минимум функции
    print(result)
    x = np.linspace(-10, 15, 100)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(x, f(x), label=r"$f(x)$")
    ax.scatter(result.x, result.fun, color="red",label="найденный минимум")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend()
    plt.show()

def var2():
    """функция minimize()"""
    # scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None,
    # hess=None, hessp=None, bounds=None, constraints=(), tol=None,
    # callback=None, options=None)

    result = optimize.minimize(f, 0) # находим минимум функции
    print(result)
    x = np.linspace(-10, 15, 100)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, f(x), label=r"$f(x)$")
    ax.scatter(result.x, result.fun, color="red", label="найденный минимум")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend()
    plt.show()

# ОПРЕДЕЛЕНИЕ МИНИМАЛЬНЫХ ЗНАЧЕНИЙ
# ОДНОМЕРОНЫХ ФУНКЦИЙ МЕТОДОМ ДИХОТОМИИ

def task1():
    # задаем начальное значение
    a, b = -5, 0
    eps = 0.001
    k = 0

    while b-a > eps:
        k = k + 1
        c = (a + b) / 2
        u = (a + c) / 2
        v = (c + b) / 2
        if (f(a) >= f(u)) and (f(u) < f(c)):
            x1, x2 = a, c
        if (f(u) >= f(c)) and (f(c) < f(v)):
            x1, x2 = u, v
        if (f(c) >= f(v)) and (f(v) < f(b)):
            x1, x2 = c, b
        a, b = x1, x2

    # выводим результат
    x_min = (a+b)/2
    y_min = f(x_min)
    print("x min = ", x_min, " y min = ", y_min, " количество итераций = ", k)
    x = np.linspace(-10, 15, 100)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, f(x), label=r"$f(x)$")
    ax.scatter(x_min, y_min, color="red", label="найденный минимум")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    var1()

    # алгоритм Брента, Bounded, способ «золотого сечения»

    # методы градиентного спуска, Нелдера-Мида, Пауэлла и Ньютона
    # якобиан и гессианы функции

    # первый способ дихотомии, т.е. разделения заданной отрезка на две части.