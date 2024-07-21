import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def task1():
    # создаем функцию для решения правой части уравнения
    def dtdx(t, x):
        return -t * x - 5

    x = np.linspace(-2, 4, 20)  # вектор моментов времени
    t0 = 1  # начальное значение
    t = odeint(dtdx, t0, x)
    t = np.array(t).flatten()  # преобразование массива
    plt.plot(x, t, "-sr", linewidth=3)  # построение графика
    plt.show()

def task2():
    # создаем функцию для решения правых частей уравнений
    def f(y, x):
        y1, y2 = y # имена функций
        return [y2, -0.2 * y2 - y1]

    # решение системы ОДУ
    x = np.linspace(0, 20, 41)
    y0 = [0, 1]
    w = odeint(f, y0, x)
    print(w)

    y1 = w[:, 0]
    y2 = w[:, 1]
    fig = plt.figure(facecolor="white")
    plt.plot(x, y1, "-o", linewidth=2)
    plt.ylabel("t")
    plt.xlabel("x")

    x = np.linspace(0, 20, 201)
    y0 = [0, 1]
    [y1, y2] = odeint(f, y0, x, full_output=False).T
    # fig = plt.figure(facecolor="white")
    plt.plot(y1, y2, linewidth=2)

    plt.show()

if __name__ == "__main__":
    # task1()
    task2()