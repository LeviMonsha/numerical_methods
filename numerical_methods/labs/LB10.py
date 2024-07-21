from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import CubicSpline

from scipy.interpolate import splev, splrep



def task1():
    # задание табличной функции
    x = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.2]
    y = [-3.5, -4.8, -2.1, 0.2, 0.9, 2.3, 3.7]

    # вывод графика табличной функции маркерами
    plt.plot(x, y, 'o', label="табличная функция")
    # задание промежуточных точек для интерполирования
    xi = np.arange(x[0], x[len(x) - 1], 0.01)
    f_nearest = interp1d(x, y, kind="nearest")
    f_linear = interp1d(x, y, kind="linear")
    f_cubic = interp1d(x, y, kind="cubic")
    plt.plot(xi, f_nearest(xi), label="по соседним элементам")
    plt.plot(xi, f_linear(xi), label="линейная")
    plt.plot(xi, f_cubic(xi), label="кубические сплайны")
    plt.legend()
    plt.show()

def task2():
    # задание табличной функции
    x = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.2]
    y = [-3.5, -4.8, -2.1, 0.2, 0.9, 2.3, 3.7]

    # вывод графика табличной функции маркерами
    plt.plot(x, y, 'o', label="табличная функция")
    # задание промежуточных точек для интерполирования
    xi = np.arange(x[0], x[len(x) - 1], 0.01)
    yi = CubicSpline(x, y)
    plt.plot(xi, yi(xi), label="S")
    plt.legend()
    plt.show()

def task3():
    x = np.linspace(0, 21, 10)
    y = np.sin(x)
    plt.plot(x, y, 'o', label="табличная функция")

    # преобразование Фурье для интерполяции
    spl = splrep(x, y)
    xi = np.arange(x[0], x[len(x) - 1], 0.1)
    yi = splev(xi, spl)
    plt.plot(xi, yi)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # task1()
    # task2()
    task3()