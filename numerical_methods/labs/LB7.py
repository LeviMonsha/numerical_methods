import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize

def f(x):
    return np.sqrt((x[0] - 3)**2 + (x[1] - 2)**2)

# импортируем функцию из библиотеки
from scipy.optimize import fsolve
x=[0]*2 # объявление массива аргументов
def sys_SNAU(x):
    r=[0]*2 # объявление массива
    r[0]=0.1*x[0]*x[0]+x[0]+0.2*x[1]*x[1]-0.3 # запись 1-го уравнения системы
    r[1]=0.2*x[0]*x[0]+x[1]-0.1*x[0]*x[1]-0.7 # запись 2-го уравнения системы
    return r # запись возвращаемого значения (окончание функции)

def init_task1():
    import numpy as np
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy import optimize

    def f(x):
        return np.sqrt((x[0] - 3) ** 4 + (x[1] - 2) ** 2)

    # оптимизация
    result = optimize.minimize(f, [0, 0])
    print(result)
    x_min = result.x
    print(x_min)

    # вывод функции
    fig = plt.figure(figsize=(10, 10))
    axes = fig.add_subplot(111, projection='3d')
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-3, 3, 50)
    xg, yg = np.meshgrid(x, y)

    zg = np.zeros_like(xg)
    for i in range(len(x)):
        for j in range(len(y)):
            zg[j][i] = f([xg[j][i], yg[j][i]])

    axes.plot_surface(xg, yg, zg, alpha=0.5, cmap="Blues")

    # точка минимума
    axes.scatter(x_min[0], x_min[1], f(x_min), c="red", marker="^", s=500)
    axes.set_xlabel("Ось X")
    axes.set_ylabel("Ось Y")
    axes.set_zlabel("Ось Z")

    plt.show()

def init_task2():
    x = fsolve(sys_SNAU, (0.25, 0.75))
    print(x)

def init_task3():
    # Ввод начального приближения
    import numpy

    x0 = [[0.25], [0.75]]
    # Вычисление значений функций в начальном приближении
    f1 = 0.1 * x0[0][0] * x0[0][0] + x0[0][0] + 0.2 * x0[1][0] * x0[1][0] - 0.3
    f2 = 0.2 * x0[0][0] * x0[0][0] + x0[1][0] - 0.1 * x0[0][0] * x0[1][0] - 0.7
    # Запишем вектор значений функций в начальном приближении
    f0 = numpy.array([[f1], [f2]])
    # Вычисление матрицы Якоби, составленной из частных производных
    M = numpy.array([[0.2 * x0[0][0] + 1, 0.4 * x0[1][0]], [0.4 * x0[0][0] - 0.1 * x0[1][0],
                                                            1 - 0.1 * x0[0][0]]])
    # Вычисление определителя матрицы Якоби
    w2 = numpy.linalg.det(M)
    # Вычисление новых значение х после первой итерации
    x = x0 - numpy.linalg.inv(M).dot(f0)
    # Задание точности вычислений
    E = 0.0001
    # Цикл вычисления решения системы нелинейных уравнений до тех пор, пока поправки | х - х0 | > e:
    while (abs(x[0][0] - x0[0][0]) > E) and (abs(x[1][0] - x0[1][0]) > E):
        # переобозначение
        x0[0][0] = x[0][0]
        x0[1][0] = x[1][0]
        # вычисление значений функций в начальном приближении
        f1 = 0.1 * x0[0][0] * x0[0][0] + x0[0][0] + 0.2 * x0[1][0] * x0[1][0] - 0.3
        f2 = 0.2 * x0[0][0] * x0[0][0] + x0[1][0] - 0.1 * x0[0][0] * x0[1][0] - 0.7
        # вектор значений функций в начальном приближении
        f0 = numpy.array([[f1], [f2]])
        # вычисление матрицы Якоби
        M = numpy.array([[0.2 * x0[0][0] + 1, 0.4 * x0[1][0]], [0.4 * x0[0][0] - 0.1 * x0[1][0],
                                                                1 - 0.1 * x0[0][0]]])
        # вычисление определителя матрицы Якоби
        w2 = numpy.linalg.det(M)
        # вычисление новых значение х после первой итерации
        x = x0 - numpy.linalg.inv(M).dot(f0)

    print(x)

if __name__ == "__main__":
    init_task1()
