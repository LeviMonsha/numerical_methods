import numpy as np
from math import exp, log, sin, cos
from scipy.integrate import (cumtrapz, cumulative_trapezoid,
                             trapz, trapezoid, quad, dblquad, tplquad, fixed_quad, quadrature, simps, simpson)


def task1():
    """ вычисления интеграла методом трапеций """
    y = [1,2,3,4,5,6,7,8,9,10]
    res1 = cumtrapz(y)
    res2 = cumulative_trapezoid(y)
    print(res1)
    print(res2)

def task2():
    """ вычисления интеграла с накоплением """
    x = np.arange(1,11,1)
    y = []
    for xi in x:
        y.append(xi * exp(-xi) + log(xi) + 1)
    res = cumtrapz(y, x)
    print(res)

def task3():
    """ вычислить интеграл методом трапеций """
    y = [1,3,5,7,9]
    res1 = trapz(y)
    res2 = trapezoid(y)
    print(res1)
    print(res2)

def task4():
    res = quad(lambda x: exp(x) + x**2 + 2 * sin(x) - 5, 1, 5)
    print(res)

def task5():
    """ функция для интеграции принимает дополнительные параметры,
    тогда они могут быть предоставлены в параметре args """
    def integrand(x, a, b):
        return a * x**2 + b

    a = 2
    b = 1
    I = quad(integrand, 0, 1, args=(a, b))
    print(I)

def task6():
    """ вычисление двойного интеграла """
    res = dblquad(lambda x, y: x * y, 0, 0.5, lambda x: 0, lambda x: 1 - 2 * x)
    print(res)

def task7():
    """ вычисление тройного интеграла
    по области, ограниченной плоскостями """
    f = lambda z, y, x: x + y + z
    res = tplquad(f, 0, 1, lambda x: 0, lambda x: 1 - x, lambda x, y: 0, lambda x, y: 1 - x - y)
    print(res)

def task8():
    """ для выполнения
    гауссовской квадратуры на фиксированном интервале """
    """ выполняет гауссовскую квадратуру
    фиксированного порядка """
    f = lambda x: x**8
    res1 = fixed_quad(f, 0.0, 1.0)
    res2 = fixed_quad(f, 0.0, 1.0, n=4)
    print(res1, res2)
    print("~~~~~~~~~~~~")
    print(1/9.0) # analytical result

def task9():
    """ функция выполняет гауссовскую квадратуру, имеет
    дополнительные параметры. Функция возвращает квадратурную
    аппроксимацию (в пределах [a,b]) интеграла и разницу между двумя
    последними значения интеграла. """
    res1 = quadrature(np.cos, 0.0, np.pi / 2)
    print(res1)
    print(np.sin(np.pi / 2) - np.sin(0)) # analytical result

def task10():
    """ вычисление интеграла методом парабол с
        четное кол-во отрезков"""
    a, b = 0, 8
    x = [1,2,3,4,5,6,7,8]
    y = np.power(x, 3)
    res = quad(lambda x: x**3, a, b)[0]
    # res1 = simps(y, x)
    # res2 = simps(y, x, even="first")
    # res3 = simps(y, x, even="last")
    res1 = simpson(y=y, x=x)
    res2 = simpson(y=y, x=x, even="first")
    res3 = simpson(y=y, x=x, even="last")
    print("quad={}, simps(avg,first,last)={},{},{}".format(res, res1, res2, res3))

def task11():
    """ вычисление интеграла методом парабол с
    нечетное кол-во отрезков"""
    a, b = 0, 9
    x = [1,2,3,4,5,6,7,8,9]
    y = np.power(x, 3)
    res = quad(lambda x: x**3, a, b)[0]
    res1 = simpson(y=y, x=x)
    res2 = simpson(y=y, x=x, even="first")
    res3 = simpson(y=y, x=x, even="last")
    print("quad={}, simps(avg,first,last)={},{},{}".format(res, res1, res2, res3))

if __name__ == "__main__":
    # task1()
    # task2()
    # task3()
    # task4()
    # task5()
    # task6()
    # task7()
    # task8()
    # task9()
    task10()
    task11()