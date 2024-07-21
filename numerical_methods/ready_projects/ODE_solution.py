from matplotlib import pyplot as plt
import unittest

import numpy as np
from scipy.integrate import odeint

def dydx(x, y):
    """ Исходное дифференциальное уравнение """
    return x + y + 2

def method_modified_Euler(func, x0, y0, h, x_end):
    """
    Решение дифференциального уравнения задачей Коши
    с помощью модифицированного метода Эйлера
    """

    x = x0
    y = y0
    result = [[x, y]]

    while x < x_end:
        tga_0 = h * func(x, y)
        x1 = x + h
        tga_1 = h * func(x1, y + tga_0)
        tga_av = (tga_0 + tga_1) / 2
        y = y + tga_av
        x = x + h

        segment = [x, round(y, 3)]
        result.append(segment)

    return result


def method_Hamming(func, x0, y0, h, x_end,
                   accuracy = 1e-3):
    """
    Решение дифференциального уравнения задачей Коши
    с помощью метода Хемминга
    """

    def method_Runge_Kutta(f, x, y, h):
        res = [[x, y]]
        for i in range(3):
            k1 = h * f(x, y)
            k2 = h * f(x + h / 2, y + k1 / 2)
            k3 = h * f(x + h / 2, y + k2 / 2)
            k4 = h * f(x + h, y + k3)
            y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            x = x + h
            res.append([x, round(y, 3)])

        return res

    def dif5t(x, y):
        for i in range(5):
            y = dydx(x, y)
        return y

    def forecast(func, h, x, y):
        y_forecast = y[0] + 4/3 * h * (2 * func(x, y[3]) - func(x, y[2]) + 2 * func(x, y[1])) + (28/90 * h**5 * dif5t(x, y[3]))
        return y_forecast

    def correction(func, h, x, y, y_forecast):
        y_correction = (9 * y[3] - y[1] + 3 * h * (func(x, y_forecast) + 2 * func(x, y[3]) - func(x, y[2]))) / 8 + (h**5 * dif5t(x, y[3]) / 40)
        return y_correction

    result = []
    arr = method_Runge_Kutta(func, x0, y0, h)
    result.extend(arr)
    x = arr[-1][0]
    while x < x_end:
        x = x + h
        y_arr = [segm[1] for segm in arr]

        y_forecast = forecast(func, h, x, y_arr)
        yp_forecast = func(x, y_forecast)
        y_correction = correction(func, h, x, y_arr, y_forecast)
        yp_correction = func(x, y_correction)
        while abs(yp_forecast - yp_correction) > accuracy:
            y_correction = correction(func, h, x, y_arr, y_forecast)
            yp_correction = func(x, y_correction)
            y_forecast = y_correction
            yp_forecast = func(x, y_forecast)
        y_correction = correction(func, h, x, y_arr, y_forecast)

        segment = [x, round(y_correction, 3)]
        arr = [*arr[1:], segment]

        result.append(segment)
    return result

if __name__ == "__main__":
    print(f"Исходное дифференциальное уравнение:\n"
          f"y` = x + y + 2\n")

    y0 = 1
    x0 = 1
    x_segment = [1, 3]
    h = 0.5

    print("Решение дифференциального уравнения задачей Коши с помощью:")
    print("модифицированного метода Эйлера")
    res1 = method_modified_Euler(dydx, x0, y0, h, x_segment[-1])
    print(res1)

    print("метода Хэмминга")
    res2 = method_Hamming(dydx, x0, y0, h, x_segment[-1])
    print(res2)

    x_seg = np.linspace(x_segment[0], x_segment[1], 5)
    correct_result = odeint(dydx, y0, x_seg)
    result = [[x, round(*y, 3)] for x, y in zip(x_seg, correct_result)]
    print(f"\nПоиск решения системы дифференциального уравнения, \n"
          f"используя функцию odeint из библиотеки scipy: {result}")

    # x = np.linspace(1, 3, 20)  # вектор моментов времени
    # t0 = 1  # начальное значение
    # t = odeint(dydx, t0, x)
    # t = np.array(t).flatten()  # преобразование массива
    # plt.plot(x, t, "-sr", linewidth=3)  # построение графика
    # plt.show()

# class UnitTest_ODE(unittest.TestCase):
#     def test_method_modified_Euler(self):
#         self.assertEqual(method_modified_Euler(dydx, 1, 1, 0.5, 3), [[1, 1],
#                                                                                         [1.5, 3.625],
#                                                                                         [2.0, 8.203],
#                                                                                         [2.5, 15.955],
#                                                                                         [3.0, 28.865]])
#
#     def test_method_Hemming(self):
#         self.assertEqual(method_Hamming(dydx, 1, 1, 0.5, 3), [[1, 1],
#                                                                                       [1.5, 3.742],
#                                                                                       [2.0, 8.587],
#                                                                                       [2.5, 16.897],
#                                                                                       [3.0, 31.063]])
