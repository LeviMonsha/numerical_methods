import numpy as np
from scipy.integrate import odeint

def dydx(x, y):
    """ Исходное дифференциальное уравнение """
    return x + y + 2

def ODE_solution(x0, y0, x_segm,
                 n = 10):
    '''
    Решение дифференциального уравнения,
        используя функцию odeint из библиотеки scipy
    '''
    x_linspace = np.linspace(x_segm[0], x_segm[1], n)
    correct_result = odeint(dydx, y0, x_linspace)
    result = np.round([[x, *y] for x, y in zip(x_linspace, correct_result)], 3)
    return result

def interpolation_parabolic_polynomial(func, set_p):
    '''
    Построение интерполяционного
        параболического полинома
    '''
    def degree_desired_polynomial(y):
        ''' определение степени искомого полинома '''
        degree = len(y) - 1
        print(f"Степень искомого полинома: {degree}")
        return degree

    def method_Lagrange(x, y, indices):
        ''' построение полинома
            по методу Лагранжа '''
        from scipy.interpolate import lagrange
        poly_lagrange = lagrange(x[indices], y[indices])

        return poly_lagrange.coefficients, poly_lagrange

    def method_Newton(x, y):
        ''' построение полинома
            по методу Ньютона '''
        poly_newton = np.polyfit(x, y, deg=2)
        return poly_newton

    def display_coefficients_polynomials():
        ''' вывод коэффициентов построенных полиномов '''
        print()

    def inaccuracy():
        ''' вычисление погрешности '''
        # other_indices = [0, -1]  # Выбираем начальную и конечную точки
        # other_points = np.array([x[index] for index in other_indices])
        # errors = np.abs(poly_lagrange(other_points) - y[other_indices]).sum()
        # print(f"Погрешность: {errors}")
        return

    # main
    x_mas = np.array([point[0] for point in set_p])
    y_mas = np.array([point[1] for point in set_p])

    degree_desired_polynomial(y_mas)

    # выделение характерных точек
    index_center = len(x_mas) // 2
    indices = [index_center - 1, index_center, index_center + 1]
    coef_lagrange, poly_lagrange = method_Lagrange(x_mas, y_mas, indices)
    print(f"Полином Лагранжа {poly_lagrange}")
    print(f"Коэффициенты построенного полинома методом Лагранжа: {coef_lagrange}")

    coef_newton = method_Newton(x_mas, y_mas)
    print(f"Коэффициенты построенного полинома методом Ньютона: {coef_newton}")

if __name__ == "__main__":
    print(f"Исходное дифференциальное уравнение:\n"
          f"y` = x + y + 2\n")

    y0 = 1
    x0 = 1
    h = 0.5
    x_segment = [1, 3]
    set_points = ODE_solution(x0, y0, x_segment)

    interpolation_parabolic_polynomial(dydx, set_points)

