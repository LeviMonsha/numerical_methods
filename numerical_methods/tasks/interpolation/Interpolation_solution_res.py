import numpy as np
from scipy.integrate import odeint

def dydx(x, y):
    """ Исходное дифференциальное уравнение """
    return x + y + 2

def ODE_solution(x0, y0, x_segm, n):
    '''
    Решение дифференциального уравнения,
        используя функцию odeint из библиотеки scipy
    '''
    x_linspace = np.linspace(x_segm[0], x_segm[1], n)
    correct_result = odeint(dydx, y0, x_linspace)
    result = np.round([[x, *y] for x, y in zip(x_linspace, correct_result)], 3)
    return result

def interpolation_parabolic_polynomial(set_p, h):
    '''
    Построение интерполяционного
        параболического полинома
    '''
    def degree_desired_polynomial(y):
        ''' определение степени искомого полинома '''
        degree = len(y) - 1
        print(f"Степень искомого полинома: {degree}")
        return degree

    def method_Lagrange(ind):
        ''' построение полинома
            по методу Лагранжа '''
        def basic_polynomial(x_val, i):
            div = 1
            res = 1
            for j in range(len(x_mas)):
                if j != i:
                    res *= (x_val - x_mas[j])
                    div *= (x_mas[i] - x_mas[j])
            return res / div

        def basic_polynomials(x_val):
            basic_polynomials_mas = [0.] * len(x_mas)
            for i in range(len(x_mas)):
                basic_polynomials_mas[i] = basic_polynomial(x_val, i)
            return basic_polynomials_mas

        def polynomial_Lagrange(x_val):
            res = 0
            for i in range(len(x_mas)):
                res += basic_polynomials(x_val)[i] * y_mas[i]
            return res

        poly_lagrange = [[]] * len(ind)
        for i in range(len(ind)):
            poly_lagrange[i] = [ind[i], polynomial_Lagrange(ind[i])]

        x_ = np.array([point[0] for point in poly_lagrange])
        y_ = np.array([point[1] for point in poly_lagrange])

        from matplotlib import pyplot as plt
        figure = plt.figure()
        ax1 = figure.add_axes([0, 0, 1, 1])
        ax1.plot(x_mas, y_mas)
        ax1.plot(x_, y_)
        plt.show()

        coef_newton = 0 #!!!!!!!!!!!!

        return poly_lagrange, coef_lagrange

    def method_Newton(ind):
        ''' построение полинома
            по методу Ньютона '''
        # poly_newton = np.polyfit(x, y, deg=2)



        return

    # def display_coefficients_polynomials():
    #     ''' вывод коэффициентов построенных полиномов '''
    #     print()

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
    index_center = [(x_mas[i - 1] + x_mas[i]) / 2
                    for i in range(len(x_mas))]
    # indices = [[ind_c - h/2, ind_c, ind_c + h/2]
    #            for ind_c in index_center]
    indices = [ind_c - h / 2 for ind_c in index_center]
    indices.extend(ind_c for ind_c in index_center)
    indices.extend(ind_c + h / 2 for ind_c in index_center)
    indices.sort()

    poly_lagrange, _ = method_Lagrange(indices)
    print(f"Полином Лагранжа {poly_lagrange}")
    # print(f"Коэффициенты построенного полинома методом Лагранжа: {coef_lagrange}")

    # coef_newton, poly_newton = method_Newton(x_mas, y_mas, indices)
    # print(f"Полином Ньютона {poly_newton}")
    # print(f"Коэффициенты построенного полинома методом Ньютона: {coef_newton}")

if __name__ == "__main__":
    print(f"Исходное дифференциальное уравнение:\n"
          f"y` = x + y + 2\n")

    y0 = 1
    x0 = 1
    h = 0.5
    x_segment = [1, 3]
    set_points = ODE_solution(x0, y0, x_segment, 10)

    interpolation_parabolic_polynomial(set_points, h)

    # from scipy.interpolate import lagrange
    # poly_lagrange = lagrange(x[indices], y[indices])

