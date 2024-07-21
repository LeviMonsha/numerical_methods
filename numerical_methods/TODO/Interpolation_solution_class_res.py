import random

import numpy as np
from scipy.integrate import odeint

def dydx(x, y):
    """ Исходное дифференциальное уравнение """
    return x + y + 2

def ODE_solution(x_segm, n):
    '''
    Решение дифференциального уравнения,
        используя функцию odeint из библиотеки scipy
    '''
    y0 = 1
    x0 = 1

    x_linspace = np.linspace(x_segm[0], x_segm[1], n)
    correct_result = odeint(dydx, y0, x_linspace)
    result = np.round([[x, *y] for x, y in zip(x_linspace, correct_result)], 3)
    return result

class __interpolation_parabolic_polynomial:
    '''
    Построение интерполяционного
        параболического полинома
    '''

    def __init__(self, x_mas, y_mas, h = 0.5,
                 accuracy = 1e-1):
        self.h = h
        self.accuracy = accuracy
        self.x_mas = x_mas
        self.y_mas = y_mas
        self.poly_method = None
        self.indices = self.__extraction_notable_points(x_mas)

    def inaccuracy(self):
        ''' вычисление погрешности '''
        n_count = 10
        other_indices = [1, 2]
        correct_values = ODE_solution(other_indices, n_count)
        intervals = np.linspace(other_indices[0], other_indices[1], n_count)
        inter_values = [self.poly_method(other_ind) for other_ind in intervals]

        errors = np.abs([inter_values[i] - correct_values[i][1] for i in range(n_count)]).sum()
        return errors

    def __extraction_notable_points(self, x_mas):
        # выделение характерных точек
        index_center = [(x_mas[i - 1] + x_mas[i]) / 2
                        for i in range(1, len(x_mas))]
        indices = [ind_c - self.h / 2 for ind_c in index_center]
        indices.extend(ind_c for ind_c in index_center)
        indices.extend(ind_c + self.h / 2 for ind_c in index_center)
        indices.sort()
        return indices

    def degree_desired_polynomial(self):
        ''' определение степени искомого полинома '''
        degree = len(y_mas) - 1
        y_mas_copy = y_mas.copy()

        while len(y_mas_copy) > self.accuracy:
            new_mas = []
            for i in range(1, len(y_mas_copy)):
                new_mas.append(y_mas_copy[i] - y_mas_copy[i - 1])
            degree -= 1
            y_mas_copy = new_mas[:-1]
        return degree

    def graph(self, poly):
        x_ = np.array([point[0] for point in poly])
        y_ = np.array([point[1] for point in poly])
        from matplotlib import pyplot as plt
        figure = plt.figure()
        ax1 = figure.add_axes([0, 0, 1, 1])
        ax1.plot(x_mas, y_mas, label="interpolation")
        ax1.plot(x_, y_, label="default")
        plt.show()

    def display_coefficients_polynomials(self, coef):
        ''' вывод коэффициентов построенных полиномов '''
        # poly = np.poly1d(coef)
        print("Коэффициенты построенного полинома\n"
              "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(coef)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

class method_Lagrange(__interpolation_parabolic_polynomial):
    ''' построение полинома
        по методу Лагранжа '''

    def __init__(self):
        super().__init__(x_mas, y_mas, h)
        self.degree = self.degree_desired_polynomial()
        self.poly_method = self.__polynomial_Lagrange
        self.main_coef_poly = None

    def __basic_polynomial(self, x_val, k):
        div = 1
        res = 1
        for j in range(self.degree):
            if j != k:
                res *= (x_val - x_mas[j])
                div *= (x_mas[k] - x_mas[j])
        return res / div

    # def __basic_polynomials(self, x_val):
    #     basic_polynomials_mas = [0.] * self.degree
    #     for i in range(self.degree):
    #         basic_polynomials_mas[i] = self.__basic_polynomial(x_val, i)
    #     return basic_polynomials_mas

    def __polynomial_Lagrange(self, x_val):
        res = 0
        self.bps = [0.] * self.degree
        for i in range(self.degree):
            # res += self.bps[i] * y_mas[i]
            self.bps[i] = self.__basic_polynomial(x_val, i)
            res += self.bps[i] * y_mas[i]
        if self.main_coef_poly is None:
            self.main_coef_poly = np.round(self.bps, 5)
        return res

    def solve(self):
        poly_lagrange = [[]] * len(self.indices)
        for i in range(len(self.indices)):
            poly_lagrange[i] = [self.indices[i], np.round(self.__polynomial_Lagrange(self.indices[i]), 3)]
        return poly_lagrange

class method_Newton(__interpolation_parabolic_polynomial):
    ''' построение полинома
        по методу Ньютона '''

    def __init__(self):
        super().__init__(x_mas, y_mas, h)
        self.degree = self.degree_desired_polynomial()
        self.poly_method = self.__polynomial_Newton
        self.main_coef_poly = None

    def __divided_difference(self, k):
        result = 0
        for i in range(k):
            mul = 1
            for j in range(k):
                if j != i:
                    mul *= (x_mas[i] - x_mas[j])
            result += y_mas[i] / mul
        return result

    # def __divided_differences(self):
    #     split_diff_mas = [0.] * (self.degree)
    #     for i in range(0, self.degree):
    #         split_diff_mas[i] = self.__divided_difference(i)
    #     return split_diff_mas

    def __polynomial_Newton(self, x_val):
        res = y_mas[0]
        # self.dds = self.__divided_differences()
        self.dds = [0.] * self.degree
        for k in range(1, self.degree + 1):
            self.dds[k - 1] = self.__divided_difference(k)
            mul = 1
            for j in range(k - 1):
                mul *= (x_val - x_mas[j])
            res += self.dds[k - 1] * mul
        if self.main_coef_poly is None:
            self.main_coef_poly = np.round(self.dds, 5)
        return res

    def solve(self):
        poly_newton = [[]] * len(self.indices)
        for i in range(len(self.indices)):
            poly_newton[i] = [self.indices[i], np.round(self.__polynomial_Newton(self.indices[i]), 3)]
        return poly_newton

if __name__ == "__main__":
    print(f"Исходное дифференциальное уравнение:\n"
          f"y` = x + y + 2")

    h = 0.5
    x_segment = [1, 3]

    set_points = ODE_solution(x_segment, 10)
    x_mas = np.array([point[0] for point in set_points])
    y_mas = np.array([point[1] for point in set_points])

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    ipp_lagrange = method_Lagrange()
    print(f"Степень искомого полинома: {ipp_lagrange.degree}")
    poly_lagrange = ipp_lagrange.solve()
    print(f"Полином Лагранжа (показаны первые 5) {poly_lagrange[:5]}")
    ipp_lagrange.display_coefficients_polynomials(ipp_lagrange.main_coef_poly)
    inaccuracy_poly_lagrange = ipp_lagrange.inaccuracy()
    print(f"Погрешность: {inaccuracy_poly_lagrange}")
    # ipp_lagrange.graph(poly_lagrange)

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    ipp_newton = method_Newton()
    print(f"Степень искомого полинома: {ipp_newton.degree}")
    poly_newton = ipp_newton.solve()
    print(f"Полином Ньютона (показаны первые 5) {poly_newton[:5]}")
    ipp_newton.display_coefficients_polynomials(ipp_newton.main_coef_poly)
    inaccuracy_poly_newton = ipp_newton.inaccuracy()
    print(f"Погрешность: {inaccuracy_poly_newton}")
    # ipp_newton.graph(poly_newton)


    correct_result_poly = np.polyfit(x_mas, y_mas, deg=ipp_newton.degree-1)
    print(f"\nПостроение интерполяционного\n"
          f"параболического полинома Лагранжа и Ньютона,\n"
          f"используя функцию polyfit\n"
          f"из библиотеки numpy: {correct_result_poly}\n")