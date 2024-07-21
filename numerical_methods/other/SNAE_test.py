import math
from scipy.optimize import fsolve
MAX_ITER = 100
accuracy = 1e-4
step_range = range(-100, 100)
# производная для заданной системы
def derivative(sys, x, h, idx):
    s_v = [x[i] + h if i == idx else x[i] for i in range(len(x))]
    d_v = [x[i] - h if i == idx else x[i] for i in range(len(x))]
    return (target_func(sys, s_v) - target_func(sys, d_v)) / (2 * h)
# система
def sys(x):
    r = [0] * 2
    r[0] = math.cos(x[0]-1) + x[1] - 0.8
    r[1] = x[0] - math.cos(x[1]) - 2
    return r
# вычисление целевой функции
def target_func(sys, x):
    return sum([i**2 for i in sys(x)])
# находим новый вектор с новым шагом
def update_values(sys, x, step):
    return [x[i] - step * derivative(sys, x, step, i) for i in range(len(x))]
# градиентный спуск
def gradient_descent(sys, x = [0, 0], step=0.4):
    iteration_count = 1
    while iteration_count < MAX_ITER:
        new_x = update_values(sys, x, step)
        for i in step_range:
            new_step = min(max(0.0001, step + step * i / 10), 10000)
            x1 = update_values(sys, x, new_step)
            if target_func(sys, new_x) > target_func(sys, x1):
                step = new_step
        new_x = update_values(sys, x, step)
        if abs(target_func(sys, x) - target_func(sys, new_x)) < accuracy:
            return new_x, iteration_count
        iteration_count += 1
        x = new_x
# метод гаусса
def gauss_inverse(matrix):
    n = len(matrix)
    inverse = [[0 for _ in range(n)] for _ in range(n)]
    # Создаем расширенную матрицу
    for i in range(n):
        for j in range(n):
            inverse[i][j] = matrix[i][j]
            if i == j:
                inverse[i].extend([1])
            else:
                inverse[i].extend([0])
    # Прямой ход метода Гаусса
    for i in range(n):
        for j in range(i + 1, n):
            factor = inverse[j][i] / inverse[i][i]
            for k in range(2 * n):
                inverse[j][k] -= factor * inverse[i][k]
    # Обратный ход метода Гаусса
    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            factor = inverse[j][i] / inverse[i][i]
            for k in range(2 * n):
                inverse[j][k] -= factor * inverse[i][k]
    # Нормализация строк обратной матрицы
    for i in range(n):
        factor = inverse[i][i]
        for j in range(2 * n):
            inverse[i][j] /= factor
    # Извлекаем обратную матрицу из расширенной
    result = []
    for i in range(n):
        result.append(inverse[i][n:])
    return result
# матрица якоби
def jacobian_matrix(sys, x):
    n = len(x)
    matrix = [[]] * n
    for i in range(n):
        row = [0] * n
        for j in range(n):
            x_accuracy = x.copy()
            x_accuracy[j] += accuracy
            sum_f = sys(x_accuracy)
            dif_f = sys(x)
            derivat = (sum_f[i] - dif_f[i]) / accuracy
            row[j] = derivat
        matrix[i] = row
    return matrix
# нахождение вектора
def calculate_root(x, inv_jac_m, f_x):
    return [x[i] - (inv_jac_m[i][0] * f_x[0] + inv_jac_m[i][1] * f_x[1]) for i in range(len(x))]
# метод ньютона
def newton(sys, x=[0, 0]):
    iteration_count = 1
    x_new = x
    while iteration_count < MAX_ITER:
        f_x = sys(x)
        jac_m = jacobian_matrix(sys, x)
        inv_jac_m = gauss_inverse(jac_m)
        x_new = calculate_root(x_new, inv_jac_m, f_x)
        delta_x = [x_new[i] - x[i] for i in range(len(x))]
        if all([abs(x[i] - x_new[i]) < accuracy for i in range(len(x))]):
            return x, iteration_count
        x = x_new
        iteration_count += 1
def main():
    nt = newton(sys, [0, 0])
    gr = gradient_descent(sys, (0, 0))
    print(gr)
    print(nt)
    correct_result = fsolve(sys, [0, 0])
    print(f"\nПоиск решения системы нелинейных уравнений, \n"
          f"используя функцию fsolve "
          "из библиотеки scipy: {correct_result}")

if __name__ == '__main__':
    main()