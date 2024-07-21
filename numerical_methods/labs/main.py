import numpy as np
import math
import time

def task1():
    time.sleep(3)
    print("Hello")

def say_hello(recipient):
    return "Hello, {}".format(recipient)

def task_say():
    say_hello("Tim")

def square(x):
    return x*x

def task2():
    x = np.random.randint(1, 10)
    y = square(x)
    print("Is %d squared is %d" % (x, y))

def task3():
    print(0b10 + 0b10)
    print(math.e)

def task4():
    print(dir(np))
    print(np.random.__doc__)

def task5():
    alfa = 150
    if alfa >= 0 and alfa < 90: a = 1
    elif alfa >= 90 and alfa < 180: a = 2
    elif alfa >= 180 and alfa < 270: a = 3
    else: a = 4
    print("угол в ", a, "-ом квадранте")

def task6():
    a = 10
    while a > 1:
        a = a / 2
    print(a)

    a = 0
    for i in [1,2,3,4,5]:
        a = a + i
    print(a)

    a = 0
    for i in range(1,10,1):
        a = a + i
    print(a)

    a = 0
    for i in range(5):
        a = a + i
    print(a)

    a = 10
    while a > 1:
        a /= 2
        a += 0.5
    print(a)

def sumdif(a, b):
    c = a +b
    d = a - b
    return c, d
def task6():
    e, f = sumdif(12, 15)
    print(e, " ", f)

def task7():
    x = input("x = ")
    print(x)

def task8():
    a, b, c = 10, 3, 7
    b = a / b
    c = a / c
    print(a, b, c)
    print("формулируем: %2d; %7.3f; %3d; %9.3e" % (a, b, b, c))

def task9():
    a, b, c = 2, 5, 9
    s = a + b + c
    print(a, b, c)
    print(a, b, c, sep="\t")
    print(a, b, c, end="\t")
    print(s)

#~~~~~~~~~~~~~~~~~~~~~~~~

def task1_numpy():
    a = np.array([1,2,3])
    b = np.array([[4,5,6], [7,8,9]])
    print(a)
    print(b)

def task2_numpy():
    a = np.empty([3,3], dtype=int)
    print(a)

    b = np.empty([3, 2], dtype=float)
    print(b)

def task3_numpy():
    A = np.array([[1,2,3],[4,5,6]])
    B = np.array([[2,2,2],[2,2,2]])
    C = A - B
    D = 5 * A
    F = A * B
    G = A / B
    H = A**B

    print(C)
    print(D)
    print(F)
    print(G)
    print(H)

from numpy import linalg as ln
def task4_numpy():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[1, 2, 3], [4, 5, 6]])
    C = np.dot(A, B)
    D = ln.matrix_power(A, 2)

    print("AxB = ")
    print(C)
    print("A^2: ")
    print(D)

def task5_numpy():
    A = np.array([[1,2,3,4], [5,6,7,8]])
    C = A.transpose()
    print("транспонированная матрица: ")
    print(C)

def task6_numpy():
    A = np.array([[1,-1,1],[2,1,1],[1,1,2]])
    a = ln.det(A)
    print("определитель = %2d" %a)
    C = ln.inv(A)
    print("обратная матрица: ")
    print(C)
    D = np.dot(A, C)
    print("проверка A*C: ")
    for i in range(3):
        for j in range(3):
            print("{:5.2f}".format(D[i][j]), end="\t")
        print()


if __name__ == '__main__':
    mat1 = np.array([[3, 4, 5],[1, 2, 3],[5, 6, 7]])
    mat2 = np.array([[1, -1, 1], [2, 1, 1], [1, 1, 2]])
    print(ln.matrix_power(mat1, 2)) # возведение в степень / квадратная матрица
    print(np.transpose(mat1)) # транспонирование матрицы / строки и столбцы меняются
    print("%2d" %ln.det(mat2)) # вычисление определителя / квадратная матрица
    print(ln.inv(mat2)) # вычисление обратной матрицы / квадратная матрица
    # Важно! В том случае, если определитель матрицы равен НУЛЮ – обратной матрицы НЕ СУЩЕСТВУЕТ.
    n = 16
    print(np.integrate())

    #task6_numpy()

