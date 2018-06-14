# МЦ с погл-м состоянием (3 состояния).
# Путём моделирования и расчёта вычислить среднее время дост-я полглащ. сост.
# Сделать теоретический расчёт. Решить систему уравнений с f с помощью матричного способа.
# 19/02/18. СЛУ
from random import *

def show_matrix(matrix, name):
    if name == 1:
        print("Матрица переходов: ")
    for row in matrix:
        for elem in row:
            print(elem, end=' ')
        print()

print("Допуск 3\nЗадайте исходные данные: ")

n = 10000
p = [[0.1, 0.4, 0.5], # [0->0, 0->1, 0->2]
     [0.2, 0.795, 0.005], # [1->0, 1->1, 1->2]
     [0.0, 0.0, 1.0]] # [2->0, 2->1, 2->2]
show_matrix(p, 1)
print("Кол-во экпериментов: " + str(n))

def experimet():
    print("Экспериметнальный расчёт")
    nwas = [0 for i in range(n)]  # Номер окна, в котором МЦ попала в поглощающее состояние

    for i in range(n):
        status = 0  # Начальное состояние МЦ в каждом эксперименте
        window = 0  # Номер окна, в котором находится МЦ
        while status != 2:
            r = random()
            if r <= p[status][0]:
                status = 0
            elif p[status][0] < r <= (p[status][0] + p[status][1]):
                status = 1
            else:
                status = 2
            window += 1
        nwas[i] = window

    middle_time = 0
    max_time = 0
    min_time = 10000

    for i in range(len(nwas)):
        min_time = min(min_time, nwas[i])
        max_time = max(max_time, nwas[i])
        middle_time += nwas[i]
    middle_time /= n

    difference = middle_time - int(middle_time)
    if (difference >= 0.5):
        middle_time = int(middle_time) + 1
    else:
        middle_time = middle_time - difference

    print("Среднее время работы системы до перехода в поглащающее состояние: " + str(middle_time))
    print("Минимальное время работы системы до перехода в состояние поглощения: " + str(min_time))
    print("Максимальное время работы системы до перехода в состояние поглощения: " + str(max_time))

def theory():
    print("Теоретический расчёт")
    dividend = (p[0][0] + p[0][1] + p[0][2] - p[0][0] * p[1][1] - p[0][2] * p[1][1] + p[0][1] * p[1][0] + p[0][1] * p[1][2])
    divider = (1 - p[0][0] - p[1][1] + p[0][0] * p[1][1] - p[0][1] * p[1][0])
    f02 = dividend / divider
    f12 = (p[1][0] + p[1][1] + p[1][2] + p[1][0] * f02) / (1 - p[1][1])
    print("f02(Среднее число переходов из состояния 0 в состояние 2 (поглощающее)) = " + str(f02))
    print("f12(Среднее число переходов из состояния 1 в состояние 2 (поглощающее)) = " + str(f12))

# Основная часть программы
experimet()
theory()