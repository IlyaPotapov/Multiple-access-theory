# Марковская цепь (2 состояния). Оценить (?) Pr{x^t=j|x^1=i} путём моделирования и вычислить! (Лекция 1)
# Оцениваем вероятность перехода из i-го состояния в j-ое, следовательно, начальным состоянием будет i-ое
from random import *
import matplotlib.pyplot as plt

def theory(t, p01, p10):
    p0 = [1 for i in range(t)]
    p1 = [0 for i in range(t)]
    p00 = 1 - p01
    p11 = 1 - p10
    for i in range(1, t):
        p0[i] = p0[i-1] * p00 + p1[i-1] * p10
        p1[i] = p0[i-1] * p01 + p1[i-1] * p11
    return p0

def experiment(n, t, pij, pji):
    matrix = [[0] * t for i in range(n)]  # matrix[num_of_n][num_of_t] (Матрица, показывающая состояния МЦ)
    for j in range(0, n):
        temp = 0  # Стартовое состояние МЦ во врем начала эксперимента
        for i in range(1, t):
            rv = random()  # Генерируем случайную величину от 0 до 1
            if temp == 0:
                if rv < pij:
                    temp = 1
                else:
                    temp = 0
            else:
                if rv < pji:
                    temp = 0
                else:
                    temp = 1
            matrix[j][i] = temp
    #print("Состояния МЦ: 1 строка - 1 эксперимент. 1 столбец - 1 момент времени")
    #show_matrix(matrix)
    # Подсчёт вероятности:
    probs = [0 for i in range(t)] # Массив вероятностей того, что мы будем находится в 0-ом состоянии
    for i in range(t):
        summa0 = 0 # Кол-во 0
        summa1 = 0 # Кол-во 1
        for j in range(n):
            if matrix[j][i] == 0:
                summa0 += 1
            else:
                summa1 += 1
        probs[i] = summa0/n
    return probs

def create_graphics(test, exp):
    plt.figure()
    plt.plot(test, color="blue", label="Тестовые значения")
    plt.plot(exp, color="red", label="Экспериментальные значения")
    plt.xlabel("t")
    plt.ylabel("P(0)")
    plt.grid(True)
    plt.xkcd(True)
    plt.legend()
    plt.show()

def show_matrix(matrix):
    for i in range(len(matrix)):
        print(matrix[i])

t = 100
n = 100000
print("Допуск 1\n"
      "Задайте исходные данные.\n"
      "Момент времени для вычисления вероятности некоторого состояния: " + str(t))   # Хорошие параметры
#t = int(input("Момент времени для вычисления вероятности некоторого состояния: "))
p01 = float(input("Вероятность перехода из состояния i в состояние j: "))
p10 = float(input("Вероятность перехода из состояния j в состояние i: "))
print("Укажите кол-во экспериментов: " + str(n))
#n = int(input("Укажите кол-во экспериментов: "))
test = theory(t, p01, p10)
exp = experiment(n, t, p01, p10)
print(test)
print(exp)
create_graphics(test, exp)