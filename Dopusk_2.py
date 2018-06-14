# Найти стационарное распределение (Подготовить теорию по работе с матрицами!)
# Теоретический расчёт добавить. Смотри конец тетради
import random
import numpy
import matplotlib.pyplot as plt

def show_matrix(matrix):
    for row in matrix:
        for elem in row:
            print(elem, end=' ')
        print()

def create_graphics(test):
    plt.figure()
    plt.plot(test, color="blue", label="Тестовые значения")
    #plt.plot(exp, color="red", label="Экспериментальные значения")
    plt.xlabel("t")
    plt.ylabel("P(0)")
    plt.grid(True)
    plt.xkcd(True)
    plt.legend()
    plt.show()

def create_transition_matrix():
    q = int(input("Создать матрицу самостоятельно?\n1 - Yes\n2 - No\nВаш выбор: "))
    if q == 1:
        n = int(input("Укажите размерность матрицы переходов (кол-во стационарных состояний): "))
        tm = [[0] * n for i in range(n)]  # Матрица переходов
        pi0 = [0 for i in range(n)]  # Вероятность перехода в нулевое состояние
        for i in range(n):
            pi0[i] = float(input("Укажите вероятность перехода из состояния " + str(i) + " в состояние 0: "))
            tm[i][0] = pi0[i]
        for i in range(n):
            for j in range(1, n):
                if j != (n - 1):
                    r = random.uniform(0, 1 - pi0[i])
                    tm[i][j] = r
                    pi0[i] += r
                else:
                    tm[i][j] = 1 - pi0[i]
                    pi0[i] += tm[i][j]
        print("Проверочный массив сумм строк в матрице переходов: " + str(pi0))
        tm = numpy.array(tm)
        print("Матрица переходов: ")
        show_matrix(tm)
        return tm
    elif q == 2:
        tm = numpy.array([[0.01, 0.99],
                          [0.01, 0.99]])
        print("Матрица переходов: ")
        show_matrix(tm)
        return tm
    else:
        print("Не верно")

def experiment(t, tm):
    # t - кол-во окон, в течени которого проходит эксперимент.
    # tm - матрица переходов
    print("Экспериментальные вероятности.")
    condition_in_time = numpy.array([0 for i in range(t+1)])
    rt = numpy.array([0 for i in range(t+1)])
    condition = 0  # Состояние системы. В начальный момент времени равно 0
    for i in range(1, t+1):
        r = random.random()
        if condition == 0:
            if r > tm[0][0]:
                condition = 1
            else:
                condition = 0
        else:
            if r > tm[1][0]:
                condition = 1
            else:
                condition = 0
        condition_in_time[i] = condition
        rt[i] = r
    print("Состояния: ", condition_in_time)
    print("Вероятности: ", rt)
    probs0 = 0
    probs1 = 0
    for i in range(t):
        if condition_in_time[i] == 0:
            probs0 += 1
        else:
            probs1 += 1
    print("P(0) = " + str(probs0 / t))
    print("P(1) = " + str(probs1 / t))

def theory(tm):
    print("Теоретические вероятности.")
    p1 = (1 - tm[0][0]) / (tm[1][0] - tm[0][0] + 1)
    p0 = 1 - p1
    print("P(0) = " + str(p0))
    print("P(1) = " + str(p1))


# Основная часть программы
matrix = create_transition_matrix()
if len(matrix) == 2:
    experiment(1000, matrix)
    theory(matrix)