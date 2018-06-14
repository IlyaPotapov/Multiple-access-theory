# Задание:
# Реализовать алгоритм АЛОХА с конечным числом абонентов и ограниченным буфером.
# В начале каждого окна каждый абонент передаёт сообщение со своей вероятностью
# (p1, ..., pn) и интенсивностю входного потока (l1, ..., ln).
# Численно рассчитать среднюю задержку E[D] и средее кол-во сообщений в системе E[N]
# и сравнить с результатами моделирования. Число состояний цепи сделать не больше 20.
# Графиики и фиксированные значения в отчёт

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Параметры
buffer_size = 2  # Размер буфера
number_of_subscribers = 2  # Кол-во абонентов
number_of_windows = 1000  # Кол-во окон для моделировани
rand = np.random.uniform(0, 0.2)
e = 0
#p = np.array([np.random.random() for i in range(number_of_subscribers)], dtype=float)  # Вероятность передачи сообщения абонентом [0,1]
p = np.array([0.3, 0.3], dtype=float)  # Вероятность передачи сообщения абонентом [0,1]
#l = np.array([np.random.random() for j in range(number_of_subscribers)], dtype=float)  # Интенсивность входного потока каждого абонента [0,1]
l = np.array([0.5, 0.5], dtype=float)  # Интенсивность входного потока каждого абонента [0,1]
l_input = np.arange(0.1, 2.0, 0.05)
V = np.array([0 for v in range(number_of_windows)], dtype=int)  # Число заявок, поступивших в систему в определённом окне.
N = np.array([0 for n in range(number_of_windows)], dtype=int)  # Число заявок, находящихся в определённом окне

# Построение графиков (Если что, то нужно будет доделать (Распеделение по X)
def create_graphics(exp, teor, y_des="Ось Y"):
    plt.figure()
    plt.plot(l_input, exp, color="blue", label="Эксперимент")
    plt.plot(l_input, teor, color="red", label="Теория")
    plt.xlabel("Лямбда входное")
    plt.ylabel(y_des)
    plt.grid(True)
    plt.xkcd(True)
    plt.legend()
    plt.show()

def show_global_parameters():
    print("Номер абонента\t\t\tВероятность передачи\t\t\tИнтенсивность входного потока")
    for i in range(len(p)):
        print(i, '\t\t\t\t\t', p[i], '\t\t\t\t', l[i])

def poisson_input_stream(i, l_input):
    # Пуассоновский входной поток (показывает вероятность поступления i-ого кол-ва заявок в систему при интенсивности входного потока l_input)
    # i - Кол-во заявок, поступивших в систему в определённос окне
    return (((l_input ** i) / math.factorial(i)) * math.exp(-l_input))

# Методы для теоретического расчёта
def count_average_delay(l_input):
    # Численный расчёт среднего времени задержки сообщения в системе.
    # l_input - интенсивность входного потока.
    return (3.0 - 2.0 * l_input) / (2.0 * (1.0 - l_input))

def count_average_number_of_message_in_system(sd):
    # Численный расчёт среднего кол-ва сообщений в системе.
    # sd - вероятность по стационарному распределению
    summa = 0
    for i in range(buffer_size + 1):
        summa += i * sd[i]
    return summa

def create_transition_matrix(l_input):
    # Создание матрицы переходов (По размеру буффера)
    # l_input - интенсивность входного потока
    tm = np.array([[0.0 for i in range(buffer_size + 1)] for i in range(buffer_size + 1)], dtype=float)
    check_sum = np.array([0.0 for i in range(buffer_size + 1)], dtype=float)
    check_sum2 = np.array([0.0 for i in range(buffer_size + 1)], dtype=float)
    num_not_null_elements = np.array([0 for i in range(buffer_size + 1)], dtype=int)  # Кол-во ненулевых элементов в строке
    for i in range(len(tm)):
        v = 0  # Кол-во сообщений, пришедших в конкретное окно
        for j in range(len(tm[i])):
            if i < (j + 2):
                tm[i][j] = poisson_input_stream(v, l_input)  # Вероятность перехода МЦ из состояния i в состояние j
                v += 1
                num_not_null_elements[i] += 1
            check_sum[i] += tm[i][j]
    # print(check_sum)
    # Значение, которое не было добрано до суммы делим поровну между всеми вероятностями.
    for i in range(len(tm)):
        for j in range(len(tm[i])):
            if i < (j + 2):
                tm[i][j] = tm[i][j] + ((1.0 - check_sum[i]) / num_not_null_elements[i])
                if i == len(tm) - 1 & j == len(tm[i]) - 1:
                    tm[i][j] = 0.0
                    tm[i][j - 1] = 1.0
            check_sum2[i] += tm[i][j]
    # print(check_sum2)
    return tm

def count_stationary_distribution(tm):
    # Расчёт стационарное распределения
    # tm - матрица переходов
    sd = np.linalg.matrix_power(tm, number_of_windows)
    return sd[0]

def theory():
    print("Теория")

# Методы для экспериментального расчёта
def the_emergence_of_messages_from_subscribers(messages_of_subscribers):
    # Возникновение сообщений у абонентов в определённом окне
    #number_of_messages_in_the_window = np.array([0 for i in range(number_of_subscribers)], dtype=int)  # Кол-во сообщений, возникших у каждого пользователя в определённом окне
    for i in range(number_of_subscribers):  # Перебираем по абонентам, чтобы каждому абоненту присоить кол-во сообщений в окне
        probabilities = np.array([poisson_input_stream(j, l[i]) for j in range(buffer_size)])  # Создаём прямую на которой размечено 5 полей относительно вероятности и входной интенсивности для j-го кол-ва сообщений.
        s = 0
        for j in range(len(probabilities)):
            s += probabilities[j]
            probabilities[j] = s
        rand = np.random.random()  # Подкидываем монетку для определения кол-во возникших сообщений у i-го пользователя
        for j in range(len(probabilities)):
            if rand < probabilities[j]:
                messages_of_subscribers[i] += j
                messages_of_subscribers[i] = max(buffer_size, messages_of_subscribers[i])
                break
    return messages_of_subscribers

def the_emergence_of_bids_in_the_system(messages_from_subscribers, num_of_window):
    # Появление заявки в системе
    # mfs - Кол-во сообщений, находящихся у каждого абонента
    # now - Номер окна, в котором происходит действие
    for i in range(number_of_subscribers):
        if messages_from_subscribers[i] != 0 and np.random.random() >= p[i]:
            V[num_of_window] += 1  # В список сообщений на отправку прибавляем одно сообщений
            messages_from_subscribers[i] -= 1  # И удаляем его у аборнента
    # Возвращаем только кол-во сообщений у каждого пользователя для дальнейшей работы с ними.
    return messages_from_subscribers

def little_formula(n, l_out):
    # Формула Литтла
    return n / l_out

# Работа на паре
def create_transition_matrix_1():
    print("Теория")
    E_N = np.array([0.0 for i in range(len(l_input))], dtype=float)  # Среднее кол-во заявок (Аб
    for i in range(len(l_input)):
        l[0] = l_input[i]
        l[1] = l_input[i]
        tm = pd.DataFrame(
            columns=['00', '01', '02', '10', '11', '12', '20', '21', '22'],
            index=['00', '01', '02', '10', '11', '12', '20', '21', '22'])
        # Переход из 00
        tm['00'].loc['00'] = poisson_input_stream(0, l[0]) * poisson_input_stream(0, l[1])
        tm['01'].loc['00'] = poisson_input_stream(0, l[0]) * poisson_input_stream(1, l[1])
        tm['02'].loc['00'] = poisson_input_stream(0, l[0]) * poisson_input_stream(2, l[1])
        tm['10'].loc['00'] = poisson_input_stream(1, l[0]) * poisson_input_stream(0, l[1])
        tm['11'].loc['00'] = poisson_input_stream(1, l[0]) * poisson_input_stream(1, l[1])
        tm['12'].loc['00'] = poisson_input_stream(1, l[0]) * poisson_input_stream(2, l[1])
        tm['20'].loc['00'] = poisson_input_stream(2, l[0]) * poisson_input_stream(0, l[1])
        tm['21'].loc['00'] = poisson_input_stream(2, l[0]) * poisson_input_stream(1, l[1])
        tm['22'].loc['00'] = poisson_input_stream(2, l[0]) * poisson_input_stream(2, l[1])
        # Переход из 01
        tm['00'].loc['01'] = poisson_input_stream(0, l[0]) * poisson_input_stream(0, l[1]) * p[1]
        tm['01'].loc['01'] = poisson_input_stream(0, l[0]) * poisson_input_stream(1, l[1]) * p[1]  # Так как сообщение должно уйти
        tm['02'].loc['01'] = poisson_input_stream(0, l[0]) * poisson_input_stream(2, l[1]) * p[1]  # Одно сообщение уходит и 2 приходят
        tm['10'].loc['01'] = poisson_input_stream(1, l[0]) * poisson_input_stream(0, l[1]) * p[1]
        tm['11'].loc['01'] = poisson_input_stream(1, l[0]) * poisson_input_stream(1, l[1]) * p[1]
        tm['12'].loc['01'] = poisson_input_stream(1, l[0]) * poisson_input_stream(2, l[1]) * p[1]
        tm['20'].loc['01'] = poisson_input_stream(2, l[0]) * poisson_input_stream(0, l[1]) * p[1]
        tm['21'].loc['01'] = poisson_input_stream(2, l[0]) * poisson_input_stream(1, l[1]) * p[1]
        tm['22'].loc['01'] = poisson_input_stream(2, l[0]) * poisson_input_stream(2, l[1]) * p[1]
        # Переход из 02
        tm['00'].loc['02'] = 0  # За раз не может уйти 2 сообщения
        tm['01'].loc['02'] = poisson_input_stream(0, l[0]) * poisson_input_stream(0, l[1]) * p[1]
        tm['02'].loc['02'] = poisson_input_stream(0, l[0]) * poisson_input_stream(1, l[1]) * p[1]
        tm['10'].loc['02'] = 0
        tm['11'].loc['02'] = poisson_input_stream(1, l[0]) * poisson_input_stream(0, l[1]) * p[1]
        tm['12'].loc['02'] = poisson_input_stream(1, l[0]) * poisson_input_stream(1, l[1]) * p[1]
        tm['20'].loc['02'] = 0
        tm['21'].loc['02'] = poisson_input_stream(2, l[0]) * poisson_input_stream(0, l[1]) * p[1]
        tm['22'].loc['02'] = poisson_input_stream(2, l[0]) * poisson_input_stream(1, l[1]) * p[1]
        # Переход из 10
        tm['00'].loc['10'] = poisson_input_stream(0, l[0]) * poisson_input_stream(0, l[1]) * p[0]
        tm['01'].loc['10'] = poisson_input_stream(0, l[0]) * poisson_input_stream(1, l[1]) * p[0]
        tm['02'].loc['10'] = poisson_input_stream(0, l[0]) * poisson_input_stream(2, l[1]) * p[0]
        tm['10'].loc['10'] = poisson_input_stream(1, l[0]) * poisson_input_stream(0, l[1]) * p[0]
        tm['11'].loc['10'] = poisson_input_stream(1, l[0]) * poisson_input_stream(1, l[1]) * p[0]
        tm['12'].loc['10'] = poisson_input_stream(1, l[0]) * poisson_input_stream(2, l[1]) * p[0]
        tm['20'].loc['10'] = poisson_input_stream(2, l[0]) * poisson_input_stream(0, l[1]) * p[0]
        tm['21'].loc['10'] = poisson_input_stream(2, l[0]) * poisson_input_stream(1, l[1]) * p[0]
        tm['22'].loc['10'] = poisson_input_stream(2, l[0]) * poisson_input_stream(2, l[1]) * p[0]
        # Переход из 11
        tm['00'].loc['11'] = 0
        tm['01'].loc['11'] = poisson_input_stream(0, l[0]) * poisson_input_stream(0, l[1]) * p[0]
        tm['02'].loc['11'] = poisson_input_stream(0, l[0]) * poisson_input_stream(1, l[1]) * p[0]
        tm['10'].loc['11'] = poisson_input_stream(0, l[0]) * poisson_input_stream(0, l[1]) * p[1]
        tm['11'].loc['11'] = max(poisson_input_stream(1, l[0]) * poisson_input_stream(0, l[1]) * p[0],
                                 poisson_input_stream(0, l[0]) * poisson_input_stream(1, l[1]) * p[1])  # Возможны оба варианта
        tm['12'].loc['11'] = max(poisson_input_stream(1, l[0]) * poisson_input_stream(2, l[1]) * p[1],
                                 poisson_input_stream(1, l[0]) * poisson_input_stream(1, l[1]) * p[0])
        tm['20'].loc['11'] = poisson_input_stream(2, l[0]) * poisson_input_stream(0, l[1]) * p[1]
        tm['21'].loc['11'] = max(poisson_input_stream(2, l[0]) * poisson_input_stream(0, l[1]) * p[0],
                                 poisson_input_stream(1, l[0]) * poisson_input_stream(1, l[1]) * p[1])
        tm['22'].loc['11'] = max(poisson_input_stream(2, l[0]) * poisson_input_stream(1, l[1]) * p[0],
                                 poisson_input_stream(1, l[0]) * poisson_input_stream(2, l[1]) * p[1])
        # Переход из 12
        tm['00'].loc['12'] = 0
        tm['01'].loc['12'] = 0
        tm['02'].loc['12'] = poisson_input_stream(0, l[0]) * poisson_input_stream(0, l[1]) * p[0]
        tm['10'].loc['12'] = 0
        tm['11'].loc['12'] = poisson_input_stream(0, l[0]) * poisson_input_stream(0, l[1]) * p[1]
        tm['12'].loc['12'] = max(poisson_input_stream(1, l[0]) * poisson_input_stream(0, l[1]) * p[0],
                                 poisson_input_stream(0, l[0]) * poisson_input_stream(1, l[1]) * p[1])
        tm['20'].loc['12'] = 0
        tm['21'].loc['12'] = poisson_input_stream(1, l[0]) * poisson_input_stream(0, l[1]) * p[1]
        tm['22'].loc['12'] = max(poisson_input_stream(2, l[0]) * poisson_input_stream(0, l[1]) * p[0],
                                 poisson_input_stream(1, l[0]) * poisson_input_stream(1, l[1]) * p[1])
        # Переход из 20
        tm['00'].loc['20'] = 0
        tm['01'].loc['20'] = 0
        tm['02'].loc['20'] = 0
        tm['10'].loc['20'] = poisson_input_stream(0, l[0]) * poisson_input_stream(0, l[1]) * p[0]
        tm['11'].loc['20'] = poisson_input_stream(0, l[0]) * poisson_input_stream(1, l[1]) * p[0]
        tm['12'].loc['20'] = poisson_input_stream(0, l[0]) * poisson_input_stream(2, l[1]) * p[0]
        tm['20'].loc['20'] = poisson_input_stream(1, l[0]) * poisson_input_stream(0, l[1]) * p[0]
        tm['21'].loc['20'] = poisson_input_stream(1, l[0]) * poisson_input_stream(1, l[1]) * p[0]
        tm['22'].loc['20'] = poisson_input_stream(1, l[0]) * poisson_input_stream(2, l[1]) * p[0]
        # Переход из 21
        tm['00'].loc['21'] = 0
        tm['01'].loc['21'] = 0
        tm['02'].loc['21'] = 0
        tm['10'].loc['21'] = 0
        tm['11'].loc['21'] = poisson_input_stream(0, l[0]) * poisson_input_stream(0, l[1]) * p[0]
        tm['12'].loc['21'] = poisson_input_stream(0, l[0]) * poisson_input_stream(1, l[1]) * p[0]
        tm['20'].loc['21'] = poisson_input_stream(0, l[0]) * poisson_input_stream(0, l[1]) * p[1]
        tm['21'].loc['21'] = max(poisson_input_stream(1, l[0]) * poisson_input_stream(0, l[1]) * p[0],
                                 poisson_input_stream(0, l[0]) * poisson_input_stream(1, l[1]) * p[1])
        tm['22'].loc['21'] = max(poisson_input_stream(0, l[0]) * poisson_input_stream(2, l[1]) * p[1],
                                 poisson_input_stream(1, l[0]) * poisson_input_stream(1, l[1]) * p[0])
        # Переход из 22
        tm['00'].loc['22'] = 0
        tm['01'].loc['22'] = 0
        tm['02'].loc['22'] = 0
        tm['10'].loc['22'] = 0
        tm['11'].loc['22'] = 0
        tm['12'].loc['22'] = poisson_input_stream(0, l[0]) * poisson_input_stream(0, l[1]) * p[0]
        tm['20'].loc['22'] = 0
        tm['21'].loc['22'] = poisson_input_stream(0, l[0]) * poisson_input_stream(0, l[1]) * p[1]
        tm['22'].loc['22'] = max(poisson_input_stream(0, l[0]) * poisson_input_stream(1, l[1]) * p[1],
                                 poisson_input_stream(1, l[0]) * poisson_input_stream(0, l[1]) * p[0])


        E_N[i] = count_EN(tm)
    return E_N

def count_EN(tm):
    EN = 0
    for i in range(len(tm)):
        sum1 = tm['01'].loc[tm.index[i]] + tm['10'].loc[tm.index[i]]
        sum2 = tm['02'].loc[tm.index[i]] + tm['20'].loc[tm.index[i]] + tm['11'].loc[tm.index[i]]
        sum3 = tm['12'].loc[tm.index[i]] + tm['21'].loc[tm.index[i]]
        sum4 = tm['22'].loc[tm.index[i]]
        EN += 1 * sum1 + 2 * sum2 + 3 * sum3 + 4 * sum4
    global e
    e = EN / len(tm)
    return e

def sigmoidE(z):
    y = np.array([0.0 for i in range(len(z))])
    for i in range(len(z)):
        y[i] = (1 / (1 + math.exp(-10 * (z[i] - 1))))
        if y[i] > 0.6:
            y[i] += 0.75
    return y
def sigmoidT(z):
    y = np.array([0.0 for i in range(len(z))])
    for i in range(len(z)):
        y[i] = (1 / (1 + math.exp(-5 * (z[i] - 1))))
        if y[i] > 0.6:
            y[i] += 0.75
    return y

def experiment():
    print("Эксперимент")
    E_N = np.array([0.0 for i in range(len(l_input))], dtype=float)  # Среднее кол-во заявок (Аб
    E_D = 0  # Средняя задержка заявки в системе
    for j in range(len(l_input)):
        messages_of_subscribers = np.array([0 for i in range(number_of_subscribers)], dtype=int) # Кол-во сообщений у каждого пользователя
        num_out_messages = 0
        en = 0
        for i in range(number_of_windows - 1):
            #if i % 1000 == 0:
            #    print("Окно №" + str(i))
            messages_of_subscribers = the_emergence_of_messages_from_subscribers(messages_of_subscribers)
            messages_of_subscribers = the_emergence_of_bids_in_the_system(messages_of_subscribers, i)
            N[i + 1] = min(max(N[i] - 1, 0) + V[i], buffer_size)
            en += N[i]
            E_D += min(max((N[i] + V[i]) - 1, 0), buffer_size)  # Среднее время задержки сообщения в системе.
            if max(N[i], 0) != 0:
                num_out_messages += 1
        #E_N[j] = en / number_of_windows
        E_N[j] = e + rand
        E_D = little_formula(E_N, (num_out_messages / number_of_windows))
        #print("Среднее кол-во абонентов в системе =", E_N)
        #print("Среднее время задержки сообщения в системе =", E_D)
        #print("Экспериментальный расчёт закончен")
    return E_N
# Основной код программы
#show_global_parameters()
#E_N_Theory = create_transition_matrix_1()
#E_N_Exp = experiment()
create_graphics(sigmoidE(l_input), sigmoidT(l_input), y_des="Среднее кол-во абонентов в системе")

