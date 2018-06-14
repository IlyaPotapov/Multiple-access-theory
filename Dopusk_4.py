# Использование МЦ для анализа систем массового обслуживания
# Поступвишие заявки с систему при заполненном буфере отбрасываются
import numpy as np
import math
import matplotlib.pyplot as plt

# Parameters
b = 2  # Размер буфера
l_input = np.arange(0.1, 2.0, 0.05)
T = 1  # Время обслуживания системы
num_windows = 100000  # Кол-во окон для моделирования
V = np.array([0 for i in range(num_windows)], dtype=int)  # Число заявок, поступивших в систему в определённом окне.
N = np.array([0 for j in range(num_windows)], dtype=int)  # Число заявок, находящихся в определённом окне


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

# Пуассоновский входной поток (показывает вероятность поступления i-ого кол-ва заявок в систему при интенсивности входного потока lamb)
def poisson_input_stream(i, lamb):
    # i - Кол-во заявок, поступивших в систему в определённос окне
    return (((lamb ** i) / math.factorial(i)) * math.exp(-lamb))

# Математическое ожидание от числа заявок (абонентов, т.к. у каждого абона может быть только 1 заявка)
def mat_og(p):
    # p - вероятность по стационарному распределению
    summa = 0
    for i in range(b + 1):
        summa += i * p[i]
    return summa

# Средняя задержка по теореме Литла
def little(n, lamb_out):
    #if lamb_out == 0:
    #    lamb_out = 0.00000001
    return n / lamb_out

# Создание матрицы переходов по МЦ (Вероятность переходов сильно зависит от входного потока) (Работает на ура!)
def create_transition_matrix(lamb):
    # lamb - интенсивность входного потока
    tm = np.array([[0.0 for i in range(b + 1)] for i in range(b + 1)], dtype=float)
    check_sum = np.array([0.0 for i in range(b + 1)], dtype=float)
    check_sum2 = np.array([0.0 for i in range(b + 1)], dtype=float)
    num_not_null_elements = np.array([0 for i in range(b + 1)], dtype=int)  # Кол-во ненулевых элементов в строке
    for i in range(len(tm)):
        v = 0  # Кол-во сообщений, пришедших в конкретное окно
        for j in range(len(tm[i])):
            if i < (j + 2):
                tm[i][j] = poisson_input_stream(v, lamb)  # Вероятность перехода МЦ из состояния i в состояние j
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

# Стационарное распределение для теоретического расчёта
def fsd(tm):
    sd = np.linalg.matrix_power(tm, num_windows)
    return sd[0]

# Приход заявок
def input_messege(lamb):
    windows = np.array([poisson_input_stream(i, lamb) for i in range(20)])
    w = np.array([0.0 for i in range(len(windows))])
    s = 0
    for i in range(len(windows)):
        s += windows[i]
        w[i] = s
    r = np.random.random()
    for i in range(len(w)):
        if r < w[i]:
            return i

def theor_l_out(sd):
    summa = 0
    for i in range(1, len(sd) - 1):
        summa += sd[i]
    return summa

# Подсчёт задержки
def count_delay(v, n):
    # v - кол-во поступивших сообщений
    # n - кол-во сообщений в буфере
    return min(max((n + v) - 1, 0), b)

def experiment():
    print("Эксперимент")
    E_N = np.array([0.0 for i in range(len(l_input))], dtype=float)  # Среднее кол-во заявок (Абонентов в системе)
    E_D = np.array([0.0 for i in range(len(l_input) // 2 - 1)], dtype=float)  # Среднее время задержки заявки в системе
    l_out = np.array([0.0 for i in range(len(l_input))], dtype=float)  # Интенсивность выходного потока
    for lamb in range(len(l_input)):
        e_N = 0  # Среднее кол-во заявок в системе в определённый момент времени
        num_out_message = 0
        e_D = 0
        for i in range(num_windows - 1):
            V[i] = input_messege(l_input[lamb])
            N[i + 1] = min(max(N[i] - 1, 0) + V[i], b)
            e_N += N[i]
            e_D += count_delay(V[i], N[i])
            if max(N[i], 0) != 0:
                num_out_message += 1
        E_N[lamb] = e_N / num_windows
        l_out[lamb] = num_out_message / num_windows
        if l_input[lamb] < 1.0:
            #E_D[lamb] = e_D / num_out_message + 0.5
            E_D[lamb] = little(E_N[lamb], l_out[lamb]) + 0.5
    return E_N, E_D, l_out

def theory():
    print("Теория")
    E_N = np.array([0.0 for i in range(len(l_input))], dtype=float)  # Среднее кол-во заявок (Абонентов в системе)
    E_D = np.array([0.0 for i in range(len(l_input) // 2 - 1)], dtype=float)  # Среднее время задержки заявки в систме
    l_out = np.array([0.0 for i in range(len(l_input))], dtype=float)  # Интенсивность выходного потока
    for lamb in range(len(l_input)):
        tm = create_transition_matrix(l_input[lamb])  # Матрица переходов
        sd = fsd(tm)  # Стационарное распределение системы
        E_N[lamb] = mat_og(sd)  # Математическое ожидание
        l_out[lamb] = theor_l_out(sd)
        #E_D[lamb] = little(E_N[lamb], l_out[lamb])
        if l_input[lamb] < 1.0:
            E_D[lamb] = (3 - 2 * l_input[lamb]) / (2 * (1 - l_input[lamb]))
    return E_N, E_D, l_out

def create_graphics_out(exp, y_des="Ось Y"):
    plt.figure()
    plt.plot(l_input, exp, color="blue", label="Эксперимент")
    #plt.plot(l_input, teor, color="red", label="Теория")
    plt.xlabel("Лямбда входное")
    plt.ylabel(y_des)
    plt.grid(True)
    plt.xkcd(True)
    plt.legend()
    plt.show()

def create_graphics_D(exp, teor, y_des="Ось Y"):
    plt.figure()
    l = np.array([l_input[i] for i in range(len(l_input) // 2 - 1)], dtype=float)  # Среднее время задержки заявки в системе
    plt.plot(l, exp, color="blue", label="Эксперимент")
    plt.plot(l, teor, color="red", label="Теория")
    plt.xlabel("Лямбда входное")
    plt.ylabel(y_des)
    plt.grid(True)
    plt.xkcd(True)
    plt.legend()
    plt.show()

exp_E_N, exp_E_D, exp_l_out = experiment()
teor_E_N, teor_E_D, teor_l_out = theory()
create_graphics_out(exp_l_out, y_des="Интенсивность выходного потока")
create_graphics(exp_E_N, teor_E_N, y_des="Среднее кол-во абонентов в системе")
create_graphics_D(exp_E_D, teor_E_D, y_des="Среднее время нахождения заявки в системе")
print(exp_l_out)

