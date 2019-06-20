import pandas
import pywt.data
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    plt.close('all')

    # Загрузка данных
    data_ecg = pandas.read_csv('Data/samples_01_14.csv')
    x_value =  data_ecg.iloc[:,0]
    y_value = data_ecg.iloc[:, 1]
    y_filter = data_ecg.iloc[:, 2]

    #size = len(x_value)
    size = 1002
    # Ось времени
    x = []
    for i in range(1,size):
        x.append(float(x_value[i].replace(':','').replace('.','').replace("'",''))/1000)

    # Зашумленный сигнал
    y = []
    for i in range(1, size):
        y.append(float(y_value[i]))

    # Отфильтрованный сигнал
    z = []
    for i in range(1, size):
        z.append(float(y_filter[i]))

    # График отфильтрованного сигнала
    plt.figure()
    plt.plot(x, z, lw=0.7, color = "blue")
    plt.grid(True)
    plt.xlabel("t, c")
    plt.ylabel("Напряжение, мВ")
    plt.legend(['Original'])

    # Получение коэффициентов для 6 уровней
    # print(pywt.wavelist(kind='discrete'))
    w = pywt.Wavelet('Sym5')
    nl = 6
    coeffs = pywt.wavedec(y, w, level=nl)

    # Построение графика восстановленого сигнала для каждого уровня
    for i in range(nl):
        # Настройка сетки и подграфиков
        fig = plt.figure()
        ax_1 = fig.add_subplot(2, 1, 1)
        ax_1.minorticks_on()
        ax_1.grid(which='major',
                color='k',
                linestyle=':',
                linewidth=1)
        ax_1.grid(which='minor',
                color='k',
                linestyle=':')
        ax_1.set_xlabel("t, c")
        ax_1.set_ylabel("Напряжение, мВ")
        ax_2 = fig.add_subplot(2, 1, 2)
        ax_2.minorticks_on()
        ax_2.grid(which='major',
                  color='k',
                  linestyle=':',
                  linewidth=1)
        ax_2.grid(which='minor',
                  color='k',
                  linestyle=':')
        ax_2.set_xlabel("t, c")
        ax_2.set_ylabel("Напряжение, мВ")

        # Построение зашумленного сигнала
        ax_1.plot(np.linspace(0, 2., num=1000), y[1:1001], lw=0.7, color = "blue")
        ax_1.legend(['Noisy'])

        # Восстановление сигнала по коэффициенитам вейвлет разложение
        coeff = pywt.waverec(coeffs[:i + 2] + [None] * (nl - i - 1), w)

        # Построение восстановленного сигнала
        ax_2.plot(np.linspace(0, 2., num=1000), coeff[1:1001], lw=0.7, color = "blue",)
        ax_2.legend([('Rec to lvl %d') % (nl - i)])

        # Вычисление норм разности восстановленного сигнала и отфильтрованного
        # np.linalg.norm((z - coeff), ord=2)
        print([('Lvl %d') % (nl - i)], abs(max(z[1:1001]-coeff[1:1001])).round(3))
    plt.show()
