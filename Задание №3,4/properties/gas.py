# coding=utf-8
"""
Модуль, описывающий расчёт различных свойств газа
"""

import math
from typing import Tuple


def pseudo_temperature(gamma_g: float) -> float:
    """Рассчитывает псевдокритичекую температуру газа

    :param float gamma_g: Относительный вес газа
    :return: t_pc(*float*) Температура
    """
    t_pc = 95 + 171 * gamma_g
    return t_pc


def pseudo_pressure(gamma_g: float) -> float:
    """Рассчитывает псевдокритичекое давление газа

    :param float gamma_g: Относительный вес газа
    :return: p_pc(*float*) Давление
    """
    p_pc = 4.9 - 0.4 * gamma_g
    return p_pc


def pseudo_temperature_standing(gamma_g: float) -> float:
    """Рассчитывает псевдокритичекую температуру газа по Стэндингу

    :param float gamma_g: Относительный вес газа
    :return: t_cr(*float*) Температура
    """
    t_pc = 93.3 + 180 * gamma_g - 6.94 * pow(gamma_g, 2)
    return t_pc


def pseudo_pressure_standing(gamma_g: float) -> float:
    """Рассчитывает псевдокритичекое давление газа по Стэндингу

    :param float gamma_g: Относительный вес газа
    :return: p_pc(*float*) Давление
    """
    p_pc = 4.6 + 0.1 * gamma_g - 0.258 * pow(gamma_g, 2)
    return p_pc


def z_factor(t_pr: float, p_pr: float) -> float:
    """Рассчитывает фактор сжимаемости газа

    :param float t_pr: Приведённая температура
    :param float p_pr: Приведённое давление
    :return: z(*float*) Фактор сжимаемости
    """
    a = 1.39 * math.sqrt(t_pr - 0.92) - 0.36 * t_pr - 0.101
    b = (
        p_pr * (0.62 - 0.23 * t_pr)
        + pow(p_pr, 2) * (0.006 / (t_pr - 0.86) - 0.037)
        + 0.32 * pow(p_pr, 6) / math.exp(20.723 * (t_pr - 1))
    )
    c = 0.132 - 0.32 * math.log(t_pr) / math.log(10)
    d = math.exp(0.715 - 1.128 * t_pr + 0.42 * pow(t_pr, 2))

    z = a + (1 - a) * math.exp(-b) + c * pow(p_pr, d)
    return z


def gas_visc(t: float, p: float, z: float, gamma_g: float) -> float:
    """Рассчитывает вязкость газа

    :param float t: Температура (С)
    :param float p: Давление (атм)
    :param float z: Фактор сжимаемости
    :param float gamma_g: Удельный вес газа
    :return: mu(*float*) Вязкость газа (cP)
    """
    r = 1.8 * t
    mwg = 28.966 * gamma_g
    gd = p * mwg / (z * t * 8.31)
    a = (9.379 + 0.01607 * mwg) * pow(r, 1.5) / (209.2 + 19.26 * mwg + r)
    b = 3.448 + 986.4 / r + 0.01009 * mwg
    c = 2.447 - 0.2224 * b
    mu = 0.0001 * a * math.exp(b * pow(gd, c))
    return mu


def gas_fvf(t: float, p: float, z: float) -> float:
    """Рассчитывает объёмный фактор газа

    :param float t: Температура (С)
    :param float p: Давление (атм)
    :param float z: Фактор сжимаеомсти
    :return: fvf(*float*) Объёмный фактор газа
    """
    fvf = 0.00034722 * t * z / p
    return fvf


A1 = 0.3265
A2 = -1.07
A3 = -0.5339
A4 = 0.01569
A5 = -0.05165
A6 = 0.5475
A7 = -0.7361
A8 = 0.1844
A9 = 0.1056
A10 = 0.6134
A11 = 0.721


def z_factor_estimate_dranchuk(t_pr: float, p_pr: float, z: float) -> float:
    """Функция возвращает 0, если фактор сжимаемости по Дранчуку
    для введённых привёднных давления и температуры посчитан верно

    :param float t_pr: Приведённая температура
    :param float p_pr: Приведённое давление
    :param float z: Фактор сжимаемости
    :return: dz(*float*) Отклонение фактора сжимаемости от истинного
    """
    rho_r = 0.27 * p_pr / (z * t_pr)

    t_pr_2 = t_pr * t_pr
    t_pr_3 = t_pr_2 * t_pr
    t_pr_4 = t_pr_3 * t_pr
    t_pr_5 = t_pr_4 * t_pr

    rho_r_2 = rho_r * rho_r
    rho_r_5 = rho_r_2 * rho_r_2 * rho_r

    dz_1 = (
        -z + (A1 + A2 / t_pr + A3 / t_pr_3 + A4 / t_pr_4 + A5 / t_pr_5) * rho_r
    )

    coeff = A7 / t_pr + A8 / t_pr_2

    dz_2 = (A6 + coeff) * rho_r_2 - A9 * coeff * rho_r_5

    dz_3 = (
        A10 * (1 + A11 * rho_r_2) * rho_r_2 / t_pr_3 * math.exp(-A11 * rho_r_2)
        + 1
    )
    return dz_1 + dz_2 + dz_3


def z_factor_dranchuk(t_pr: float, p_pr: float) -> float:
    """Ищет корень уравнения для вычисления фактора сжимаемости методом бисекции

    :param float t_pr: Приведённая температура
    :param float p_pr: Приведённое давление
    :return: z(*float*) Фактор сжимаемости газа
    """
    z_low: float = 0.1
    z_mid: float = 1
    z_hi: float = 5

    for i in range(0, 20):
        z_mid = 0.5 * (z_hi + z_low)
        y_low = z_factor_estimate_dranchuk(t_pr, p_pr, z_low)
        y_hi = z_factor_estimate_dranchuk(t_pr, p_pr, z_mid)

        if y_low * y_hi < 0:
            z_hi = z_mid
        else:
            z_low = z_mid

        if abs(z_low - z_hi) < 0.001:
            break

    return z_mid


def gas_fvf_gamma(
    temperature_k: float, p_mpa: float, gamma_g: float, correl: int = 0
) -> Tuple[float, float]:
    """Рссчитывает объёмный фактор газа

    :param float temperature_k: Температура (К)
    :param float p_mpa: Давление (МПа)
    :param float gamma_g: Удельный весь газа
    :param int correl: Флаг для использования набора корреляций Стэндинга
    :return: fvf(*float*) Объёмный фактор газа
    """

    if correl:
        t_pc = pseudo_temperature_standing(gamma_g)
        p_pc = pseudo_pressure_standing(gamma_g)
        z = z_factor_dranchuk(temperature_k / t_pc, p_mpa / p_pc)
    else:
        t_pc = pseudo_temperature(gamma_g)
        p_pc = pseudo_pressure(gamma_g)
        z = z_factor(temperature_k / t_pc, p_mpa / p_pc)

    fvf = gas_fvf(temperature_k, p_mpa, z)
    return fvf, z


def gas_max_velocity(rho_g: float) -> float:
    """Рассчитывает скорость газа

    :param float rho_g: Плотность газа в рабочих условиях (кг/м3)
    :return: v(*float*) Скорость газа (м/с)
    """
    v = 146 / math.sqrt(rho_g)
    return v


def gas_density(temperature_k: float, p_mpa: float, gamma_g: float) -> float:
    """Рассчитывает плотность газа в рабочих условиях

    :param float temperature_k: Температура (К)
    :param float p_mpa: Давление (МПа)
    :param float gamma_g: Относительный вес газа
    :return: rho_g(*float*) Плотность газа (кг/м3)
    """
    #  Плотность воздуха в нормальных условиях
    m = 29e-3
    p_pa = p_mpa * 1000000
    r = 8.314
    rho_air = p_pa * m / temperature_k / r
    rho_g = rho_air * gamma_g
    return rho_g
