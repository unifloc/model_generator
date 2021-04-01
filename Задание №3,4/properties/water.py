# coding=utf-8
"""
Модуль, описывающий расчёт различных свойств воды
"""
from math import sqrt


def water_viscosity(
    pressure_mpa: float,
    temperature_k: float,
    salinity_mg_liter: float,
    gamma_w: float,
) -> float:
    """Расчитывает вязкость воды

    :param float pressure_mpa: Давление (МПа)
    :param float temperature_k: Температура (К)
    :param float salinity_mg_liter: Солёность (мг/литр)
    :param float gamma_w: Относительный вес воды
    :return: viscosity(*float*) Вязкость воды (cP)
    """
    wptds = salinity_mg_liter / (10000 * gamma_w)
    a = (
        109.574
        - 8.40564 * wptds
        + 0.313314 * pow(wptds, 2)
        + 0.00872213 * pow(wptds, 3)
    )
    b = (
        -1.12166
        + 0.0263951 * wptds
        - 0.000679461 * pow(wptds, 2)
        - 5.47119e-5 * pow(wptds, 3)
        + 1.55586e-6 * pow(wptds, 4)
    )
    visc = a * pow(1.8 * temperature_k - 460, b)
    psi = pressure_mpa * 145.04
    mu = visc * (0.9994 + 4.0295e-5 * psi + 3.1062e-9 * pow(psi, 2))
    return mu


def water_density(
    pressure_mpa: float,
    temperature_k: float,
    salinity_mg_liter: float,
    gamma_w: float,
) -> float:
    """Рассчитывает плотность воды

    :param float pressure_mpa: Давление (МПа)
    :param float temperature_k: Температура (К)
    :param float salinity_mg_liter: Солёность (мг/литр)
    :param float gamma_w: Относительный вес воды
    :return: density(*float*) Плотность воды (кг/м3)
    """
    wptds = salinity_mg_liter / (10000 * gamma_w)
    wd = 0.0160185 * (62.368 + 0.438603 * wptds + 0.00160074 * pow(wptds, 2))
    rho = wd / water_fvf(pressure_mpa, temperature_k)
    return rho


def water_fvf(pressure_mpa: float, temperature_k: float) -> float:
    """Расчитывает объёмный фактор воды

    :param float pressure_mpa: Давление (МПа)
    :param float temperature_k: Температура (К)
    :return: fvf(*float*) объёмный фактор воды
    """
    f = 1.8 * temperature_k - 460
    psi = pressure_mpa * 145.04
    dvwp = (
        -1.95301e-9 * psi * f
        - 1.72834e-13 * pow(psi, 2) * f
        - 3.58922e-7 * psi
        - 2.25341e-10 * pow(psi, 2)
    )
    dvwt = -1.0001e-2 + 1.33391e-4 * f + 5.50654e-7 * pow(f, 2)
    fvf = (1 + dvwp) * (1 + dvwt)
    return fvf


def water_compressibilty(pressure_mpa: float, temperature_k: float, salinity_mg_liter: float) -> float:
    """Рассчитывает сжимаемость воды

    :param float pressure_mpa: Давление (МПа)
    :param float temperature_k: Температура (К)
    :param float salinity_mg_liter: Солёность (мг/литр)
    :return: compressibilty(*float*) Сжимаемость воды (1/МПа)
    """
    f = 1.8 * temperature_k - 460
    psi = pressure_mpa * 145.04
    compressibilty = (
        0.1
        * 145.04
        / (7.033 * psi + 0.5415 * salinity_mg_liter - 537 * f + 403300)
    )
    return compressibilty


def water_density_sc(salinity_mg_liter: float, gamma_w: float) -> float:
    """Рассчитывает плотность воды при стандартных условиях

    :param float salinity_mg_liter: Солёность (мг/литр)
    :param float gamma_w: Относительный вес воды
    :return: density(*float*) Плотность воды (кг/м3)
    """
    wptds = salinity_mg_liter / (10000 * gamma_w)
    density = 0.0160185 * (
        62.368 + 0.438603 * wptds + 0.00160074 * pow(wptds, 2)
    )
    return density


def salinity_from_water_surface_density(density: float) -> float:
    """

    :param density кг/м3:
    :return: Солёность (мг/литр)
    """

    gamma_w = 1
    return -1370000.74965329 * gamma_w + 0.0523533577593272 * sqrt(
        gamma_w ** 2 * (1.42288e15 * density / 1000 - 736734305836203.0)
    )
