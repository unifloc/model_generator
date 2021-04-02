# coding=utf-8
"""
Модуль, содержащий в себе функции для рассчёта перепада давления и прочих параметров в трубе
используя корреляцию Беггза и Брилла.
"""

import math

from scipy.integrate import ode
from properties.water import (
    water_density_sc,
    salinity_from_water_surface_density,
)
from pvt import calc_pvt
from properties.gas import (
    pseudo_temperature_standing,
    pseudo_pressure_standing,
    gas_density,
    gas_max_velocity,
    z_factor_dranchuk,
)


def result_with_errors():
    var_error = 999999
    parameters = {
        "p": [var_error] * 2,
        "r_s": [var_error] * 2,
        "z": [var_error] * 2,
        "v_sl": [var_error] * 2,
        "v_sg": [var_error] * 2,
        "v_m": [var_error] * 2,
        "mu_l": [var_error] * 2,
        "mu_g": [var_error] * 2,
        "mu_n": [var_error] * 2,
        "rho_l": [var_error] * 2,
        "rho_g": [var_error] * 2,
        "rho_s": [var_error] * 2,
        "h_l": [1] * 2,
        "b_o": [var_error] * 2,
        "b_w": [var_error] * 2,
        "b_g": [var_error] * 2,
    }

    return var_error, parameters


def pipe_p_out(
    d,
    length,
    theta,
    p_sn,
    t_en,
    q_osc,
    wct,
    gamma_o,
    gamma_g,
    r_p,
    cor_set,
    mu_oil_20,
    mu_oil_50,
    t_sn=16,
    eps=0.000015,
    salinity=-1,
    r_sb=-1,
    p_rb=-1,
    b_ro=-1,
):
    """Рассчитывает давление на выходном узле трубы нефтепроводной по заданному давлению на входном узле используя
    корелляцию Беггза и Брилла

    :param float d: Диаметр трубы (м)
    :param list[float] length: Длина трубы (м)
    :param float theta: Угол наклона трубы (град.)
    :param float p_sn: Давление в начале трубы (атм)
    :param float t_en: Температура в конце трубы (С)
    :param float q_osc: Расход нефти (м3/день)
    :param float wct: Обводнённости (доли) (0..1)
    :param float gamma_o: Удельный вес нефти
    :param float gamma_g: Удельный вес газа
    :param float r_p: Газовый фактор (м3/м3)
    :param float t_sn: Температура в начале трубы (С)
    :param float eps: Шероховатость трубы (мм)
    :param float salinity: соленость (мг/литр)
    :param list[int[4]] cor_set:  Набор корреляций для расчёта
    :param float mu_oil_20: Вязкость нефти при 20 С(сп)
    :param float mu_oil_50: Вязкость нефти при 50 С(сп)
    :param float r_sb: Калибровочный газовый фактор в точке насыщения (scf/stb, м3/м3) опционально
    :param float p_rb: Калибровочное давление насыщения (psi, атм) опционально
    :param float b_ro: Калибровочный объёмный фактор нефти в точке насыщения (доли) опционально
    :return: p(*float*) Давление, parameters(*dict*) детальное распределение параметров в трубе
    """

    eps /= 1000
    p, parameters = segmented_pipe_pressure_drop(
        d,
        length,
        theta,
        eps,
        p_sn,
        t_sn,
        t_en,
        q_osc,
        wct,
        gamma_o,
        gamma_g,
        r_p,
        cor_set,
        mu_oil_20,
        mu_oil_50,
        salinity,
        True,
        r_sb,
        p_rb,
        b_ro,
    )
    return p, parameters


def pipe_p_in_runge_kutta(
    d,
    length,
    theta,
    p_sn,
    t_en,
    q_osc,
    wct,
    gamma_o,
    gamma_g,
    r_p,
    cor_set,
    mu_oil_20,
    mu_oil_50,
    t_sn,
    eps,
    calc_p_in=False,
    salinity=-1,
    r_sb=-1,
    p_rb=-1,
    b_ro=-1,
    rho_water=None,
):
    """Рассчитывает давление на входном узле нефтепроводной трубы по заданному давлению на выходном узле используя
    корелляцию Беггза и Брилла

    :param float d: Диаметр трубы (м)
    :param list[float] length: Длина трубы (м)
    :param float list theta: Угол наклона трубы (град.)
    :param float p_sn: Давление в начале трубы (атм)
    :param float t_en: Температура в конце трубы (С)
    :param float q_osc: Расход нефти (м3/день)
    :param float wct: Обводнённости (доли) (0..1)
    :param float gamma_o: Удельный вес нефти
    :param float gamma_g: Удельный вес газа
    :param float r_p: Газовый фактор (м3/м3)
    :param float t_sn: Температура в начале трубы (С)
    :param float eps: Шероховатость трубы (мм)
    :param bool calc_p_in: флаг для расчёта выходного(False) или входного(True) давления
    :param float salinity: соленость (мг/литр)
    :param list[int[4]] cor_set:  Набор корреляций для расчёта
    :param float mu_oil_20: Вязкость нефти при 20 С(сп)
    :param float mu_oil_50: Вязкость нефти при 50 С(сп)
    :param float r_sb: Калибровочный газовый фактор в точке насыщения (scf/stb, м3/м3) опционально
    :param float p_rb: Калибровочное давление насыщения  (psi, атм) опционально
    :param float b_ro: Калибровочный объёмный фактор нефти в точке насыщения (доли) опционально
    :return: p(*float*) Давление, parameters(*dict*) детальное распределение параметров в трубе
    """
    if rho_water:
        salinity = salinity_from_water_surface_density(rho_water)
    eps /= 1000

    method = "dopri5"

    solver = ode(pipe_pressure_drop_for_runge_kutta).set_integrator(method)
    solver.set_initial_value(p_sn, 0).set_f_params(
        d,
        theta,
        eps,
        t_sn,
        q_osc,
        wct,
        gamma_o,
        gamma_g,
        r_p,
        cor_set,
        mu_oil_20,
        mu_oil_50,
        salinity,
        calc_p_in,
        1,
        r_sb,
        p_rb,
        b_ro,
    )
    try:
        p_int = solver.integrate(sum(length)).tolist()[0]
        parameters = {"p": [p_sn, p_int]}
        for pressure in parameters["p"]:
            dp, param = pipe_pressure_drop_for_runge_kutta(
                0,
                pressure,
                d,
                theta,
                eps,
                t_sn,
                q_osc,
                wct,
                gamma_o,
                gamma_g,
                r_p,
                cor_set,
                mu_oil_20,
                mu_oil_50,
                salinity,
                calc_p_in,
                1,
                r_sb,
                p_rb,
                b_ro,
            )
            for key in param:
                if parameters.get(key):
                    parameters[key].append(param[key])
                else:
                    parameters[key] = [param[key]]
    except OverflowError:
        p_int, parameters = result_with_errors()
    return p_int, parameters


def pipe_p_in(
    d,
    length,
    theta,
    p_sn,
    t_en,
    q_osc,
    wct,
    gamma_o,
    gamma_g,
    r_p,
    cor_set,
    mu_oil_20,
    mu_oil_50,
    t_sn,
    eps,
    calc_p_in=False,
    salinity=-1,
    r_sb=-1,
    p_rb=-1,
    b_ro=-1,
    rho_water=None,
):
    """Рассчитывает давление на входном узле нефтепроводной трубы по заданному давлению на выходном узле используя
    корелляцию Беггза и Брилла

    :param float d: Диаметр трубы (м)
    :param list[float] length: Длина трубы (м)
    :param float list theta: Угол наклона трубы (град.)
    :param float p_sn: Давление в начале трубы (атм)
    :param float t_en: Температура в конце трубы (С)
    :param float q_osc: Расход нефти (м3/день)
    :param float wct: Обводнённости (доли) (0..1)
    :param float gamma_o: Удельный вес нефти
    :param float gamma_g: Удельный вес газа
    :param float r_p: Газовый фактор (м3/м3)
    :param float t_sn: Температура в начале трубы (С)
    :param float eps: Шероховатость трубы (мм)
    :param float salinity: соленость (мг/литр)
    :param list[int[4]] cor_set:  Набор корреляций для расчёта
    :param float mu_oil_20: Вязкость нефти при 20 С(сп)
    :param float mu_oil_50: Вязкость нефти при 50 С(сп)
    :param float r_sb: Калибровочный газовый фактор в точке насыщения (scf/stb, м3/м3) опционально
    :param float p_rb: Калибровочное давление насыщения  (psi, атм) опционально
    :param float b_ro: Калибровочный объёмный фактор нефти в точке насыщения (доли) опционально
    :return: p(*float*) Давление, parameters(*dict*) детальное распределение параметров в трубе
    """
    if rho_water:
        salinity = salinity_from_water_surface_density(rho_water)
    eps /= 1000
    p = segmented_pipe_pressure_drop( #, parameters
        d,
        length,
        theta,
        eps,
        p_sn,
        t_sn,
        t_en,
        q_osc,
        wct,
        gamma_o,
        gamma_g,
        r_p,
        cor_set,
        mu_oil_20,
        mu_oil_50,
        salinity,
        calc_p_in,
        r_sb,
        p_rb,
        b_ro,
    )
    parameters = 0
    if p <= 1:
        p = 0.9
    return p, parameters


def pipe_p_in_water(
    d,
    length,
    theta,
    p_sn,
    t_en,
    q_wsc,
    gamma_o,
    gamma_g,
    r_p,
    cor_set,
    mu_oil_20,
    mu_oil_50,
    t_sn=20,
    eps=0.000015,
    reverse=False,
    salinity=-1,
    r_sb=-1,
    p_rb=-1,
    b_ro=-1,
    rho_water=None,
):
    """Рассчитывает давление на входном узле водопроводной трубы по заданному давлению на выходном узле используя
    корелляцию Беггза и Брилла или корелляцию Дарси и Вейсбаха

    :param float d: Диаметр трубы (м)
    :param list[float] length: Длина трубы (м)
    :param float theta: Угол наклона трубы (град.)
    :param float p_sn: Давление в начале трубы (атм)
    :param float t_en: Температура в конце трубы (С)
    :param float q_wsc: Расход воды (м3/день)
    :param float gamma_o: Удельный вес нефти
    :param float gamma_g: Удельный вес газа
    :param float r_p: Газовый фактор (м3/м3)
    :param float t_sn: Температура в начале трубы (С)
    :param float eps: Шероховатость трубы (мм)
    :param float salinity: соленость (мг/литр)
    :param list[int[4]] cor_set:  Набор корреляций для расчёта
    :param float mu_oil_20: Вязкость нефти при 20 С(сп)
    :param float mu_oil_50: Вязкость нефти при 50 С(сп)
    :param float r_sb: Калибровочный газовый фактор в точке насыщения (scf/stb, м3/м3) опционально
    :param float p_rb: Калибровочное давление насыщения (psi, атм) опционально
    :param float b_ro: Калибровочный объёмный фактор нефти в точке насыщения (доли) опционально
    :return: p(*float*) Давление, parameters(*dict*) детальное распределение параметров в трубе
    """
    if rho_water:
        salinity = salinity_from_water_surface_density(rho_water)
    wct = 0.9999
    q_osc = q_wsc * (1 - wct)

    eps /= 1000
    p, parameters = segmented_pipe_pressure_drop(
        d,
        length,
        theta,
        eps,
        p_sn,
        t_sn,
        t_en,
        q_osc,
        wct,
        gamma_o,
        gamma_g,
        r_p,
        cor_set,
        mu_oil_20,
        mu_oil_50,
        salinity,
        reverse,
        r_sb,
        p_rb,
        b_ro,
    )

    return p, parameters


def pipe_p_in_gas(d, length, p_sn, t_m, q_g, gamma_g):
    """Рассчитывает давление на входном узле газоводной трубы по заданному давлению на выходном узле используя
    корелляцию Фланигана

    :param float d: диаметр трубы (м)
    :param list[float] length: длина трубы (м)
    :param float p_sn: давление на выходном узле (атм)
    :param float t_m: средняя температура (С)
    :param float q_g: Расход газа (млн. м3/день)
    :param float gamma_g: Удельный вес газа
    :return: p(*float*) Давление, parameters(*dict*)
    v_gas-скорость газа
    """
    tot_length = 0
    velocity_gas = []
    p_pc = pseudo_pressure_standing(gamma_g)
    t_pc = pseudo_temperature_standing(gamma_g)
    parameters = {"v_gas": velocity_gas}

    p_mpa = p_sn * 0.101325
    p_kpa = p_sn * 101.325
    p_pr = p_mpa / p_pc
    t_m += 273
    t_pr = t_m / t_pc
    z = z_factor_dranchuk(t_pr, p_pr)
    k = 1.346e7
    e_tp = 0.92
    q_g *= 1e6
    y = q_g / (
        k
        * pow((293.0 / 101.325), 1.0788)
        * pow((1.0 / gamma_g), 0.4606)
        * pow(d, 2.6182)
        * e_tp
    )
    x = pow(y, 1 / 0.5394)

    p = math.sqrt(pow(p_kpa, 2) + t_m * sum(length) * z * x)
    p /= 101.325
    r_gas = gas_density(t_m, p_mpa, gamma_g)
    r_pr = gas_density(273.15, 0.101325, gamma_g)
    v_gas = q_g * r_pr / (r_gas * math.pi * d ** 2 / 4)
    v_gas /= 24 * 60 * 60
    if math.isnan(v_gas) or math.isinf(v_gas):
        v_gas = None

    parameters["v_gas"].append(v_gas)
    return p, parameters


def pipe_pressure_drop_for_runge_kutta(
    length,
    p_sn,
    d,
    theta,
    eps,
    t_sn,
    q_osc,
    wct,
    gamma_o,
    gamma_g,
    r_p,
    cor_set,
    mu_oil_20,
    mu_oil_50,
    salinity=-1,
    calc_p_in=False,
    units=1,
    r_sb=-1,
    p_rb=-1,
    b_ro=-1,
    calcdirect=1,
):
    """Рассчитывает перепад давления в сегменте трубы, используя корреляцию Беггза и Брилла

    :param float length: Длина трубы (м)
    :param float p_sn: Давление на начальном узле (атм)
    :param float d: Диаметр сегмента трубы (м)
    :param float theta: Угол наклона (град)
    :param float eps: Шероховатость (м)
    :param float t_sn: Температура на начальном узле (С)
    :param float t_en: Температура на конечном узле (С)
    :param float q_osc: Расход нефти (м3/день)
    :param float wct: Обводнённости (доли) (0..1)
    :param float gamma_o: Относительный вес нефти
    :param float gamma_g: Относительный вес газа
    :param float r_p: Газосодержание нефти (м3/м3)
    :param float salinity: Солёность (мг/литр)
    :param bool calc_p_in: флаг для расчёта выходного(False) или входного(True) давления
    :param list[int[4]] cor_set:  Набор корреляций для расчёта
    :param int units: Единицы измерения (0 - полевые, 1 - СИ)
    :param float mu_oil_20: Вязкость нефти при 20 С(сп)
    :param float mu_oil_50: Вязкость нефти при 50 С(сп)
    :param float r_sb: Калибровочный газовый фактор в точке насыщения (scf/stb, м3/м3) опционально
    :param float p_rb: Калибровочное давление насыщения (psi, атм) опционально
    :param float b_ro: Калибровочный объёмный фактор нефти в точке насыщения (доли) опционально
    :param int calcdirect: Направление потока (1 - вперёд, 0 - назад)
    :return: dp(*float*) Перепад давления, r_s(*float*) газовый фактор, z(*float*) фактор сжимаемости газа,
        v_sl(*float*) распределённая скорость жидкости, v_sg(*float*) распределённая скорость газа,
        v_m(*float*) средняя скорость жидкости, mu_l(*float*) вязкость жидкости, mu_g(*float*) вязкость газа,
        mu_n(*float*) вязкость ГЖС, rho_l(*float*) плотность жидкости, rho_g(*float*) плотность газа,
        rho_s(*float*) плотность ГЖС, h_l(*float*) объёмный фактор заполнения, b_o(*float*) объёмный фактор нефти,
        b_w(*float*)  объёмный фактор воды, b_g(*float*)  объёмный фактор газа
    """
    if calc_p_in:
        sign = -1
    else:
        sign = 1

    # Константы для конвертации величин
    c_length = [0.3048, 1]

    t_f = [0.557, 1]

    t_offs = [-17.78, 0]

    c_pr = [1 / 14.6959, 1]

    c_ql = [0.159, 1]

    c_rs = [0.17811, 1]

    eps = c_length[units] * eps

    # Плотность воздуха в нормальных условиях
    rho_air = 1.2217

    # Относительный вес воды
    gamma_w = 1

    # Коэффициент трения на поверхности вода-газ (Н/м)
    sigma_w = 0.01

    # Коэффициент трения на поверхности нефть-газ (Н/м)
    sigma_o = 0.00841

    # Плотность для расчёта плотности нефти
    rho_ref = 1000

    if r_sb > 0:
        r_sb = r_sb * c_rs[units]
    else:  # ' no reservoir gas-oil solution ratio specified - set from producing
        r_sb = r_p

    if salinity < 0:
        salinity = 50000

    if (
        q_osc < 0.000001
    ):  # Задаётся минимальный расход нефти, так расчёт не может вестись при нулевом и отрицательном
        # значении
        q_osc = 0.000001

    q_gsc = r_p * q_osc
    q_wsc = wct * q_osc / (1 - wct)

    rho_osc = gamma_o * rho_ref
    rho_gsc = gamma_g * rho_air

    rho_w_sc = water_density_sc(salinity, gamma_w) * 1000

    p = p_sn
    t = t_sn
    p_b, r_s, b_o, mu_o, z, b_g, mu_g, b_w, mu_w = calc_pvt(
        p,
        t,
        gamma_o,
        gamma_g,
        cor_set,
        mu_oil_20,
        mu_oil_50,
        r_sb,
        p_rb,
        b_ro,
        salinity,
        gamma_w,
        1,
    )

    dp_dl, v_sl, v_sg, v_m, mu_l, mu_g, mu_n, rho_l, rho_g, rho_s, h_l = begs_brill_gradient(
        d,
        theta,
        eps,
        q_osc,
        q_wsc,
        q_gsc,
        b_o,
        b_w,
        b_g,
        r_s,
        mu_o,
        mu_w,
        mu_g,
        sigma_o,
        sigma_w,
        rho_osc,
        rho_w_sc,
        rho_gsc,
        1,
        0,
        1,
    )

    dp = dp_dl / c_pr[units] * sign
    parameters = {
        "r_s": r_s,
        "z": z,
        "v_sl": v_sl,
        "v_sg": v_sg,
        "v_m": v_m,
        "mu_l": mu_l,
        "mu_g": mu_g,
        "mu_n": mu_n,
        "rho_l": rho_l,
        "rho_g": rho_g,
        "rho_s": rho_s,
        "h_l": h_l,
        "b_o": b_o,
        "b_w": b_w,
        "b_g": b_g,
    }
    return dp, parameters


def segmented_pipe_pressure_drop(
    d_l,
    length_l,
    theta_l,
    eps_l,
    p_sn,
    t_sn,
    t_en,
    q_osc,
    wct,
    gamma_o,
    gamma_g,
    r_p,
    cor_set,
    mu_oil_20,
    mu_oil_50,
    salinity=-1,
    calc_p_in=False,
    r_sb=-1,
    p_rb=-1,
    b_ro=-1,
):
    """Рассчитывает перепад давление в сегментированной  трубе по заданному давлению на выходном узле используя
    корелляцию Беггза и Брилла. Длина, диаметр, угол наклона и шероховатость могут задаваться списком.

    :param float list d_l: Диаметр трубы (м)
    :param list[float] length_l: Длина трубы (м)
    :param float list theta_l: Угол наклона трубы (град.)
    :param float p_sn: Давление в начале трубы (атм)
    :param float t_en: Температура в конце трубы (С)
    :param float q_osc: Расход нефти (м3/день)
    :param float wct: Обводнённости (доли) (0..1)
    :param float gamma_o: Удельный вес нефти
    :param float gamma_g: Удельный вес газа
    :param float r_p: Газовый фактор (м3/м3)
    :param float t_sn: Температура в начале трубы (С)
    :param float eps_l: Шероховатость трубы (м)
    :param float salinity: соленость (мг/литр)
    :param bool calc_p_in: флаг для расчёта входного(False) или выходного(True) давления
    :param list[int[4]] cor_set:  Набор корреляций для расчёта
    :param float mu_oil_20: Вязкость нефти при 20 С(сп)
    :param float mu_oil_50: Вязкость нефти при 50 С(сп)
    :param float r_sb: Калибровочный газовый фактор в точке насыщения (scf/stb, м3/м3) опционально
    :param float p_rb: Калибровочное давление насыщения (psi, атм) опционально
    :param float b_ro: Калибровочный объёмный фактор нефти в точке насыщения (доли) опционально
    :return: dp(*float*) Перепад давление, parameters(*dict*) детальное распределение параметров в трубе
    """
    p = -1
    tot_length = 0
    i = 0
    # Выводимые параметры
    rs = []
    bo = []
    z = []
    bg = []
    bw = []
    vsl = []
    vsg = []
    vm = []
    mul = []
    mug = []
    mun = []
    rhol = []
    rhog = []
    rhos = []
    hl = []
    pout = []
    parameters = {
        "r_s": rs,
        "b_o": bo,
        "z": z,
        "b_g": bg,
        "b_w": bw,
        "v_sl": vsl,
        "v_sg": vsg,
        "v_m": vm,
        "mu_l": mul,
        "mu_g": mug,
        "mu_n": mun,
        "rho_l": rhol,
        "rho_g": rhog,
        "rho_s": rhos,
        "h_l": hl,
        "p": pout,
    }

    if isinstance(length_l, list):
        tot_length = sum(length_l)

    delta_t = t_en - t_sn

    if isinstance(length_l, list):

        for length in length_l:

            if isinstance(d_l, list):
                if d_l[i] == "undef" or d_l[i] <= 0:
                    d = d_l[i - 1]
                    d_l[i] = d
                else:
                    d = d_l[i]
            else:
                d = d_l

            if isinstance(theta_l, list):
                if theta_l[i] == "undef":
                    theta = theta_l[i - 1]
                    theta_l[i] = theta
                else:
                    theta = theta_l[i]
            else:
                theta = theta_l

            if isinstance(eps_l, list):
                if eps_l[i] < 0 or eps_l[i] == "undef":
                    eps = eps_l[i - 1]
                    eps_l[i] = eps
                else:
                    eps = eps_l[i]
            else:
                eps = eps_l

            t_en = t_sn + delta_t * length / tot_length
            try:
                p, r_s, z, v_sl, v_sg, v_m, mu_l, mu_g, mu_n, rho_l, rho_g, rho_s, h_l, b_o, b_w, b_g = pipe_pressure_drop(
                    d,
                    length,
                    theta,
                    eps,
                    p_sn,
                    t_sn,
                    t_en,
                    q_osc,
                    wct,
                    gamma_o,
                    gamma_g,
                    r_p,
                    cor_set,
                    mu_oil_20,
                    mu_oil_50,
                    salinity,
                    calc_p_in,
                    1,
                    r_sb,
                    p_rb,
                    b_ro,
                )
            except:
                r_s = r_sb
                mu_g = 0.005

            # parameters["r_s"].append(r_s)
            # parameters["v_sl"].append(v_sl)
            # parameters["v_sg"].append(v_sg)
            # parameters["v_m"].append(v_m)
            # parameters["mu_l"].append(mu_l)
            # parameters["mu_g"].append(mu_g)
            # parameters["mu_n"].append(mu_n)
            # parameters["rho_l"].append(rho_l)
            # parameters["rho_g"].append(rho_g)
            # parameters["rho_s"].append(rho_s)
            # parameters["h_l"].append(h_l)
            # parameters["b_o"].append(b_o)
            # parameters["z"].append(z)
            # parameters["b_g"].append(b_g)
            # parameters["b_w"].append(b_w)

            p_sn = p

            #parameters["p"].append(p_sn)
            t_sn = t_en
            i += 1

    return p #, parameters


def pipe_pressure_drop(
    d,
    length,
    theta,
    eps,
    p_sn,
    t_sn,
    t_en,
    q_osc,
    wct,
    gamma_o,
    gamma_g,
    r_p,
    cor_set,
    mu_oil_20,
    mu_oil_50,
    salinity=-1,
    calc_p_in=False,
    units=1,
    r_sb=-1,
    p_rb=-1,
    b_ro=-1,
    calcdirect=1,
):
    """Рассчитывает перепад давления в сегменте трубы, используя корреляцию Беггза и Брилла

    :param float d: Диаметр сегмента трубы (м)
    :param float length: Длина сегмента трубы (м)
    :param float theta: Угол наклона (град)
    :param float eps: Шероховатость (м)
    :param float p_sn: Давление на начальном узле (атм)
    :param float t_sn: Температура на начальном узле (С)
    :param float t_en: Температура на конечном узле (С)
    :param float q_osc: Расход нефти (м3/день)
    :param float wct: Обводнённости (доли) (0..1)
    :param float gamma_o: Относительный вес нефти
    :param float gamma_g: Относительный вес газа
    :param float r_p: Газосодержание нефти (м3/м3)
    :param float salinity: Солёность (мг/литр)
    :param bool calc_p_in: флаг для расчёта входного(False) или выходного(True) давления
    :param list[int[4]] cor_set:  Набор корреляций для расчёта
    :param int units: Единицы измерения (0 - полевые, 1 - СИ)
    :param float mu_oil_20: Вязкость нефти при 20 С(сп)
    :param float mu_oil_50: Вязкость нефти при 50 С(сп)
    :param float r_sb: Калибровочный газовый фактор в точке насыщения (scf/stb, м3/м3) опционально
    :param float p_rb: Калибровочное давление насыщения (psi, атм) опционально
    :param float b_ro: Калибровочный объёмный фактор нефти в точке насыщения (доли) опционально
    :param int calcdirect: Направление потока (1 - вперёд, 0 - назад)
    :return: dp(*float*) Перепад давления, r_s(*float*) газовый фактор, z(*float*) фактор сжимаемости газа,
        v_sl(*float*) распределённая скорость жидкости, v_sg(*float*) распределённая скорость газа,
        v_m(*float*) средняя скорость жидкости, mu_l(*float*) вязкость жидкости, mu_g(*float*) вязкость газа,
        mu_n(*float*) вязкость ГЖС, rho_l(*float*) плотность жидкости, rho_g(*float*) плотность газа,
        rho_s(*float*) плотность ГЖС, h_l(*float*) объёмный фактор заполнения, b_o(*float*) объёмный фактор нефти,
        b_w(*float*)  объёмный фактор воды, b_g(*float*)  объёмный фактор газа
    """

    # Выходные параметры
    r_s, z, v_sl, v_sg, v_m, mu_l, mu_g, mu_n, rho_l, rho_g, rho_s, h_l, b_o, b_w, b_g = (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    maxiter = 5

    # Константы для конвертации величин
    c_length = [0.3048, 1]

    t_f = [0.557, 1]

    t_offs = [-17.78, 0]

    c_pr = [1 / 14.6959, 1]

    c_ql = [0.159, 1]

    c_rs = [0.17811, 1]

    length = c_length[units] * length
    d = c_length[units] * d
    eps = c_length[units] * eps

    p_sn *= c_pr[units]

    if p_rb > 0:  # 'user specified
        p_rb *= c_pr[units]

    # Плотность воздуха в нормальных условиях
    rho_air = 1.2217

    # Относительный вес воды
    gamma_w = 1

    # Коэффициент трения на поверхности вода-газ (Н/м)
    sigma_w = 0.01

    # Коэффициент трения на поверхности нефть-газ (Н/м)
    sigma_o = 0.00841

    # Плотность для расчёта плотности нефти
    rho_ref = 1000

    q_osc *= c_ql[units]

    r_p *= c_rs[units]
    if r_sb > 0:
        r_sb = r_sb * c_rs[units]
    else:  # ' no reservoir gas-oil solution ratio specified - set from producing
        r_sb = r_p

    if salinity < 0:
        salinity = 50000

    if (
        q_osc < 0.000001
    ):  # Задаётся минимальный расход нефти, так расчёт не может вестись при нулевом и отрицательном
        # значении
        q_osc = 0.000001

    q_gsc = r_p * q_osc
    q_wsc = wct * q_osc / (1 - wct)

    rho_osc = gamma_o * rho_ref
    rho_gsc = gamma_g * rho_air

    t_sn = t_f[units] * t_sn + t_offs[units]
    t_en = t_f[units] * t_en + t_offs[units]

    rho_w_sc = water_density_sc(salinity, gamma_w) * 1000

    p = p_sn
    t = t_sn
    l = 0
    delta_l = length / 2
    delta_t = (t_en - t_sn) / 2

    index = 0
    while l < length - 0.001:
        if calc_p_in:
            sign = -1
        else:
            sign = 1

        delta_p = 0

        for counter in range(0, maxiter + 1):
            p_pvt = p + 0.5 * delta_p
            t_pvt = t + 0.5 * delta_t
            p_b, r_s, b_o, mu_o, z, b_g, mu_g, b_w, mu_w = calc_pvt(
                p_pvt,
                t_pvt,
                gamma_o,
                gamma_g,
                cor_set,
                mu_oil_20,
                mu_oil_50,
                r_sb,
                p_rb,
                b_ro,
                salinity,
                gamma_w,
                1,
            )
            if mu_g == 'exc':
                mu_g = 0.005

            dp_dl, v_sl, v_sg, v_m, mu_l, mu_g, mu_n, rho_l, rho_g, rho_s, h_l = begs_brill_gradient(
                d,
                theta,
                eps,
                q_osc,
                q_wsc,
                q_gsc,
                b_o,
                b_w,
                b_g,
                r_s,
                mu_o,
                mu_w,
                mu_g,
                sigma_o,
                sigma_w,
                rho_osc,
                rho_w_sc,
                rho_gsc,
                1,
                0,
                1,
            )

            delta_p = sign * dp_dl * delta_l * calcdirect
            counter += 1
            if abs(p_pvt - (p + 0.5 * delta_p)) < 0.5:
                break
        index += 1
        p += delta_p
        t += delta_t
        l += delta_l

    dp = p / c_pr[units]
    return (
        dp,
        r_s,
        z,
        v_sl,
        v_sg,
        v_m,
        mu_l,
        mu_g,
        mu_n,
        rho_l,
        rho_g,
        rho_s,
        h_l,
        b_o,
        b_w,
        b_g,
    )


def begs_brill_gradient(
    d,
    theta,
    eps,
    q_osc,
    q_wsc,
    q_gsc,
    b_o,
    b_w,
    b_g,
    r_s,
    mu_o,
    mu_w,
    mu_g,
    sigma_o,
    sigma_w,
    rho_osc,
    rho_wsc,
    rho_gsc,
    units=1,
    payne_et_all_holdup=0,
    payne_et_all_friction=1,
):
    """Рассчитывает градиент давления в элементе трубы использую корреляцию Беггза и Брилла

    :param float d: Диаметр элемента трубы (м)
    :param float theta: Угол наклона (град)
    :param float eps: Шероховатость (м)
    :param float q_osc: Расход нефти (м3/день)
    :param float q_wsc: Расход волы (м3/день)
    :param float q_gsc: Расход газа (м3/день)
    :param float b_o: Объёмный фактор нефти (доли)
    :param float b_w: Объёмный фактор воды (доли)
    :param float b_g: Объёмный фактор газа (доли)
    :param float r_s: Газосодержание нефти (м3/м3)
    :param float mu_o: Вязкость нефти (сР)
    :param float mu_w: Вязкость воды (сР)
    :param float mu_g: Вязкость газа (сР)
    :param float sigma_o: Коэффициент трения на певерхности нефть-газ (Н/м)
    :param float sigma_w: Коэффициент трения на певерхности вода-газ (Н/м)
    :param float rho_osc: Плотность нефти в нормальных условиях (кг/м3)
    :param float rho_wsc: Плотность воды в нормальных условиях (кг/м3)
    :param float rho_gsc: Плотность газа в нормальных условиях (кг/м3)
    :param int units: Единицы измерения (1 - СИ, 0 - Полевые)
    :param int payne_et_all_holdup: Флаг для использования корреляции Пэйна и ко. для объёмного фактора заполнения
    :param int payne_et_all_friction: Флаг для использования корреляции Пэйна и ко. для трения
    :return: dp(*float*), v_sl(*float*) распределённая скорость жидкости, v_sg(*float*) распределённая скорость газа,
        v_m(*float*) средняя скорость жидкости, mu_l(*float*) вязкость жидкости, mu_g(*float*) вязкость газа,
        mu_n(*float*) вязкость ГЖС, rho_l(*float*) плотность жидкости, rho_g(*float*) плотность газа,
        rho_s(*float*) плотность ГЖС, h_l(*float*) объёмный фактор заполнения
    """

    # Константы для конвертации величин
    g = [32.174, 9.8]

    c_q = [5.6146, 1]

    c_rs = [0.17811, 1]

    c_re = [1.488, 1]

    c_p = [0.00021583, 0.000009871668]

    c_sl = [4.61561, 1]

    a_p = math.pi * pow(d, 2) / 4

    q_o = c_q[units] * q_osc * b_o
    q_w = c_q[units] * q_wsc * b_w
    q_l = q_o + q_w
    q_g = b_g * (q_gsc - r_s * q_osc*0)

    if q_g < 0:
        q_g = 0

    f_w = q_w / q_l

    lambda_l = q_l / (q_l + q_g)

    rho_o = (rho_osc + c_rs[units] * r_s * rho_gsc) / b_o
    rho_w = rho_wsc / b_w
    rho_l = rho_o * (1 - f_w) + rho_w * f_w
    rho_g = rho_gsc / b_g

    rho_n = rho_l * lambda_l + rho_g * (1 - lambda_l)

    sigma_l = sigma_o * (1 - f_w) + sigma_w * f_w

    mu_l = mu_o * (1 - f_w) + mu_w * f_w

    mu_n = mu_l * lambda_l + mu_g * (1 - lambda_l)

    v_sl = 0.000011574 * q_l / a_p
    v_sg = 0.000011574 * q_g / a_p

    v_m = v_sl + v_sg

    n_re = c_re[units] * 1000 * rho_n * v_m * d / mu_n

    n_fr = pow(v_m, 2) / (g[units] * d)

    n_lv = c_sl[units] * v_sl * pow((rho_l / (g[units] * sigma_l)), 0.25)

    e = eps / d

    l1 = 316 * pow(lambda_l, 0.302)
    l2 = 0.000925 * pow(lambda_l, -2.468)
    l3 = 0.1 * pow(lambda_l, -1.452)
    l4 = 0.5 * pow(lambda_l, -6.738)

    if (lambda_l < 0.01 and n_fr < l1) or (lambda_l >= 0.01 and n_fr < l2):
        # Разделённый
        flow_pattern = 0
    elif lambda_l >= 0.01 and l2 <= n_fr <= l3:
        # Переходный
        flow_pattern = 3
    elif (0.01 <= lambda_l < 0.4 and l3 <= n_fr <= l1) or (
        lambda_l >= 0.4 and l3 < n_fr <= l4
    ):
        # Прерывистый
        flow_pattern = 1
    else:
        # Распределённый
        flow_pattern = 2

    if flow_pattern == 0 or flow_pattern == 1 or flow_pattern == 2:
        h_l = h_l_theta(
            flow_pattern, lambda_l, n_fr, n_lv, theta, payne_et_all_holdup
        )
    else:
        aa = (l3 - n_fr) / (l3 - l2)
        h_l = aa * h_l_theta(
            0, lambda_l, n_fr, n_lv, theta, payne_et_all_holdup
        ) + (1 - aa) * h_l_theta(
            1, lambda_l, n_fr, n_lv, theta, payne_et_all_holdup
        )

    f_n = calc_friction_factor(n_re, e, payne_et_all_friction)

    y = max(lambda_l / pow(h_l, 2), 0.001)

    if 1 < y < 1.2:
        s = math.log(2.2 * y - 1.2)
    else:
        s = math.log(y) / (
            -0.0523
            + 3.182 * math.log(y)
            - 0.8725 * pow(math.log(y), 2)
            + 0.01853 * pow(math.log(y), 4)
        )

    f = f_n * math.exp(s)

    rho_s = rho_l * h_l + rho_g * (1 - h_l)

    dpdl_g = c_p[units] * rho_s * g[units] * math.sin(math.pi / 180 * theta)

    dpdl_f = c_p[units] * f * rho_n * pow(v_m, 2) / (2 * d)

    dp = dpdl_g + dpdl_f

    return dp, v_sl, v_sg, v_m, mu_l, mu_g, mu_n, rho_l, rho_g, rho_s, h_l


def h_l_theta(flow_pattern, lambda_l, n_fr, n_lv, theta, payne_et_all):
    """Рассчитывает объёмный фактор заполнения

    :param int flow_pattern: Режим потока (0 -segregated, 1 - intermittent, 2 - distributed)
    :param float lambda_l: Объёмный фактор заполнения в случая несмещения фаз
    :param float n_fr: Число Фруде
    :param float n_lv: Число скорости жидкости
    :param float theta: Угол наклон (град)
    :param int payne_et_all: Флаг для использования корреляции Пэйна и ко. для объёмного фактора заполнения
    :return: h_l(*float*) Объёмный фактор заполнения
    """

    a = [0.98, 0.845, 1.065]

    b = [0.4846, 0.5351, 0.5824]

    c = [0.0868, 0.0173, 0.0609]

    e = [0.011, 2.96, 1]

    f = [-3.768, 0.305, 0]

    g = [3.539, -0.4473, 0]

    h = [-1.614, 0.0978, 0]

    h_l_0 = (
        a[flow_pattern]
        * pow(lambda_l, b[flow_pattern])
        / pow(n_fr, c[flow_pattern])
    )
    cc = max(
        0,
        (1 - lambda_l)
        * math.log(
            e[flow_pattern]
            * pow(lambda_l, f[flow_pattern])
            * pow(n_lv, g[flow_pattern])
            * pow(n_fr, h[flow_pattern])
        ),
    )

    theta_d = math.pi / 180 * theta
    psi = 1 + cc * (
        math.sin(1.8 * theta_d) + 0.333 * pow((math.sin(1.8 * theta_d)), 3)
    )

    if payne_et_all > 0:
        if theta > 0:  # Восходящий поток
            h_l = max(min(1, 0.924 * h_l_0 * psi), lambda_l)
        else:  # Нисходящий поток
            h_l = max(min(1, 0.685 * h_l_0 * psi), lambda_l)
    else:
        h_l = max(min(1, h_l_0 * psi), lambda_l)

    return h_l


def calc_friction_factor(n_re, e, rough_pipe):
    """Рассчитывает коээфициент трения для ГЖС в трубе

    :param float n_re: Число Рейнольдса
    :param float e: Относительная шероховатость
    :param float rough_pipe: Флаг для использовании корреляции Муди для шероховатой трубы
    :return: f_r(*float*) Коээфициент трения
    """
    if n_re > 2000:
        if rough_pipe > 0:
            f_n = pow(
                (
                    2
                    * math.log10(
                        2 / 3.7 * e
                        - 5.02 / n_re * math.log10(2 / 3.7 * e + 13 / n_re)
                    )
                ),
                -2,
            )
            maxiter = 19
            for i in range(1, maxiter):
                f_n_new = pow(
                    (
                        1.74
                        - 2 * math.log10(2 * e + 18.7 / (n_re * pow(f_n, 0.5)))
                    ),
                    -2,
                )
                f_int = f_n
                f_n = f_n_new
                if abs(f_n_new - f_int) > 0.001:
                    break
        else:
            f_n = 0.0056 + 0.5 * pow(n_re, -0.32)
    else:
        f_n = 64 / n_re

    return f_n


def gas_pipe_diameter(t, p, q_g, gamma_g):
    """Рассчитыает диаметр газовой трубы

    :param float t: Температура (С)
    :param float p: Давление (атм)
    :param float q_g: Расход газа (м3/день)
    :param float gamma_g: Относительный вес газа
    :return: d(*float*) Диаметр (м)
    """
    p_pc = pseudo_pressure_standing(gamma_g)
    t_pc = pseudo_temperature_standing(gamma_g)
    p_mpa = p * 0.101325
    p_kpa = p_mpa * 1000
    p_pr = p_mpa / p_pc
    temperature_k = t + 273
    t_pr = t / t_pc
    z = z_factor_dranchuk(t_pr, p_pr)
    rho_g = gas_density(temperature_k, p_mpa, gamma_g)
    v_max = gas_max_velocity(rho_g)
    q = q_g / 1000000
    d = math.sqrt(5.19 * q * z * temperature_k / v_max / p_kpa)
    return d
