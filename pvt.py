# coding=utf-8
"""
Модуль, отвечающий за расчёт PVT свойств воды, нефти и газа по входным параметрам состояния.
"""

from math import exp
from properties.water import (
    water_fvf,
    water_viscosity,
)
from properties.oil import (
    dead_oil_viscosity_beggs_robinson,
    gor_velarde_si,
    saturated_oil_viscosity_beggs_robinson,
    bubble_point_standing,
    bubble_point_valko_mccain_si,
    fvf_saturated_oil_standing,
    compressibility_oil_vb,
    oil_viscosity_vasquez_beggs,
    gor_standing,
    density_mccain_si,
    fvf_mccain_si,
    oil_viscosity_standing,
    dead_oil_viscosity_standing,
    dead_oil_viscosity_two_points,
)
from properties.gas import gas_fvf_gamma, gas_visc


def calc_pvt(
    p,
    t,
    gamma_o,
    gamma_g,
    cor_set,
    mu_oil_20,
    mu_oil_50,
    r_sb=-1,
    p_rb=-1,
    b_ro=-1,
    salinity=50000,
    gamma_w=1,
    units=1,
):
    """Рассчитывает PVT свойства для газа, нефти и воды используя заданный набор корреляций
    1) z-фактор:
    True - Стэндинг
    False - ???
    2) Вязкость дегазированной нефти:
    True - Беггз и Робинсон
    False - Стэндинг
    3) Давление насыщения:
    True - Стэндинг
    False - Валко и МакКейн
    4) Остальные свойсвта нефти
    True:  Газовый фактор - Веларде, объёмный фактор - МакКейн
    False:  Газовый фактор - Стэндинг, объёмный фактор - Стэндиг

    :param float p: Давлние (psi, атм)
    :param float t: Температура (F, C)
    :param float gamma_o: Относительный вес нефти
    :param float gamma_g: Относительный вес газа
    :param list[bool[4]] cor_set:  Набор корреляций для расчёта
    :param float r_sb: Калибровочный газовый фактор в точке насыщения (scf/stb, м3/м3) опционально
    :param float p_rb: Калибровочное давление насыщения (psi, атм) опционально
    :param float b_ro: Калибровочный объёмный фактор нефти в точке насыщения (доли) опционально
    :param float salinity: Солёность (мг/литр)
    :param float gamma_w: Относительный вес воды
    :param int units: Единицы измерения(0 - полевые, 1 - СИ)
    :return: p_b(*float*) давление насыщения, r_s(*float*) газовый фактор, b_o(*float*) объёмный фактор нефти,
        mu_o(*float*) вязкость нефти, z(*float*) фактор сжимаемости газа, b_g(*float*) объёмный фактор газа,
        mu_g(*float*) вязкость газа, b_w(*float*) объёмный фактор воды, mu_w(*float*) вязкость воды

    """
    # Константы для конвертации величин
    c_rs = [0.17811, 1]

    c_p = [0.068, 1]

    t_f = [0.557, 1]

    t_offs = [255.22, 273]

    c_p_mpa = 0.1013

    p = c_p[units] * p
    p_rb = c_p[units] * p_rb
    r_sb = c_rs[units] * r_sb
    temperature_k = t_f[units] * t + t_offs[units]
    p_mpa = p * c_p_mpa
    p_rb *= c_p_mpa

    # Расчёт параметров воды

    b_wi = water_fvf(p_mpa, temperature_k)
    mu_wi = water_viscosity(p_mpa, temperature_k, salinity, gamma_w)

    # Расчёт параметров газа
    b_gi, zi = gas_fvf_gamma(temperature_k, p_mpa, gamma_g, cor_set[0])
    try:
        mu_gi = gas_visc(temperature_k, p_mpa, zi, gamma_g)
    except:
        mu_gi = 'exc'

    if r_sb <= 0:
        r_sb = 100000
        p_rb = -1
        b_ro = -1

    # Расчёт параметров нефти
    mu_do = dead_oil_viscosity_two_points(mu_oil_20, mu_oil_50, t, gamma_o)

    mu_o_sat = saturated_oil_viscosity_beggs_robinson(r_sb, mu_do)

    p_bi = bubble_point_standing(r_sb, gamma_g, temperature_k, gamma_o)

    if p_rb > 0:  # Калибровочное значение
        p_fact = p_bi / p_rb
        p_offs = p_bi - p_rb
    else:  # По умолчанию
        p_fact = 1
        p_offs = 0

    if b_ro > 0:  # Калибровочное значние
        b_o_sat = fvf_saturated_oil_standing(
            r_sb, gamma_g, temperature_k, gamma_o
        )
        b_fact = (b_ro - 1) / (b_o_sat - 1)
    else:  # По умолчанию
        b_fact = 1

    if p_mpa > (p_bi / p_fact):  # Недонасыщенная нефть
        p_mpa += p_offs
        r_si = r_sb
        b_o_sat = (
            b_fact
            * (
                fvf_saturated_oil_standing(
                    r_si, gamma_g, temperature_k, gamma_o
                )
                - 1
            )
            + 1
        )
        c_o = compressibility_oil_vb(
            r_sb, gamma_g, temperature_k, gamma_o, p_mpa
        )
        b_oi = b_o_sat * exp(c_o * (p_bi - p_mpa))
        mu_oi = oil_viscosity_vasquez_beggs(mu_o_sat, p_mpa, p_bi)

    else:  # Насыщенная нефть
        p_mpa *= p_fact
        r_si = gor_standing(p_mpa, gamma_g, temperature_k, gamma_o)
        b_oi = (
            b_fact
            * (
                fvf_saturated_oil_standing(
                    r_si, gamma_g, temperature_k, gamma_o
                )
                - 1
            )
            + 1
        )
        mu_oi = saturated_oil_viscosity_beggs_robinson(r_si, mu_do)

    return p_bi, r_si, b_oi, mu_oi, zi, b_gi, mu_gi, b_wi, mu_wi
