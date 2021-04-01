# coding=utf-8
"""
Модуль, описывающий расчёт различных свойств нефти
"""


import math


def gor_standing(
    pressure_mpa: float,
    gamma_gas: float,
    temperature_k: float,
    gamma_oil: float,
) -> float:
    """Рассчитывает газосодержание нефти по Стэндингу

    :param float pressure_mpa: Давление (МПа)
    :param float gamma_gas: Относительный вес газа
    :param float temperature_k: Темпераутра (К)
    :param float gamma_oil: Относительный весь нефти
    :return: gor(*float*) Газосодержание (м3/м3)
    """
    yg = 1.225 + 0.001648 * temperature_k - 1.769 / gamma_oil
    gor = gamma_gas * pow(1.92 * pressure_mpa / pow(10, yg), 1.204)
    return gor


def gor_velarde_si(
    pressure_mpa: float,
    bubble_point_pressure_mpa: float,
    gamma_gas: float,
    temperature_k: float,
    gamma_oil: float,
    rsb_m3_m3: float,
) -> float:
    """Рассчитывает газосодержание нефти

    :param float pressure_mpa: Давление (МПа)
    :param float bubble_point_pressure_mpa: Давление насыщения (МПа)
    :param float gamma_gas: Относительный вес газа
    :param float temperature_k: ТЕмпература (К)
    :param float gamma_oil: Относительный вес нефти
    :param float rsb_m3_m3: Газосодержание в точке насыщения (м3/м3)
    :return: gor(*float*) Газосодержание (м3/м3)
    """
    api = 141.5 / gamma_oil - 131.5
    maxrs = 800

    if bubble_point_pressure_mpa > bubble_point_valko_mccain_si(
        maxrs, gamma_gas, temperature_k, gamma_oil
    ):
        if pressure_mpa < bubble_point_pressure_mpa:
            gor = rsb_m3_m3 * pressure_mpa / bubble_point_pressure_mpa
        else:
            gor = rsb_m3_m3
        return gor

    if bubble_point_pressure_mpa > 0:
        pr = (pressure_mpa - 0.101) / bubble_point_pressure_mpa
    else:
        pr = 0

    if pr <= 0:
        gor = 0
        return gor
    elif pr >= 1:
        gor = rsb_m3_m3
        return gor
    elif pr < 1:
        a_0 = 1.8653 * pow(10, -4)
        a_1 = 1.672608
        a_2 = 0.92987
        a_3 = 0.247235
        a_4 = 1.056052

        a1 = (
            a_0
            * pow(gamma_gas, a_1)
            * pow(api, a_2)
            * pow((1.8 * temperature_k - 460), a_3)
            * pow(bubble_point_pressure_mpa, a_4)
        )

        b_0 = 0.1004
        b_1 = -1.00475
        b_2 = 0.337711
        b_3 = 0.132795
        b_4 = 0.302065

        a2 = (
            b_0
            * pow(gamma_gas, b_1)
            * pow(api, b_2)
            * pow((1.8 * temperature_k - 460), b_3)
            * pow(bubble_point_pressure_mpa, b_4)
        )

        c_0 = 0.9167
        c_1 = -1.48548
        c_2 = -0.164741
        c_3 = -0.09133
        c_4 = 0.047094

        a3 = (
            c_0
            * pow(gamma_gas, c_1)
            * pow(api, c_2)
            * pow((1.8 * temperature_k - 460), c_3)
            * pow(bubble_point_pressure_mpa, c_4)
        )

        rsr = a1 * pow(pr, a2) + (1 - a1) * pow(pr, a3)

        gor = rsr * rsb_m3_m3
        return gor

    return 0


def bubble_point_standing(
    rsb_m3m3: float, gamma_gas: float, temperature_k: float, gamma_oil: float
) -> float:
    """Рассчитывает давление насыщения по Стэндигу

    :param float rsb_m3m3: Газосодержание в точке насыщения (м3/м3)
    :param float gamma_gas: Относительный вес газа
    :param float temperature_k: Температура (К)
    :param float gamma_oil: Относительный вес нефти
    :return: p_b(*float*) Давление насыщения (Мпа)
    """
    min_rsb = 1.8
    rsb_old = rsb_m3m3
    if rsb_m3m3 < min_rsb:
        rsb_m3m3 = min_rsb
    yg = 1.225 + 0.001648 * temperature_k - 1.769 / gamma_oil
    p_b = 0.5197 * pow(rsb_m3m3 / gamma_gas, 0.83) * pow(10, yg)

    if rsb_old < min_rsb:
        p_b = (p_b - 0.1013) * rsb_old / min_rsb + 0.1013
    return p_b


def bubble_point_valko_mccain_si(
    rsb_m3m3: float, gamma_gas: float, temperature_k: float, gamma_oil: float
) -> float:
    """Рассчитывает давление насыщения по МакКейну

    :param float rsb_m3m3: Газосодержание в точке насыщения (м3/м3)
    :param float gamma_gas: Относительный вес газа
    :param float temperature_k: Температура (К)
    :param float gamma_oil: Относительный вес нефти
    :return: pb(*float*) Давление (МПа)
    """
    min_rsb = 1.8
    max_rsb = 800
    rsb_old = rsb_m3m3

    if rsb_m3m3 < min_rsb:
        rsb_m3m3 = min_rsb

    if rsb_m3m3 > max_rsb:
        rsb_m3m3 = max_rsb

    api = 141.5 / gamma_oil - 131.5

    z1 = (
        -4.814074834
        + 0.7480913 * math.log(rsb_m3m3)
        + 0.1743556 * pow(math.log(rsb_m3m3), 2)
        - 0.0206 * pow(math.log(rsb_m3m3), 3)
    )
    z2 = (
        1.27
        - 0.0449 * api
        + 4.36 * pow(10, -4) * pow(api, 2)
        - 4.76 * pow(10, -6) * pow(api, 3)
    )
    z3 = (
        4.51
        - 10.84 * gamma_gas
        + 8.39 * pow(gamma_gas, 2)
        - 2.34 * pow(gamma_gas, 3)
    )
    z4 = (
        -7.2254661
        + 0.043155 * temperature_k
        - 8.5548 * pow(10, -5) * pow(temperature_k, 2)
    )
    z4 += 6.00696 * pow(10, -8) * pow(temperature_k, 3)
    z = z1 + z2 + z3 + z4

    lnpb = 2.498006 + 0.713 * z + 0.0075 * pow(z, 2)
    pb = pow(2.718282, lnpb)

    if rsb_old < min_rsb:
        pb = (pb - 0.1013) * rsb_old / min_rsb + 0.1013

    if rsb_old > max_rsb:
        pb = (pb - 0.1013) * rsb_old / max_rsb + 0.1013

    return pb


def fvf_mccain_si(
    rs_m3m3: float,
    gamma_gas: float,
    sto_density_kg_m3: float,
    reservoir_oil_density_kg_m3: float,
) -> float:
    """Рассчитывает объёмный фактор нефти по МакКейну

    :param float rs_m3m3: Газосодержание (м3/м3)
    :param float gamma_gas: Относительный вес газа
    :param float sto_density_kg_m3: Плотность нефти (кг/м3)
    :param float reservoir_oil_density_kg_m3: Плотность пластовой нефти (кг/м3)
    :return: fvf(*float*) Объёмный фактор
    """
    fvf = (
        sto_density_kg_m3 + 1.22117 * rs_m3m3 * gamma_gas
    ) / reservoir_oil_density_kg_m3
    return fvf


def fvf_saturated_oil_standing(
    rs_m3m3: float, gamma_gas: float, temperature_k: float, gamma_oil: float
) -> float:
    """Рассчитывает объёмный фактор насыщенной нефти по Стэндингу

    :param float rs_m3m3: Газосодержание (м3/м3)
    :param float gamma_gas: Относительный вес газа
    :param float temperature_k: Температура (К)
    :param float gamma_oil: Относительный вес нефти
    :return: fvf(*float*) Объёмный фактор
    """
    f = (
        5.615 * rs_m3m3 * pow(gamma_gas / gamma_oil, 0.5)
        + 2.25 * temperature_k
        - 575
    )
    fvf = 0.972 + 0.000147 * pow(f, 1.175)
    return fvf


def fvf_above_bubble_point_standing(
    pressure_mpa: float,
    bubble_point_pressure_mpa: float,
    oil_compressibility: float,
    fvf_saturated_oil: float,
) -> float:
    """Рассчитывает объёмный фактор нефти выше точки насыщения по Стэндингу

    :param float pressure_mpa: Давление (МПа)
    :param float bubble_point_pressure_mpa: Давление насыщения (МПа)
    :param float oil_compressibility: Сжимаемость нефти (1/МПа)
    :param float fvf_saturated_oil: Объёмный фактор насыщенной нефти
    :return: fvf(*float*) Объёмный фактор
    """
    if pressure_mpa <= bubble_point_pressure_mpa:
        fvf = fvf_saturated_oil
    else:
        fvf = fvf_saturated_oil * math.exp(
            oil_compressibility * (bubble_point_pressure_mpa - pressure_mpa)
        )

    return fvf


def fvf_oil_standing(
    p_mpa: float,
    temperature_k: float,
    gamma_o: float,
    gamma_g: float,
    r_sb: float,
    p_rb: float,
    b_ro: float,
) -> float:
    """Рассчитывает объёмный фактор нефти по Стэндингу

    :param float p_mpa: Давление (Мпа)
    :param float temperature_k: Температура (К)
    :param float gamma_o: Относительный вес нефти
    :param float gamma_g: Относительный вес газа
    :param float r_sb: Калибровочный газовый фактор в точке насыщения
                       (scf/stb, м3/м3)
    :param float p_rb: Калибровочное давление насыщения (psi, атм)
    :param float b_ro: Калибровочный объёмный фактор нефти в точке насыщения
                       (доли)
    :return: fvf(*float*) Объёмный фактор
    """

    p_bi = bubble_point_standing(r_sb, gamma_g, temperature_k, gamma_o)

    #  'calculate bubble point correction factor
    if p_rb > 0:  # 'user specified
        p_fact = p_bi / p_rb
        p_offs = p_bi - p_rb
    else:  # ' not specified, use from correlations
        p_fact = 1
        p_offs = 0
    # 'calculate oil formation volume factor correction factor
    if b_ro > 0:  # 'user specified
        b_o_sat = fvf_saturated_oil_standing(
            r_sb, gamma_g, temperature_k, gamma_o
        )
        b_fact = (b_ro - 1) / (b_o_sat - 1)
    else:  # ' not specified, use from correlations
        b_fact = 1
    if p_mpa > (p_bi / p_fact):  # Недонасыщенная нефть
        p_mpa += p_offs
        r_si = r_sb
        #    'standing
        b_o_sat = (
            b_fact
            * (
                fvf_saturated_oil_standing(
                    r_si, gamma_g, temperature_k, gamma_o
                )
                - 1
            )
            + 1
        )  # ' it is assumed that at pressure 1 atm bo=1

        c_o = compressibility_oil_vb(
            r_sb, gamma_g, temperature_k, gamma_o, p_mpa
        )
        fvf = b_o_sat * math.exp(c_o * (p_bi - p_mpa))
    else:  # 'saturated oil
        # 'apply correction to saturated oil
        p_mpa *= p_fact

        r_si = gor_standing(p_mpa, gamma_g, temperature_k, gamma_o)

        fvf = (
            b_fact
            * (
                fvf_saturated_oil_standing(
                    r_si, gamma_g, temperature_k, gamma_o
                )
                - 1
            )
            + 1
        )  # ' it is assumed that at pressure 1 atm bo=1

    return fvf


def dead_oil_viscosity_standing(
    temperature_k: float, gamma_oil: float
) -> float:
    """Рассчитывает вязкость дегазированный нефти

    :param float temperature_k: Температура (К)
    :param float gamma_oil: Относительный вес нефти
    :return: mu(*float*) Вязкость (сР)
    """
    mu: float = 0.32 + 1.8 * pow(10, 7) / pow((141.5 / gamma_oil - 131.5), 4.53)
    mu *= pow(
        (360 / (1.8 * temperature_k - 260)),
        (pow(10, (0.43 + 8.33 / (141.5 / gamma_oil - 131.5)))),
    )
    return mu


def oil_viscosity_standing(
    rs_m3m3: float,
    dead_oil_viscosity: float,
    pressure_mpa: float,
    bubble_point_pressure_mpa: float,
) -> float:
    """Рассчитывает вязкость нефти по Стэндингу

    :param float rs_m3m3: Газосодержание нефти (м3/м3)
    :param float dead_oil_viscosity: Вязкость дегазированной нефти (сР)
    :param float pressure_mpa: Давление (МПа)
    :param float bubble_point_pressure_mpa: Давление неасыщения (МПа)
    :return: mu(*float*) Вязкость (сР)
    """
    a = 5.6148 * rs_m3m3 * (0.1235 * pow(10, -5) * rs_m3m3 - 0.00074)
    b = (
        0.68 / pow(10, (0.000484 * rs_m3m3))
        + 0.25 / pow(10, (0.006176 * rs_m3m3))
        + 0.062 / pow(10, (0.021 * rs_m3m3))
    )

    mu: float = pow(10, a) * pow(dead_oil_viscosity, b)

    if bubble_point_pressure_mpa < pressure_mpa:
        mu += (
            0.14504
            * (pressure_mpa - bubble_point_pressure_mpa)
            * (0.024 * pow(mu, 1.6) + 0.038 * pow(mu, 0.56))
        )
    return mu


def dead_oil_viscosity_two_points(
    mu_oil_20: float, mu_oil_50: float, t_pvt: float, gamma_o: float
) -> float:

    mu_oil_20 /= gamma_o
    mu_oil_50 /= gamma_o
    koef = math.log(mu_oil_50 / mu_oil_20) / (20 - 50)
    dov = mu_oil_50 * math.exp(-koef * (t_pvt - 50)) * gamma_o

    return dov


def dead_oil_viscosity_beggs_robinson(
    temperature_k: float, gamma_oil: float
) -> float:
    """Рассчитывает вязкость дегазированной нефти по Беггзу и Робинсону

    :param float temperature_k: Температура (К)
    :param float gamma_oil: Относительный вес газа
    :return: mu(*float*) Вязкость (cP)
    """

    x = pow((1.8 * temperature_k - 460), (-1.163)) * math.exp(
        13.108 - 6.591 / gamma_oil
    )
    mu = pow(10, x) - 1
    return mu


def saturated_oil_viscosity_beggs_robinson(
    gor_pb_m3m3: float, dead_oil_viscosity: float
) -> float:
    """Рассчитывает вязкость насыщенной нефти по Беггзу и Робинсону

    :param float gor_pb_m3m3: Газосодержание в точке насыщения (м3/м3)
    :param float dead_oil_viscosity: Вязкость дегазированной нефти (сР)
    :return: mu(*float*) Вязкость (сР)
    """
    a = 10.715 * pow((5.615 * gor_pb_m3m3 + 100), (-0.515))
    b = 5.44 * pow((5.615 * gor_pb_m3m3 + 150), (-0.338))
    mu = a * pow(dead_oil_viscosity, b)
    return mu


def oil_viscosity_vasquez_beggs(
    saturated_oil_viscosity: float, pressure_mpa: float, bp_pressure_mpa: float
) -> float:
    """Рассчитывает вязкость нефти по Васкезу и Беггзу

    :param float saturated_oil_viscosity: Вязкость насыщенной нефти
    :param float pressure_mpa: Давление (МПа)
    :param float bp_pressure_mpa: Давление насыщения (МПа)
    :return: mu(*float*) Вязкость (cP)
    """
    c1 = 957
    c2 = 1.187
    c3 = -11.513
    c4 = -0.01302
    m = c1 * pow(pressure_mpa, c2) * math.exp(c3 + c4 * pressure_mpa)
    mu = saturated_oil_viscosity * pow((pressure_mpa / bp_pressure_mpa), m)
    return mu


def viscosity_grace(
    pressure_mpa: float,
    bubble_point_pressure_mpa: float,
    density_kg_m3: float,
    bp_density_kg_m3: float,
) -> float:
    """Рассчитывает вязкость нефти по Грэйсу

    :param float pressure_mpa: Давление (МПа)
    :param float bubble_point_pressure_mpa: Давление насыщения (МПа)
    :param float density_kg_m3: Плотность нефти (кг/м3)
    :param float bp_density_kg_m3: Плотность нефти в точке насыщения (кг/м3)
    :return: mu(*float*) Вязкость (сР)
    """
    density = density_kg_m3 * 0.06243
    bubblepoint_density = bp_density_kg_m3 * 0.06243
    rotr = (
        0.0008 * pow(density, 3)
        - 0.1017 * pow(density, 2)
        + 4.3344 * density
        - 63.001
    )
    mu = math.exp(
        0.0281 * pow(rotr, 3) - 0.0447 * pow(rotr, 2) + 1.2802 * rotr + 0.0359
    )

    if bubble_point_pressure_mpa < pressure_mpa:
        robtr = (
            -68.1067 * pow(math.log(bubblepoint_density), 3)
            + 783.2173 * pow(math.log(bubblepoint_density), 2)
            - 2992.2353 * math.log(bubblepoint_density)
            + 3797.6
        )
        m = math.exp(
            0.1124 * pow(robtr, 3)
            - 0.0618 * pow(robtr, 2)
            + 0.7356 * robtr
            + 2.3328
        )
        mu *= pow((density_kg_m3 / bp_density_kg_m3), m)

    return mu


def oil_viscosity_beggs_robinson_vasques_beggs(
    rs_m3m3: float,
    gor_pb_m3m3: float,
    pressure_mpa: float,
    bubble_point_pressure_mpa: float,
    dead_oil_viscosity: float,
) -> float:
    """Рассчитывает вязкость нефти по Беггзу, Робинсону, Васкезу и Беггзу

    :param float rs_m3m3: Газосодержание (м3/м3)
    :param float gor_pb_m3m3: Газосодержание в точке насыщения (м3/м3)
    :param float pressure_mpa: Давление (МПа)
    :param float bubble_point_pressure_mpa: Давление в точке насышения (МПа)
    :param float dead_oil_viscosity: Вязкость дегазированноый нефти (сР)
    :return: mu(*float*) Вязкость (сР)
    """
    if pressure_mpa < bubble_point_pressure_mpa:  # 'saturated
        mu = saturated_oil_viscosity_beggs_robinson(rs_m3m3, dead_oil_viscosity)
    else:  # Недонасыщенная нефть
        mu = oil_viscosity_vasquez_beggs(
            saturated_oil_viscosity_beggs_robinson(
                gor_pb_m3m3, dead_oil_viscosity
            ),
            pressure_mpa,
            bubble_point_pressure_mpa,
        )

    return mu


def compressibility_oil_vb(
    rs_m3m3: float,
    gamma_gas: float,
    temperature_k: float,
    gamma_oil: float,
    pressure_mpa: float,
) -> float:
    """Рассчитывает сжимаемость нефти

    :param float rs_m3m3: Газосодержание (м3/м3)
    :param float gamma_gas: Относительный вес газа
    :param float temperature_k: Темпераутра (К)
    :param float gamma_oil: Относительный вес нефти
    :param float pressure_mpa: Давление (МПа)
    :return: c_o(*float*) Сжимаемость (1/МПа)
    """
    c_o = (
        28.1 * rs_m3m3
        + 30.6 * temperature_k
        - 1180 * gamma_gas
        + 1784 / gamma_oil
        - 10910
    ) / (100000 * pressure_mpa)
    return c_o


def density_oil_standing(
    rs_m3_m3: float,
    gamma_gas: float,
    gamma_oil: float,
    pressure_mpa: float,
    fvf_m3_m3: float,
    bubble_point_pressure_mpa: float,
    compressibility_1mpa: float,
) -> float:
    """Рааситывает плотность нефти по Стэндингу

    :param float rs_m3_m3: Газосодержание (м3/м3)
    :param float gamma_gas: Относительный вес газа
    :param float gamma_oil: Относительный вес нефти
    :param float pressure_mpa: Давление (МПа)
    :param float fvf_m3_m3: Объёмный фактор
    :param float bubble_point_pressure_mpa: Давление насыщения (МПа)
    :param float compressibility_1mpa: Сжимаемость (1/МПа)
    :return: rho(*float*) Плотность нефти (кг/м3)
    """

    rho = (1000 * gamma_oil + 1.224 * gamma_gas * rs_m3_m3) / fvf_m3_m3

    if pressure_mpa > bubble_point_pressure_mpa:
        rho *= math.exp(
            compressibility_1mpa * (pressure_mpa - bubble_point_pressure_mpa)
        )

    return rho


def density_mccain_si(
    pressure_mpa: float,
    gamma_gas: float,
    temperature_k: float,
    gamma_oil: float,
    rs_m3_m3: float,
    bubble_point_pressure_mpa: float,
    compressibility_1mpa: float,
) -> float:
    """Рааситывает плотность нефти по МакКейну

    :param float rs_m3_m3: Газосодержание (м3/м3)
    :param float gamma_gas: Относительный вес газа
    :param float temperature_k: Темепература
    :param float gamma_oil: Относительный вес нефти
    :param float pressure_mpa: Давление (МПа)
    :param float bubble_point_pressure_mpa: Давление насыщения (МПа)
    :param float compressibility_1mpa: Сжимаемость (1/МПа)
    :return: rho(*float*) Плотность нефти (кг/м3)
    """

    if rs_m3_m3 > 800:
        rs_m3_m3 = 800
        bubble_point_pressure_mpa = bubble_point_valko_mccain_si(
            rs_m3_m3, gamma_gas, temperature_k, gamma_oil
        )

    ropo = 845.8 - 0.9 * rs_m3_m3
    pm = ropo
    epsilon = 0.000001
    maxiter = 1000
    for i in range(0, maxiter):
        pmmo = pm
        a0 = -799.21
        a1 = 1361.8
        a2 = -3.70373
        a3 = 0.003
        a4 = 2.98914
        a5 = -0.00223
        roa = (
            a0
            + a1 * gamma_gas
            + a2 * gamma_gas * ropo
            + a3 * gamma_gas * pow(ropo, 2)
            + a4 * ropo
            + a5 * pow(ropo, 2)
        )
        ropo = (rs_m3_m3 * gamma_gas + 818.81 * gamma_oil) / (
            0.81881 + rs_m3_m3 * gamma_gas / roa
        )
        pm = ropo

        if abs(pmmo - pm) <= epsilon:
            break

    if pressure_mpa <= bubble_point_pressure_mpa:
        dpp = (0.167 + 16.181 * (pow(10, (-0.00265 * pm)))) * (
            2.32328 * pressure_mpa
        ) - 0.16 * (0.299 + 263 * (pow(10, (-0.00376 * pm)))) * pow(
            (0.14503774 * pressure_mpa), 2
        )
        pbs = pm + dpp
        dpt = (0.04837 + 337.094 * pow(pbs, (-0.951))) * pow(
            (1.8 * temperature_k - 520), 0.938
        ) - (0.346 - 0.3732 * (pow(10, (-0.001 * pbs)))) * pow(
            (1.8 * temperature_k - 520), 0.475
        )
        pm = pbs - dpt
        rho = pm
    else:
        dpp = (0.167 + 16.181 * (pow(10, (-0.00265 * pm)))) * (
            2.32328 * bubble_point_pressure_mpa
        ) - 0.16 * (0.299 + 263 * (pow(10, (-0.00376 * pm)))) * pow(
            (0.14503774 * bubble_point_pressure_mpa), 2
        )
        pbs = pm + dpp
        dpt = (0.04837 + 337.094 * pow(pbs, (-0.951))) * pow(
            (1.8 * temperature_k - 520), 0.938
        ) - (0.346 - 0.3732 * (pow(10, (-0.001 * pbs)))) * pow(
            (1.8 * temperature_k - 520), 0.475
        )
        pm = pbs - dpt
        rho = pm * math.exp(
            compressibility_1mpa * (pressure_mpa - bubble_point_pressure_mpa)
        )

    return rho


def oil_volume_rate(
    mass_rate: float, gamma_o: float, gamma_g: float, r_s: float, b_o: float
) -> float:
    """Рассчитывает объёмный расход нефти

    :param float mass_rate: Массовый расход нефти (тыс. тонн / сут.)
    :param float gamma_o: Относительный вес нефти
    :param float gamma_g: Относительный вес газа
    :param float r_s: Газосодержание (м3/м3)
    :param float b_o: Объёмный фактор (доли)
    :return: rate*float* Объёмный расход нефти
    """
    rho_wsc = 1000
    rho_air = 1.2217
    rho_osc = rho_wsc * gamma_o
    rho_gsc = rho_air * gamma_g
    rho_o = (rho_osc + r_s * rho_gsc) / b_o
    mass_rate *= 1e6
    return mass_rate / rho_o
