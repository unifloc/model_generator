import os
from ecl.summary import EclSum
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import subprocess
from plotly.offline import iplot
from jinja2 import Environment, FileSystemLoader
from ecl2df import grid, EclFiles
import plotly.express as px
import math as m
from scipy.interpolate import interp1d

"""
@alexeyvodopyan
@sevrn
"""

# метод для очистки результатов расчетов
def clear_folders():
    subprocess.call("rm -f model_folder/*", shell=True)
    subprocess.call("rm -f csv_folder/*", shell=True)


class ModelGenerator:
    """
    Класс для расчета модели
    """

    def __init__(self, start_date="1 'SEP' 2020", mounths=2, days=30, nx=100, ny=100, nz=5, dx=100, dy=100, dz=5, por=0.3, permx=100,
                 permy=100, permz=10, prod_names=None, prod_xs=None, prod_ys=None, prod_z1s=None, prod_z2s=None, prod_q_oil=None,
                 inj_names=None, inj_xs=None, inj_ys=None, inj_z1s=None, inj_z2s=None, inj_bhp=None, skin=None, oil_den=860, wat_den=1010, gas_den=0.9,
                 p_depth=2500, p_init=250, o_w_contact=2600, pc_woc=0, g_o_contact=2450, pc_goc=0, tops_depth=2500, rock_compr=1.5E-005,
                 rezim='ORAT', prod_bhp=None, horizontal=False, y_stop=None, only_prod=False,
                 lgr=False, lx=None, ly=None, cells_cy=None, cells_v=None, cells_cx=None,
                 upr_rezim_water=False, upr_rezim_gas=False, rw=None, template=1, neogr=False,
                 grp=False, nz_grp=1, xs_start_grp=1, xs_stop_grp=2, ys_grp=1, k_grp=100, roughness=False, wellstep=180, kust=4, bur_ust=None):

        # продолжительность расчета
        self.start_date = f'{start_date}'
        self.tstep = ''

        # размеры модели
        self.dimens = f'{nx} {ny} {nz}'
        self.dx = f'{nx*ny*nz}*{dx} /'
        self.dy = f'{nx*ny*nz}*{dy} /'
        self.dz = f'{nx*ny*nz}*{dz}'
        if template == 2 or template == 4: 
            self.dz = f'DZ {dz} / \n'
            self.dz += '/'
        self.top_box = ''
        if template == 2 or template == 4:
            self.top_box = 'BOX \n'
            self.top_box += f'1 {nx} 1 {ny} 1 1 /'
        self.tops_depth = f'{nx*ny}*{tops_depth} '

        # LGR
        if lgr == True:
            if template == 1 or template == 3:
                # формируем измельченную сетку по x
                dx_lgr = self.setcas(nx, lx, cells_cx, cells_v)
                self.dx = str(dx_lgr[0]) + ' \n'
                for i in range(2, nx + 1):    
                    self.dx += str(dx_lgr[i-1]) + ' \n'
                self.dx = self.dx*ny
                self.dx += '/\n'

                # формируем измельченную сетку по y
                dy_lgr = self.setcas(ny, ly, cells_cy, cells_v)
                self.dy = ''
                for i in range(1, ny + 1):    
                    dim = str(dy_lgr[i-1]) + ' \n'
                    self.dy += dim*nx
                self.dy += '/\n'
            elif template == 2 or template == 4:
                # формируем измельченную сетку по x
                dx_lgr = self.setcas(nx, lx, cells_cx, cells_v)
                self.dx = 'DX ' + str(dx_lgr[0]) + ' 1 1 /\n'
                for i in range(2, nx + 1):    
                    self.dx += 'DX ' + str(dx_lgr[i-1]) + f' {i} {i} /\n'
                self.dx += '/\n'

                # формируем измельченную сетку по y
                dy_lgr = self.setcas(ny, ly, cells_cy, cells_v)
                self.dy = ''
                for i in range(1, ny + 1):    
                    self.dy += 'DY ' + str(dy_lgr[i-1]) + f' 2* {i} {i} /\n'
                self.dy += '/\n'

        # физические свойства
        self.por = f'{nx*ny*nz}*{por}'
        if template == 1 or template == 3:
            self.permx = f'{nx*ny*nz}*{permx}'
            self.permy = f'{nx*ny*nz}*{permy}'
            self.permz = f'{nx*ny*nz}*{permz}'
        elif template == 2 or template == 4:
            self.permx = f'PERMX {permx} 6*/ \n'
            self.permy = f'PERMY {permy} 6*/ \n'
            self.permz = f'PERMZ {permz} 6*/ \n'


        # EQUILIBRIUM DATA
        self.equil = f'{p_depth} {p_init} {o_w_contact} {pc_woc} {g_o_contact} {pc_goc} 1 1* 1*  /'

        # свойства продукции
        self.density = f'{oil_den} {wat_den} {gas_den} /'

        # свойства породы
        self.rock = f'{p_init} {rock_compr}'

        # индикаторы режимов (в разработке)
        self.only_prod = only_prod # оставляет только добывающие скважины
        self.upr_rezim_water = upr_rezim_water # умножает на 1000 пористость последнего пропласта (предварительно необходимо установить ВНК)
        self.upr_rezim_gas = upr_rezim_gas # умножает на 100 пористость первого пропластка (предварителньо необходимо установить ГНК)
        self.neogr = neogr

        # умножение порового объема (неограниченный пласт)
        self.poro_box = ''
        if template == 2 or template == 4: 
            self.poro_box = 'BOX \n'
            self.poro_box += f'1 {nx} 1 {ny} 1 {nz} /'
        self.por = ''
        if self.upr_rezim_water and self.upr_rezim_gas and not self.neogr:
            self.por = str(nx*ny) + '*1000' + ' ' + str(nx*ny*(nz-2)) + '*' + str(por) + ' ' + str(nx*ny) + '*1000' +  ' /'
        elif self.upr_rezim_gas and not self.upr_rezim_water and not self.neogr:
            self.por = str(nx*ny) + '*1000' + ' ' + str(nx*ny*(nz-1)) + '*' + str(por) +  ' /'
        elif self.upr_rezim_water and not self.upr_rezim_gas and not self.neogr:
            self.por = str(nx*ny*(nz-1)) + '*' + str(por) + ' ' + str(nx*ny) + '*1000' +  ' /'
        elif self.neogr and self.upr_rezim_water and self.upr_rezim_gas:
            for j in range(0, nz):
                val = por 
                if j == 0: val = 1000
                if j == nz-1: val = 1000
                self.por += str(nx+1) + '*' + str(1000) + ' \n'
                for i in range(0, ny-3):
                    self.por += str(nx-2) + '*' + str(val) + ' \n'
                    self.por += str(2) + '*' + str(1000) + ' \n'
                self.por += str(nx-2) + '*' + str(val) + ' \n'
                self.por += str(nx+1) + '*' + str(1000) + ' \n'
        elif self.neogr and self.upr_rezim_water and not self.upr_rezim_gas:
            for j in range(0, nz):
                val = por 
                if j == nz-1: val = 1000
                self.por += str(nx+1) + '*' + str(1000) + ' \n'
                for i in range(0, ny-3):
                    self.por += str(nx-2) + '*' + str(val) + ' \n'
                    self.por += str(2) + '*' + str(1000) + ' \n'
                self.por += str(nx-2) + '*' + str(val) + ' \n'
                self.por += str(nx+1) + '*' + str(1000) + ' \n'
        elif self.neogr and not self.upr_rezim_water and self.upr_rezim_gas:
            for j in range(0, nz):
                val = por 
                if j == 0: val = 1000
                self.por += str(nx+1) + '*' + str(1000) + ' \n'
                for i in range(0, ny-3):
                    self.por += str(nx-2) + '*' + str(val) + ' \n'
                    self.por += str(2) + '*' + str(1000) + ' \n'
                self.por += str(nx-2) + '*' + str(val) + ' \n'
                self.por += str(nx+1) + '*' + str(1000) + ' \n'
        elif self.neogr and not self.upr_rezim_water and not self.upr_rezim_gas:
            for j in range(0, nz):
                self.por += str(nx+1) + '*' + str(1000) + ' \n'
                for i in range(0, ny-3):
                    self.por += str(nx-2) + '*' + str(por) + ' \n'
                    self.por += str(2) + '*' + str(1000) + ' \n'
                self.por += str(nx-2) + '*' + str(por) + ' \n'
                self.por += str(nx+1) + '*' + str(1000) + ' \n'
        elif not self.neogr and not self.upr_rezim_water and not self.upr_rezim_gas:
            self.por += str(nx*ny*nz) + '*' + str(por) +  ' /'

        # SCHEDULE секция
        if not only_prod:
            all_well_names = prod_names + inj_names
            all_well_xs = prod_xs + inj_xs
            all_well_ys = prod_ys + inj_ys
            all_well_z1s = prod_z1s + inj_z1s
            all_well_z2s = prod_z2s + inj_z2s
            all_well_fluid = ['OIL' for _ in range(len(prod_names))] + ['WAT' for _ in range(len(inj_names))]
            rezim = rezim + ['BHP']*len(inj_names)
            well_q = prod_q_oil + ['']*len(inj_names)
            well_bhp = prod_bhp + inj_bhp
        else:
            all_well_names = prod_names
            all_well_xs = prod_xs
            all_well_ys = prod_ys 
            all_well_z1s = prod_z1s
            all_well_z2s = prod_z2s
            all_well_fluid = ['OIL' for _ in range(len(prod_names))]
        
        # пересобираем порядок ввода с учетом кустования
        if bur_ust != None:
            pos = []
            i, j, indic = 0, 0, 0

            k = 0
            while indic <= len(all_well_names):
                for name in all_well_names:
                    for l in range(kust):
                        try:
                                bur_ust[k][j]
                        except:
                            k += 1
                    if name == bur_ust[k][j]:
                        pos.append(i)
                        k += 1
                        indic += 1
                    if indic == len(all_well_names):
                            break
                    i += 1
                    if k == 3:
                        k = 0
                        j += 1
                i = 0
                if indic == len(all_well_names):
                            break
            names_new, x_new, y_new, rezim_new, z1_new, z2_new, q_new, bhp_new, rw_new, skin_new, new_fluid = [],[],[],[],[],[],[],[],[],[], []
            for i in pos:
                names_new.append(all_well_names[i])
                x_new.append(all_well_xs[i])
                y_new.append(all_well_ys[i])
                rezim_new.append(rezim[i])
                z1_new.append(all_well_z1s[i])
                z2_new.append(all_well_z2s[i])
                q_new.append(well_q[i])
                bhp_new.append(well_bhp[i])
                rw_new.append(rw[i])
                skin_new.append(skin[i])
                new_fluid.append(all_well_fluid[i])
            all_well_names = names_new
            all_well_xs = x_new
            all_well_ys = y_new
            rezim = rezim_new
            all_well_z1s = z1_new
            all_well_z2s = z2_new
            well_q = q_new
            well_bhp = bhp_new
            rw = rw_new
            skin = skin_new
            all_well_fluid = new_fluid
            
        self.welspecs = ''
        self.wconprod = ''
        self.compdat = ''
        self.wconinje = ''
        self.wellstep_indic = 0
        self.wellstep = ''
        if wellstep == 0:
            self.tstep = ''f'{mounths}*{days}'
            for name, x, y, fluid in zip(all_well_names, all_well_xs,
                                                all_well_ys, all_well_fluid):
                if template == 2 or template == 4: name = f'"{name}"'
                self.welspecs += name + ' G1 ' + str(x) + ' ' + str(y) + ' 1* ' + fluid + ' /\n'

            for x, name, y, z1, z2, skin, rw in zip(all_well_xs, all_well_names, all_well_ys,
                                                all_well_z1s, all_well_z2s, skin, rw):
                if template == 2 or template == 4: name = f'"{name}"'
                if horizontal:
                    self.compdat = name + ' ' + str(x) + ' ' + str(y) + ' ' + str(z2) + ' ' + str(z2) + ' OPEN	1*	1* ' + str(rw) +  ' 1* ' + str(skin) + ' 1* Y /\n' 
                    for i in range(y+1, y_stop[0]+1):
                        self.compdat += name + ' ' + str(x) + ' ' + str(i) + ' ' + str(z2) + ' ' + str(z2) + ' OPEN	1*	1* ' + str(rw) +  ' 1* ' + str(skin) + ' 1* Y /\n'
                else:
                    self.compdat += name + ' ' + str(x) + ' ' + str(y) + ' ' + str(z1) + ' ' + str(z2) + ' OPEN	1*	1*	' + str(rw) +  ' 1* ' + str(skin) + ' /\n'

            for prod, rezim, q_oil, prod_bhp in zip(prod_names, rezim, prod_q_oil, prod_bhp):
                if template == 2 or template == 4: prod = f'"{prod}"'
                self.wconprod += prod + ' OPEN ' + rezim + ' ' + str(q_oil) + ' 4* ' + str(prod_bhp) + ' /\n'

            if not only_prod:
                for inj, inj_bhp in zip(inj_names, inj_bhp):
                    if template == 2 or template == 4: inj = f'"{inj}"'
                    self.wconinje += inj + ' WAT OPEN BHP ' + str(inj_bhp) + ' 1* /'
        else:
            self.prod_num = 0
            self.inj_num = 0
            for fluid, x, name, y, z1, z2, skin, rw in zip(all_well_fluid, all_well_xs, all_well_names, all_well_ys,
                                                all_well_z1s, all_well_z2s, skin, rw):
                self.wellstep += 'WELSPECS \n'
                self.wellstep += name + ' G1 ' + str(x) + ' ' + str(y) + ' 1* ' + fluid + ' /\n'
                self.wellstep += '/\n\n'
                self.wellstep += 'COMPDAT \n'
                self.wellstep += name + ' ' + str(x) + ' ' + str(y) + ' ' + str(z1) + ' ' + str(z2) + ' OPEN	1*	1*	' + str(rw) +  ' 1* ' + str(skin) + ' /\n'
                self.wellstep += '/\n\n'
                if fluid == 'OIL':
                    self.wellstep += 'WCONPROD \n'
                    self.wellstep += name + ' OPEN ' + rezim[self.prod_num] + ' ' + str(prod_q_oil[self.prod_num]) + ' 4* ' + str(prod_bhp[self.prod_num]) + ' /\n'
                    self.wellstep += '/\n\n'
                    self.prod_num += 1
                elif fluid == 'WAT':
                    self.wellstep += 'WCONINJE \n'
                    self.wellstep += name + ' WAT OPEN BHP ' + str(inj_bhp[self.inj_num]) + ' 1* /\n'
                    self.wellstep += '/\n\n'
                    self.inj_num += 1
                self.wellstep_indic += 1
                if self.wellstep_indic%kust == 0:
                    self.wellstep += 'TSTEP\n'
                    self.wellstep += f'1*{wellstep} /\n\n'
            
            if self.wellstep_indic%kust != 0:
                self.wellstep += 'TSTEP\n'
                self.wellstep += f'1*{wellstep} /\n\n'
                self.wellstep_indic += 1

            self.wellstep += 'TSTEP\n'
            self.wellstep += f'400*{(mounths*days-wellstep*self.wellstep_indic/kust)/400} /\n\n'


        self.template = template # выбираем шаблон data файла для различных симуляторов
        # templates: 1-opm; 2-ecl (в разработке)

        # Моделирование ГРП:
        self.grp = grp
        self.nz_grp = nz_grp
        self.xs_start_grp = xs_start_grp
        self.xs_stop_grp = xs_stop_grp
        self.ys_grp = ys_grp
        self.k_grp = k_grp

        self.grp_word = ''
        if grp == True:
            self.grp_word = 'EQUALS \n'
            self.grp_word += f"'PERMX' {k_grp} {xs_start_grp} {xs_stop_grp} {ys_grp} {ys_grp} {nz_grp} {nz_grp} /\n"
            self.grp_word += f"'PERMY' {k_grp} {xs_start_grp} {xs_stop_grp} {ys_grp} {ys_grp} {nz_grp} {nz_grp} /\n"
            self.grp_word += '/'

        # Моделирование потерь на трение
        self.welsegs = ''
        self.compsegs = ''
        if roughness:
            for name in zip(all_well_names):
                self.welsegs += f'"{name}" {tops_depth+z2} 0.0 1* INC HF- /\n'
                self.welsegs += f'2 {y_stop[0]}  1 1 {y_stop[0]+all_well_z2s[0]} 0 {rw*2} {roughness} /\n'
                self.compsegs += f'"{name}" /\n'
                self.compsegs +=  f' {all_well_xs[0]} {all_well_ys[0]} {all_well_z2s[0]} 1 1 1* Y {all_well_ys[-1]} /\n'

        # переменные для расчета
        self.result_df = None
        self.fig = None
        self.fig_npv = None
        self.dir = None
        self.result_data = None
        self.create_data_file()

        # переменный для рисования
        self.grid_dx = nx
        self.grid_dy = ny
     
    def create_data_file(self):
        if self.template == 1:
            template_name = 'templates/opm.DATA'
        elif self.template == 2:
            template_name = 'templates/ecl.DATA'
        elif self.template == 3:
            template_name = 'templates/opm_multphase.DATA'
        elif self.template == 4:
            template_name = 'templates/ecl_multphase.DATA'
        env = Environment(loader=FileSystemLoader(''))
        template = env.get_template(template_name)
        self.result_data = template.render(DIMENS=self.dimens, START=self.start_date,
            DX=self.dx, DY=self.dy, DZ=self.dz, TOP_BOX=self.top_box, TOPS=self.tops_depth, PORO_BOX=self.poro_box, PORO=self.por,
            PERMX=self.permx, PERMY=self.permy, PERMZ=self.permz, ROCK=self.rock,  DENSITY=self.density,
            EQUIL=self.equil, WELSPECS=self.welspecs, COMPDAT=self.compdat,
            WCONPROD=self.wconprod, WCONINJE=self.wconinje, TSTEP=self.tstep, GRP=self.grp_word, WELSEGS=self.welsegs,
            COMPSEGS=self.compsegs, WELLSTEP=self.wellstep)


    def create_model(self, name, result_name, keys):
        self.save_file(name=name)
        if self.template == 1 or self.template == 3:
            self.calculate_file(name)
            self.create_result(name=name, keys=keys)
            self.read_result(name=result_name)


    def save_file(self, name):
        with open('model_folder/'+ name + '.DATA', "w") as file:
            file.write(self.result_data)

    @staticmethod
    def calculate_file(name):
        os.system("mpirun flow model_folder/%s.DATA" % name)

    @staticmethod
    def create_result(name, keys):
        summary = EclSum('model_folder/%s.DATA' % name)
        dates = summary.dates
        results = []
        all_keys = []

        if keys is None:
            keys = ["WOPR:*"]

        for key in keys:
            key_all_wells = summary.keys(key)
            all_keys = all_keys + list(key_all_wells)

        for key in all_keys:
            results.append(list(summary.numpy_vector(key)))

        if len(results) == 0:
            return print('Результаты из модели не загрузились. Файл с результатами не был создан')

        result_df = pd.DataFrame(data=np.array(results).T, index=dates, columns=all_keys)
        result_df.index.name = 'Time'
        result_df.to_csv('csv_folder/%s.csv' % name)
        print('%s_RESULT.csv is created' % name)


    def read_result(self, name):
        self.result_df = pd.read_csv('csv_folder/%s.csv' % name, parse_dates=[0], index_col=[0])
        print('%s.csv is read' % name)
        

    # метод построения графика по заданному параметру
    def summ_plot(self, parameters=None, mode='lines', x_axis=None, y_axis=None, title=None, name=None):
        directory = "csv_folder/"
        self.fig = go.Figure()
        files = [f for f in os.listdir(directory)]
        files.sort(key=lambda x:int(x.split('.')[1]))
        i = 0
        for file in files:
            df = pd.read_csv('csv_folder/%s' % file, parse_dates=[0], index_col=[0])
            # i = int(file.split('.')[1])
            self.fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[parameters[0]],
                    mode=mode,line_shape='spline',
                    name='Модель:' + name[i]))
            i += 1
        if x_axis is None:
            x_axis = 'Дата'
        if y_axis is None:
            y_axis = ''
        if title is None:
            title = 'Динамика показателей по моделям'
        self.fig.update_xaxes(tickformat='%d.%m.%y')
        self.fig.update_layout(title=go.layout.Title(text=title),
                               xaxis_title=x_axis,
                               yaxis_title=y_axis)
        iplot(self.fig)


    # метод для формирования измельченной сетки
    # необходимо помнить, что размеры ячеек должны отличаться не более чем в 2 раза
    @staticmethod     
    def setcas(nx, lx, s, v):
        k = 1
        l = 0
        delta = lx*0.01
        while abs(l - lx) > delta:
            if k <= 2:
                k += 0.001
            else:
                print('Невозможно разбить сетку!')
                raise Exception() 
            n = round((nx - s) / 2)
            l = v*s + 2 * (v * k * (k ** n - 1) / (k - 1))
            if abs(l - lx) < delta:
                x = []
                for i in range(1, n + 1):
                    x.append(round(v * k ** i, 3))
                x = x[::-1] + [v] * s + x
                return x


    @staticmethod     
    def grid_df(filename, parametr, dx, dy):
        eclfiles = EclFiles(f'model_folder/{filename}.DATA')
        dates = 'last'
        dframe = grid.df(eclfiles, rstdates=dates)
        df = []
        k = dx
        for i in range(0, dy):
            ra = []
            for j in range(0, dx):
                ra.append(dframe[str(parametr)].iloc[j+i*k])
            df.append(ra)
        
        return df


    def grid_plot(self, filename, parametr, title=''):
        df = self.grid_df(filename, parametr, self.grid_dx, self.grid_dy)
        self.fig = px.imshow(df, labels=dict(x="nx", y="ny", color=title))
        iplot(self.fig)


    # # метод для построения графиков на основе summary (тест)
    # def summary_df():
    #     from ecl2df import summary, EclFiles

    #     eclfiles = EclFiles("model_folder/TEST_MODEL_HORIZONTAL.0.DATA")
    #     dframe = summary.df(eclfiles, column_keys="FOPT", time_index="monthly") # daily
