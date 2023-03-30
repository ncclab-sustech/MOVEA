#encoding: utf-8
import random
import numpy as np
from public import pareto,NDsort,glo

def init_designparams(particals,in_min,in_max):
    if glo.type == 'ti':
        in_dim = len(in_max)   
        solution = np.zeros((1, 6))
        solution[0, 0] = 0.5 + glo.prior[4]/75
        solution[0, 1] = 0.5 - glo.prior[4]/75
        solution[0, 2] = glo.prior[0] / 74
        solution[0, 3] = glo.prior[1] / 74
        solution[0, 4] = glo.prior[2] / 74
        solution[0, 5] = glo.prior[3] / 74
        print(solution)
        solution=np.repeat(solution, 5, axis=0)
        in_temp = np.random.uniform(-5, 5, (particals-5, in_dim))
        in_temp[-1 > in_temp] = 0
        in_temp[in_temp > 1] = 0
        in_temp = np.vstack([in_temp,solution])
        print(in_temp)
    if glo.type == 'mti':    
        in_dim = len(in_max)     #输入参数维度
        print(in_dim)
        print(glo.prior)
        solution = np.zeros(150)
        solution[glo.prior[0]] = 1
        solution[glo.prior[1]] = 1
        solution[glo.prior[2]] = -1
        solution[glo.prior[3]] = -1
        solution[glo.prior[0]+75] = 1
        solution[glo.prior[1]+75] = 1
        solution[glo.prior[2]+75] = -1
        solution[glo.prior[3]+75] = -1
        print(solution)
        solution=np.repeat([solution], 5, axis=0)
        in_temp = np.random.uniform(-10, 10, (particals-5, in_dim))
        in_temp[-1 > in_temp] = 0
        in_temp[in_temp > 1] = 0
        in_temp = np.vstack([in_temp,solution])
        print(in_temp)
    if glo.type == 'tdcs':
        in_dim = len(in_max)     #输入参数维度
        print(in_dim)
        print(glo.prior)
        solution = np.zeros(75)
        solution[glo.prior[0]] = 1
        solution[glo.prior[1]] = 1
        solution[glo.prior[2]] = -1
        solution[glo.prior[3]] = -1
        print(solution)
        solution=np.repeat([solution],5, axis=0)
        in_temp = np.random.uniform(-5, 5, (particals-5, in_dim))
        in_temp[-1 > in_temp] = 0
        in_temp[in_temp > 1] = 0
        in_temp = np.vstack([in_temp,solution])
        print(in_temp)
    return in_temp


def init_v(particals,v_max,v_min):
    v_dim = len(v_max)    
    v_ = np.random.uniform(0,1,(particals,v_dim))*(v_max-v_min)+v_min
    return v_

def init_pbest(in_,fitness_):
    return in_,fitness_

def init_archive(in_,fitness_):

    pareto_c = pareto.Pareto_(in_,fitness_)
    curr_archiving_in,curr_archiving_fit = pareto_c.pareto()
    return curr_archiving_in,curr_archiving_fit


