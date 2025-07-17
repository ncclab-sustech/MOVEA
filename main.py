import geatpy as ea  
import sys
import os
import argparse
from public import glo
import numpy as np

NUM_ELE = glo.NUM_ELE

def argdet():
 if len(sys.argv) <= 9:
     args = myargs()
     return args
 else:
     print('Cannot recognize the inputs!')
     print("-i data -opt optimizer -dim dimension")
     exit()

def myargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', '-t', default="ti", help='stimulation method')
    parser.add_argument('--position', '-p', default='hippo', help='target location')
    parser.add_argument('--head', '-m', default='ernie', help='head model name')
    parser.add_argument('--gen', '-g', default= 0 , help='max epochs')
    parser.add_argument('--gen2', '-g2', default= 100 , help='stage2 max epochs')
    parser.add_argument('--m2m', '-f', default="", help='m2m file path')
    #parser.add_argument('--input', '-o', default= os.path.abspath(os.path.dirname(__file__))+'/data' , help='input path')
    #parser.add_argument('--output', '-o', default= os.path.abspath(os.path.dirname(__file__)) , help='output path')
    args = parser.parse_args()
    parser.add_argument('--name', '-n', default= args.type + "_" + args.position + "_" + args.head , help='output name')
    args = parser.parse_args()
    return args

print("start")

args = argdet()
glo.head_model = args.head
glo.type = args.type

glo.name = args.position
if args.position == 'hippo':
    glo.position = np.array([-31, -20, -14])
elif args.position == 'pallidum':
    glo.position = np.array([-17, 3, -1])
elif args.position == 'thalamus':
    glo.position = np.array([10, -19, 6])
elif args.position == 'sensory':
    glo.position = [41,-36,66]
elif args.position == 'dorsal':
    glo.position = [25,42,37]
elif args.position == 'v1':
    glo.position = np.array([10,-92,2])
elif args.position == 'dlpfc':
    glo.position = np.array([-39, 34, 37])
elif args.position == 'motor':
    glo.position = np.array([47, -13, 52])
else:
    print("coordinate")
    glo.position = np.array(args.position)



if args.type == 'ti':
    from ti_problem import MyProblem  
    problem = MyProblem()
elif args.type == 'mti':
    from tdcs_problem import MyProblem  
    problem = MyProblem()
elif args.type == 'tdcs':
    from tdcs_problem import MyProblem  
    problem = MyProblem()
else:
    print('ERROR: STIMULATION TYPE')
    sys.exit(1)
gen = 50
if int(args.gen) != 0:
    gen = int(args.gen)


algorithm = ea.soea_SEGA_templet(
    problem,
    ea.Population(Encoding='RI', NIND=30),
    MAXGEN=gen,  # iteration
    logTras=1,  # print log per logTras epoch ，0 means not。
    #trappedValue=1e-2,  # early stopping parameter
    maxTrappedCount=10)  
algorithm.mutOper.F = 0.5  
algorithm.recOper.XOVR = 0.2  

res = ea.optimize(algorithm,
                  verbose=True,
                  drawing=10,
                  outputMsg=False,
                  drawLog=False,
                  saveFlag=False)
print(res)

prior = np.array(res['Vars'][0])
glo.prior = prior


from Mopso import *
from public import P_objective

particals = 100  # size of population 
cycle_ = args.gen2  # iteration2
mesh_div = 10  # grid parameter
thresh = 100  # size of archive

Problem = "TES"
M = 2 # number of obejctive

print("init")
_, Boundary, _ = P_objective.P_objective("init", Problem, M, particals)
max_ = Boundary[0]
min_ = Boundary[1]

print("start")
mopso_ = Mopso(particals, max_, min_, thresh, mesh_div)  
pareto_in, pareto_fitness = mopso_.done(cycle_)
path_fitness = "./pareto_fitness_" + args.name + ".txt"
path_in = "./pareto_in_" + args.name + ".txt"

if args.type == 'ti':
    fp = open(path_in,'w+')
    for solution in (pareto_in):
        result = ' '.join([str(elem) for elem in [int(round(solution[2] * (NUM_ELE-1))),2 * solution[0],int(round(solution[3] * (NUM_ELE-1))),2 * solution[0],int(round(solution[4] * (NUM_ELE-1))),-2 * solution[1],int(round(solution[5] * (NUM_ELE-1))),-2 * solution[1]]]) + '\n'
        fp.write(result)
    fp.close()
else:
    np.savetxt(path_in, pareto_in)
np.savetxt(path_fitness, pareto_fitness)
print("\n", "pareto_position:" + path_in)
print("pareto_value:" + path_fitness)
print("\n,over")

if args.m2m:
    glo.m2m = args.m2m
    from visualization import visual
    with open(path_in, 'r') as file:
        for i, line in enumerate(file):
            arr = np.array(line.strip().split(' '))
            visual(arr, i, args.type)
