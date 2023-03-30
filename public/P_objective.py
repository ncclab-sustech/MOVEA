import numpy as np
from public import glo
from public.util import tis_constraint6,tis_function6,mti,h_tdcs,mti_avoid,h_mti,tdcs_function
import multiprocessing



def fun_(x):
    if glo.type == 'ti':
        if tis_constraint6(x):  
            result = tis_function6(x) 
            if result[0] > 10:
                result = result * result[0]
        else:
            result = [10000, 10000]
    if glo.type == 'mti':
        cv_value = h_mti(x)
        if cv_value > 0.01:
            result = [100000 * cv_value, 100000 * cv_value]
        else:
            intensity = mti(x)
            #print(intensity)
            if intensity[0] > 10:
                result = intensity * intensity[0]
            else:
                result = intensity
    if glo.type == 'tdcs':
        cv_value = h_tdcs(x)

        if cv_value > 0.01:
            result = [100000 * cv_value, 100000 * cv_value]
        else:
            result = tdcs_function(x)
            if result[0] > 10:
                result = result * result[0]
    return result

def P_objective(Operation,Problem,M,Input,epoch=0):
    [Output, Boundary, Coding] = TES(Operation, Problem, M, Input,epoch)
    if Boundary == []:
        return Output
    else:
        return Output, Boundary, Coding

def TES(Operation,Problem,M,Input,epoch):
    if glo.type == 'ti':
        Boundary = []
        Coding = ""
        if Operation == "init":
            MaxValue = np.ones((1, 6))
            MinValue = np.zeros((1, 6))
            Population = np.random.uniform(0, 1, size=(Input, 6))
            Boundary = np.vstack((MaxValue, MinValue))
            Coding = "Real"
            return Population, Boundary, Coding
        elif Operation == "value":
            Population = Input
            FunctionValue = np.zeros((Population.shape[0], M))
            if Problem == "TEScv2":
                p = multiprocessing.Pool(20)
                FunctionValue = np.array(p.map(fun_, Population))
                p.close()
                p.join()
    if glo.type == 'mti':
        Boundary = []
        Coding = ""
        if Operation == "init":
            MaxValue = np.ones((1, 150))
            MinValue = -np.ones((1, 150))
            Population = np.random.uniform(-1, 1, size=(Input, 150))
            Boundary = np.vstack((MaxValue, MinValue))
            Coding = "Real"
            return Population, Boundary, Coding
        elif Operation == "value":
            Population = Input
            FunctionValue = np.zeros((Population.shape[0], M))
            if Problem == "TEScv2":
                p = multiprocessing.Pool(100)
                FunctionValue = np.array(p.map(fun_, Population))
                p.close()
                p.join()
    if glo.type == 'tdcs':      
        Boundary = []
        Coding = ""
        if Operation == "init":
            MaxValue = np.ones((1, 75))
            MinValue = -np.ones((1, 75))
            Population = np.random.uniform(-1, 1, size=(Input, 75))
            Boundary = np.vstack((MaxValue, MinValue))
            Coding = "Real"
            return Population, Boundary, Coding
        elif Operation == "value":
            Population = Input
            FunctionValue = np.zeros((Population.shape[0], M))
            if Problem == "TEScv2":
                p = multiprocessing.Pool(1)
                FunctionValue = np.array(p.map(fun_, Population))
                p.close()
                p.join()
  
    return FunctionValue, Boundary, Coding
