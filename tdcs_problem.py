import numpy as np
from public import glo
import geatpy as ea
from public import util


class MyProblem(ea.Problem): 

    def __init__(self):
        name = 'MyProblem'  
        M = 1  
        maxormins = [1]  
        self.var_set = np.arange(0,75,1) 
        Dim = 4  
        varTypes = [1] * Dim  
        lb = [0, 0,0,0]  
        ub = [74, 74,74,74]  
        lbin = [1] * Dim  
        ubin = [1] * Dim  

        ea.Problem.__init__(self,
                            name,
                            M,
                            maxormins,
                            Dim,
                            varTypes,
                            lb,
                            ub,
                            lbin,
                            ubin)

    def evalVars(self, Vars):  
        Vars = Vars.astype(np.int32) 
        x1 = self.var_set[Vars[:, [0]]]  
        x2 = self.var_set[Vars[:, [1]]]  
        x3 = self.var_set[Vars[:, [2]]]  
        x4 = self.var_set[Vars[:, [3]]] 
        r = np.zeros(len(x1)).tolist()
        for i in range(len(x1)):
            lst = [int(x1[i]),int(x2[i]),int(x3[i]),int(x4[i])]
            set_lst = set(lst)
            if len(set_lst) == len(lst):
                x = np.zeros(75)
                x[x1[i]] = 1
                x[x2[i]] = 1
                x[x3[i]] = -1
                x[x4[i]] = -1
                r[i] = [util.tdcs_function1(x)]
            else:
                r[i] = [10000]
        return np.array(r)
