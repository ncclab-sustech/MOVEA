import numpy as np
from public import glo,util
import geatpy as ea

NUM_ELE =glo.NUM_ELE
class MyProblem(ea.Problem):  
    def __init__(self):
        name = 'MyProblem'  
        M = 1  
        maxormins = [1] 
        self.var_set = np.arange(0,NUM_ELE,1) 
        Dim = 5  
        varTypes = [1] * Dim  
        lb = [0, 0, 0, 0, 0]  
        ub = [(NUM_ELE-1), (NUM_ELE-1), (NUM_ELE-1), (NUM_ELE-1), (NUM_ELE-1)]  
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
        x5 = self.var_set[Vars[:, [4]]]  
        r = np.zeros(len(x1)).tolist()
        for i in range(len(x1)):
            lst = [int(x1[i]),int(x2[i]),int(x3[i]),int(x4[i])]
            set_lst = set(lst)
            if len(set_lst) == len(lst):
                r[i] = [util.tis_function5(x1[i], x2[i], x3[i], x4[i],x5[i])]
            else:
                r[i] = [1000]
        return np.array(r)
