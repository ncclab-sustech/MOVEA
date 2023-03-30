#encoding: utf-8
import numpy as np
from public import init,update,plot,P_objective,util

import time

class Mopso:
    def __init__(self,particals,max_,min_,thresh,mesh_div=10):

        self.i = 0
        self.mesh_div = mesh_div
        self.particals = particals
        self.thresh = thresh
        self.max_ = max_
        self.min_ = min_
        self.m = 2          
        self.max_v = 100 * np.ones(len(max_), )  
        self.min_v = -100 * np.ones(len(min_), )  

        self.plot_ = plot.Plot_pareto()

    def evaluation_fitness(self):
        self.fitness_ = P_objective.P_objective("value", "TEScv2", self.m, self.in_,self.i)

    def initialize(self):

        self.in_ = init.init_designparams(self.particals,self.min_,self.max_)

        self.v_ = init.init_v(self.particals,self.max_v,self.min_v)

        self.evaluation_fitness()

        self.in_p,self.fitness_p = init.init_pbest(self.in_,self.fitness_)

        self.archive_in,self.archive_fitness = init.init_archive(self.in_,self.fitness_)

        self.in_g,self.fitness_g = update.update_gbest_1(self.archive_in,self.archive_fitness,self.mesh_div,self.particals)
    def update_(self):


        self.v_ = update.update_v(self.v_,self.min_v,self.max_v,self.in_,self.in_p,self.in_g)
        self.in_ = update.update_in(self.in_,self.v_,self.min_,self.max_)

        self.evaluation_fitness()

        self.in_p,self.fitness_p = update.update_pbest(self.in_,self.fitness_,self.in_p,self.fitness_p)

        self.archive_in, self.archive_fitness = update.update_archive_1(self.in_, self.fitness_, self.archive_in,
                                                                      self.archive_fitness,
                                                                      self.thresh, self.mesh_div)


        self.in_g,self.fitness_g = update.update_gbest_1(self.archive_in,self.archive_fitness,self.mesh_div,self.particals)

    def done(self,cycle_):
        self.initialize()
        self.plot_.show(self.in_,self.fitness_,self.archive_in,self.archive_fitness,-1)
        since = time.time()
        for i in range(cycle_):
            self.i = i
            self.update_()

            print('Epoch',i,'time consuming: ',np.round(time.time() - since, 2), "s")
            print(self.archive_fitness)
            if self.i % 10 == 0 :
                self.in_ = init.init_designparams(self.particals, self.min_, self.max_)
                self.v_ = init.init_v(self.particals, self.max_v, self.min_v)
                self.evaluation_fitness()
                self.in_p, self.fitness_p = init.init_pbest(self.in_, self.fitness_)

                self.plot_.show(self.in_,self.fitness_,self.archive_in,self.archive_fitness, i, self.m)
        return self.archive_in,self.archive_fitness

