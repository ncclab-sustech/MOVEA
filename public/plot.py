#encoding: utf-8
import numpy as np
import os
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d import Axes3D
import time
class Plot_pareto:
    def __init__(self):
        self.start_time = time.time()

    def show(self,in_,fitness_,archive_in,archive_fitness,i,m=2):

        fig = plt.figure('Epoch' + str(i + 1) )

        plt.legend(loc='best', fontsize=16, markerscale=0.5)
        plt.title("trade-off_for_electric_and_focality", fontdict={'size': 20})
        plt.xlabel("TARGET1_Electric_Field(V/m)", fontdict={'size': 16})
        plt.ylabel("TARGET2_Electric_Field(V/m)", fontdict={'size': 16})
        plt.scatter(1/archive_fitness[:, 0], archive_fitness[:, 1], s=30, c='black', marker=".", alpha=1.0, label='MOPSO')
        plt.savefig('./pic/'+str(i)+'_result.png')

        # 3 objections
        # plt.figure('epoch' + str(i + 1) , figsize=(10, 10), dpi=100)
        # ax = plt.axes(projection='3d')  
        # ax.scatter3D(fitness_[:, 1], fitness_[:, 0], fitness_[:, 2], s=10, c='blue', marker=".")  
        # ax.scatter3D(archive_fitness[:, 1], archive_fitness[:, 0], archive_fitness[:, 2], s=30, c='red', marker=".", alpha=1.0)
        # # plt.xticks(range(11)) 
        # plt.rcParams.update({'font.size': 20})
        # plt.xlabel('intensity:1/E')
        # plt.ylabel('focality:R', rotation=38) 
        # ax.set_zlabel('constraint violation')  
        # plt.show()
        # plt.ion()

