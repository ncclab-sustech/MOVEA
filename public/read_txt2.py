import numpy as np
import matplotlib.pyplot as plt

gt_data = []
f2 = open(r"/home/ncclab306/zfs-pool/wangmo/tes/mopso/img_txt/pareto_in.txt", "r")
lines = f2.readlines()
for line3 in lines:
	print(line3)
	cur = line3.strip().split(" ")
	cur = list(map(float, cur))
	gt_data.append(cur)

gt_data = np.array(gt_data)
#print(gt_data)


from util import *
#print([function1_s(i) for i in gt_data])
#print([function2_s(i) for i in gt_data])
print([h(i) for i in gt_data])

fig = plt.figure('第' + str(i + 1) + '次迭代')
ax3 = fig.add_subplot(111)#133
# ax3.set_xlim((0,1))
# ax3.set_ylim((0,1))
ax3.set_xlabel('intensity:1/E')
ax3.set_ylabel('focality:R')
ax3.scatter([1/function1_s(i) for i in gt_data], [function2_s(i) for i in gt_data], s=10, c='blue', marker=".")
plt.show()

# e_ori , E_all , R_ori, R_all, R_roast
data = np.array([[0.13, 0.137, 38.5, 42.8, 40.6],
		[0.14, 0.147, 39.2, 43.9, 41.5],
		[0.15, 0.157, 40.0, 45.2, 42.7],
		[0.16, 0.167, 40.8, 47.2, 44.4],
		[0.17, 0.178, 42.0, 49.3, 46.4],
		[0.18, 0.187, 43.0, 51.0, 48.2],
		[0.19, 0.197, 44.2, 52.6, 49.0],
		[0.20, 0.206, 45.7, 54.6, 50.7],
		[0.21, 0.216, 47.9, 55.5, 51.6],
		[0.22, 0.227, 51.1, 56.2, 51.7],
		[0.23, 0.235, 55.8, 58.8, 55.6],
		[0.24, 0.250, 60.5, 59.8, 56.7],
		[0.25, 0.264, 65.9, 65.3, 60.6],
		[0.26, 0.283, 68.4, 68.3, 60.2]])


plt.figure(figsize=(10, 10), dpi=100)
plt.scatter(data[:,0], data[:,2],c='r',label='single_direction')
plt.scatter(data[:,1], data[:,3],c='b',label='all_direction')
plt.scatter(data[:,0], data[:,4],c='y',label='roast_method')
plt.scatter([1/function1_s(i) for i in gt_data], [function2_s(i) for i in gt_data], s=10, c='black', marker=".")
plt.legend(loc='best', fontsize=16, markerscale=0.5)
plt.title("trade-off_for_electric_and_focality", fontdict={'size': 20})
plt.xlabel("Electric_Field(V/m)", fontdict={'size': 16})
plt.ylabel("Half-Max_Radius(mm)", fontdict={'size': 16})
plt.show()

plt.savefig("result.png")