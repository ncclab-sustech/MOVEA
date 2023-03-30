import h5py
import numpy as np
import math
import simnibs
#import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.io as io
from nibabel.affines import apply_affine



# grey matter and mapping to surface
# lfm = np.load(r'/home/ncclab306/database7/wangm/eeg10/leadfield.npy')
# pos = np.load(r'/home/ncclab306/database7/wangm/eeg10/pos.npy')
# print("load grey matter")

# grey and white matter
#lfm = np.load(r'/media/ncclab/database7/wangm/simnibs_leadfield/lfm_gw.npy')

lfm = np.load(r'/mnt/database7/wangm/simnibs_leadfield/lfm_hcp4_2.npy')
#lfm = lfm[[0,2,11,15]]
#lfm = h4['mesh_leadfield']['leadfields']['tdcs_leadfield']
print("load grey and white matter",lfm.shape)

mask = nib.load('/mnt/database7/wangm/T1/m2m_hcp4/hcp4_masks_contr.nii.gz')
mni =  nib.load('/mnt/database7/wangm/T1/m2m_hcp4/toMNI/T1fs_nu_nonlin_MNI.nii.gz')
pos = np.load(r'/mnt/database7/wangm/simnibs_leadfield/pos_hcp4_2.npy')

def envelop(e1,e2):
    eam = np.zeros(len(e1))
    l_x = np.sqrt(np.sum(e1 * e1, axis=1))
    l_y = np.sqrt(np.sum(e2 * e2, axis=1))
    point = np.sum(e1 * e2, axis=1)
    cos_ = point / (l_x * l_y)

    mask = cos_ <= 0
    e1[mask] = -e1[mask]
    cos_[mask] = -cos_[mask]

    equal_vectors = np.all(e1 == e2, axis=1)
    
    eam[equal_vectors] = 2 * l_x[equal_vectors]
    not_equal_vectors = ~equal_vectors
    mask2 = not_equal_vectors & (l_y < l_x)
    mask3 = not_equal_vectors & (l_x < l_y * cos_)
    eam[mask2 & mask3] = 2 * l_y[mask2 & mask3]
    eam[mask2 & ~mask3] = 2 * np.linalg.norm(np.cross(e2[mask2 & ~mask3], (e1[mask2 & ~mask3] - e2[mask2 & ~mask3])), axis=1) / np.linalg.norm(e1[mask2 & ~mask3] - e2[mask2 & ~mask3], axis=1)

    mask4 = not_equal_vectors & (l_y < l_x * cos_)
    mask5 = not_equal_vectors & (l_x < l_y)
    eam[mask5 & mask4] = 2 * l_x[~mask2 & mask4]
    eam[mask5 & ~mask4] = 2 * np.linalg.norm(np.cross(e1[mask5 & ~mask4], (e2[mask5 & ~mask4] - e1[mask5 & ~mask4])), axis=1) / np.linalg.norm(e2[mask5 & ~mask4] - e1[mask5 & ~mask4], axis=1)
   
    return eam

def tis_function6(x):

    electrode1 = int(round(x[2] * 74))
    electrode2 = int(round(x[3] * 74))
    stimulation1 = np.zeros(75)
    stimulation1[electrode1] = 2 * x[0]
    stimulation1[electrode2] = -(2 * x[0])
    e1 = np.array([np.matmul(lfm[:, :, 0].T, stimulation1), np.matmul(lfm[:, :, 1].T, stimulation1),np.matmul(lfm[:, :, 2].T, stimulation1)]).T /1000

    electrode3 = int(round(x[4] * 74))
    electrode4 = int(round(x[5] * 74))
    stimulation2 = np.zeros(75)

    stimulation2[electrode3] = 2 * x[1]
    stimulation2[electrode4] = -(2 * x[1])
    e2 = np.array([np.matmul(lfm[:, :, 0].T, stimulation2), np.matmul(lfm[:, :, 1].T, stimulation2),np.matmul(lfm[:, :, 2].T, stimulation2)]).T / 1000

    return envelop(e1,e2)

def mti(x):
        x = x/2
        x[np.where(abs(x)<0.01)] = 0
        e1 = np.array([np.matmul(lfm[:, :, 0].T, x[:75]), np.matmul(lfm[:, :, 1].T, x[:75]),np.matmul(lfm[:, :, 2].T, x[:75])]).T /1000
        e2 = np.array([np.matmul(lfm[:, :, 0].T, x[75:]), np.matmul(lfm[:, :, 1].T, x[75:]),np.matmul(lfm[:, :, 2].T, x[75:])]).T /1000
        return envelop(e1,e2)



def tdcs(s):
    field = ((np.matmul(lfm[:, :, 0].T, s)) ** 2 + (np.matmul(lfm[:, :, 1].T, s)) ** 2 + (
        np.matmul(lfm[:, :, 2].T, s)) ** 2) ** 0.5
    return abs(field)/1000

# import numpy as np
# gt_data = []
# f2 = open(r"tdcs_in.txt", "r")
# lines = f2.readlines()
# for line3 in lines:
#     #print(line3)
#     cur = line3.strip().split(" ")
#     cur = list(map(float, cur))
#     gt_data.append(cur)
# data = np.array(gt_data)
# #index1 = np.argsort(1/data[:,0])
# value= data[1]
# print(value)
# print(value[np.where(abs(value)>0.02)])
# print(np.where(abs(value)>0.02))


# input = np.zeros(len(x))
# for value in x[np.where(abs(x)>0.02)]:

# mask = nib.load('/media/ncclab/database7/wangm/T1/m2m_hcp4/hcp4_final_contr.nii.gz')
ob = mask.get_fdata()
# gm_mni =  nib.load('/media/ncclab/database7/wangm/T1/m2m_hcp4/toMNI/gm_MNI.nii.gz')
# obm = gm_mni.get_fdata()
# wm_mni =  nib.load('/media/ncclab/database7/wangm/T1/m2m_hcp4/toMNI/wm_MNI.nii.gz')
# obw = wm_mni.get_fdata()

x = tis_function6([0.49678407, 0.49939584, 1.      ,   0.04649278, 0.97325818, 0.16342326])

print("have saved eam")

pos_hcp4 = apply_affine(np.linalg.inv(mni.affine), pos)  # pos coor should be same with model. here should be ernie

# loc2 = apply_affine(np.linalg.inv(mask.affine), loc2)

con = np.zeros([260, 311, 260])

for idx, i in enumerate(pos_hcp4):
    con[int(i[0]), int(i[1]), int(i[2])] = x[idx]

    
point = np.array([np.where(con > 0)[0], np.where(con > 0)[1], np.where(con > 0)[2]]).T
print(point)
##some values in eam were overlapped
v = np.zeros(len(point))
for i in range(len(point)):
   v[i] = con[point[i, 0], point[i, 1], point[i, 2]]
point_grid = np.array([np.where(((ob == 2) | (ob == 1)))[0], np.where(((ob == 2) | (ob == 1)))[1],
                      np.where(((ob == 2) | (ob == 1)))[2]]).T  # mask < 4, gm == 1
print(point_grid)

#point_grid = np.array(apply_affine(np.linalg.inv(mask.affine), point_grid) )  ##

from scipy.interpolate import griddata
value = griddata(point, v, point_grid, method='linear')
for i in range(len(point_grid)):
    ob[point_grid[i, 0], point_grid[i, 1], point_grid[i, 2]] = value[i]
# ob = np.where(ob>=2,0,ob)
# print(np.max(ob))

ob = np.where(ob>1,0,ob) #remove skull etc

img = nib.Nifti1Image(ob, mni.affine)
nib.save(img, 'tis322_hcp4.nii')  # _scatter
