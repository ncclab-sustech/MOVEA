import os

import h5py
import numpy as np
import math
import nibabel as nib
import numpy as np
import scipy.io as io
from matplotlib import pyplot as plt
from nibabel.affines import apply_affine
from nilearn import plotting, image

from public import glo
from public.util import magnitude_modulation

NUM_ELE = glo.NUM_ELE
lfm = np.load(r'data/lfm_hcp4.npy')
print("load grey and white matter", lfm.shape)
pos = np.load(r'data/pos_hcp4.npy')
mask = nib.load(glo.m2m)



def affine(l, b, i):  # from left to right, -5 to 5
    return np.dot(np.linalg.inv(mask.affine), np.append([l, b, i], 1))[:3]


def envelop(e1, e2):
    e1 = np.sqrt(np.sum(e1 * e1, axis=1))
    e2 = np.sqrt(np.sum(e2 * e2, axis=1))
    min_values = np.minimum(e1, e2)
    result = 2 * min_values

    return result


def envelop2(e1, e2):
    # print(e1.shape)
    mag_e1 = np.abs(e1)
    mag_e2 = np.abs(e2)
    # index = np.where(mag_e1<=mag_e2)

    return 2 * np.where(mag_e1 <= mag_e2, e1, e2)


def mti(x):
    x = x / 2
    x[np.where(abs(x) < 0.01)] = 0
    e1 = np.array([np.matmul(lfm[:, :, 0].T, x[:NUM_ELE]), np.matmul(lfm[:, :, 1].T, x[:NUM_ELE]),
                   np.matmul(lfm[:, :, 2].T, x[:NUM_ELE])]).T / 1000
    e2 = np.array([np.matmul(lfm[:, :, 0].T, x[NUM_ELE:]), np.matmul(lfm[:, :, 1].T, x[NUM_ELE:]),
                   np.matmul(lfm[:, :, 2].T, x[NUM_ELE:])]).T / 1000
    return envelop(e1, e2)


def tis_function6(x):
    electrode1 = int(x[2])
    electrode2 = int(x[3])
    stimulation1 = np.zeros(NUM_ELE)
    stimulation1[electrode1] = 2 * x[0]
    stimulation1[electrode2] = -(2 * x[0])

    electrode3 = int(x[4])
    electrode4 = int(x[5])
    stimulation2 = np.zeros(NUM_ELE)
    stimulation2[electrode3] = 2 * x[1]
    stimulation2[electrode4] = -(2 * x[1])

    ex1 = np.matmul(lfm[:, :, 0].T, stimulation1)
    ex2 = np.matmul(lfm[:, :, 0].T, stimulation2)
    ey1 = np.matmul(lfm[:, :, 1].T, stimulation1)
    ey2 = np.matmul(lfm[:, :, 1].T, stimulation2)
    ez1 = np.matmul(lfm[:, :, 2].T, stimulation1)
    ez2 = np.matmul(lfm[:, :, 2].T, stimulation2)
    ea = np.array([ex1, ey1, ez1]).T / 1000
    eb = np.array([ex2, ey2, ez2]).T / 1000
    return magnitude_modulation(ea, eb)


def tdcs_function(s):
    s[abs(s[:]) < 0.01] = 0
    eam = ((np.matmul(lfm[:, :, 0].T, s)) ** 2 + (np.matmul(lfm[:, :, 1].T, s)) ** 2 + (
        np.matmul(lfm[:, :, 2].T, s)) ** 2) ** 0.5 /1000
    return eam


def visual(arr, num, type):

    ob = mask.get_fdata()

    if type == "ti":
        ele = np.zeros(6)
        ele[0] = arr[1]
        ele[1] = arr[5]
        ele[2:] = arr[[0, 2, 4, 6]]
        print(ele)
        x = tis_function6(ele)
    elif type == "mti":
        x = mti(arr.astype(np.float32))
    elif type == "tdcs":
        x = tdcs_function(arr.astype(np.float32))
    print(x.shape)
    con = np.zeros(ob.shape)

    for idx, i in enumerate(pos):
        i = affine(i[0], i[1], i[2])
        con[int(i[0]), int(i[1]), int(i[2])] = x[idx]

    point = np.array([np.where(con > 0)[0], np.where(con > 0)[1], np.where(con > 0)[2]]).T
    # print(point)
    ##some values in eam were overlapped
    v = np.zeros(len(point))
    for i in range(len(point)):
        v[i] = con[point[i, 0], point[i, 1], point[i, 2]]
    point_grid = np.array([np.where(((ob == 2) | (ob == 1)))[0], np.where(((ob == 2) | (ob == 1)))[1],
                           np.where(((ob == 2) | (ob == 1)))[2]]).T

    print(point_grid.shape)
    print(len(np.where(x > 0)[0]))

    from scipy.interpolate import griddata

    value = griddata(point, v, point_grid, method='nearest')
    print(value)
    print('value1', value[0])
    print("finish interpolate")
    ob = np.zeros(ob.shape)
    for i in range(len(point_grid)):
        ob[point_grid[i, 0], point_grid[i, 1], point_grid[i, 2]] = value[i]

    print(ob[ob > 0].size)
    print(ob.shape)

    # ob = np.where(ob>1.5,0,ob)
    save_path = f'results/result_{num}.nii.gz'

    img = nib.Nifti1Image(ob, mask.affine)
    nib.save(img, save_path)

    if not os.path.exists(save_path):
        print('TI model does not exist. ')
    else:
        img = image.load_img(save_path)
        bg = image.load_img(mask)
        target_pos = glo.position.copy()
        plotting.plot_roi(img,
                          bg_img=bg,
                          cut_coords=target_pos,
                          black_bg=False,
                          cmap=plt.get_cmap('jet'),
                          output_file=f'results/result_{num}.png')
