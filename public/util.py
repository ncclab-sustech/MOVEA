import numpy as np
import math
from public import glo

inf = 9999999999
NUM_ELE = glo.NUM_ELE

POINT_NUM = 1000




if glo.head_model == 'hcp4':
    print("loading hcp4")
    # grey and white matter
    lfm = np.load(r'./data/lfm_hcp4.npy')
    print("load grey and white matter",lfm.shape)
    pos = np.load(r'./data/pos_hcp4.npy')
    print("load position")
    print(pos.shape)


#for avoidance 
# position =[-39, 34, 37] # dlpfc
# distance = np.zeros(len(pos))
# print(position)

# for i in range(len(pos)):
#     distance[i] = (pos[i, 0] - position[0])**2 + (pos[i, 1] - position[1])**2 + (pos[i, 2] - position[2])**2
# AVOID_POSITION = np.where(distance < 10**2)
# AVOID_POSITION = AVOID_POSITION[0]
# print(len(AVOID_POSITION))

position =  glo.position 
distance = np.zeros(len(pos))
print(position)

for i in range(len(pos)):
    distance[i] = (pos[i, 0] - position[0])**2 + (pos[i, 1] - position[1])**2 + (pos[i, 2] - position[2])**2

print('min_distance:' + str(min(distance)))
TARGET_POSITION = np.where(distance < 10**2) #roi size (CM^2)
TARGET_POSITION = TARGET_POSITION[0]
print('volume in roi:' + str(len(TARGET_POSITION)))

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


def magnitude_modulation(ea, eb, size=5):
    max_em_brain = np.zeros(ea.shape[0])
    dot_product = np.einsum('ij,ij->i', ea, eb)
    norm_a = np.linalg.norm(ea, axis=1)
    norm_b = np.linalg.norm(eb, axis=1)
    phi_rad = np.arccos(dot_product / (norm_a * norm_b))
    phi = np.degrees(phi_rad)
    for alpha in range(0, 360, size):
        em = envelop2(np.nan_to_num(norm_a * np.abs(np.cos(np.deg2rad(alpha))), nan=0.),
                      np.nan_to_num(norm_b * np.abs(np.cos(np.deg2rad(alpha - phi))), nan=0.))
        max_em_brain = np.where(max_em_brain < em, em, max_em_brain)
    return max_em_brain


# max intensity for stage one
def tis_function5(x1, x2, x3, x4,x5):

    electrode1 = x1
    electrode2 = x2
    stimulation1 = np.zeros(NUM_ELE)
    stimulation1[electrode1] = 1 + x5/NUM_ELE
    stimulation1[electrode2] = -1 - x5/NUM_ELE

    electrode3 = x3
    electrode4 = x4
    stimulation2 = np.zeros(NUM_ELE)

    stimulation2[electrode3] = 1 - x5/NUM_ELE
    stimulation2[electrode4] = -1 + x5/NUM_ELE
    ex1 = np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1)
    ex2 = np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2)
    ey1 = np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1)
    ey2 = np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2)
    ez1 = np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)
    ez2 = np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)
    ea = np.array([ex1, ey1, ez1]).T / 1000
    eb = np.array([ex2, ey2, ez2]).T / 1000
    max_em_brain = magnitude_modulation(ea, eb)
    return 1 / np.average(max_em_brain)


# safe constaints for tis
def tis_constraint6(x):
    lst = [math.ceil(x[i] * NUM_ELE) for i in [2, 3, 4, 5]]
    set_lst = set(lst)
    if len(set_lst) == len(lst) and x[0] + x[1] <= 1 and x[0] > 0.2 and x[1] > 0.2 and x[0] < 0.75 and x[1] < 0.75:
        return True
    else:
        return False

def envelop2(e1, e2):
    # print(e1.shape)
    mag_e1 = np.abs(e1)
    mag_e2 = np.abs(e2)
    index = np.where(mag_e1<=mag_e2)

    return 2 * np.where(mag_e1 <= mag_e2, e1,e2)
    
def tis_function6(x):

    electrode1 = int(round(x[2] * (NUM_ELE-1)))
    electrode2 = int(round(x[3] * (NUM_ELE-1)))
    stimulation1 = np.zeros(NUM_ELE)
    stimulation1[electrode1] = 2 * x[0] 
    stimulation1[electrode2] = -(2 * x[0])
    
    electrode3 = int(round(x[4] * (NUM_ELE-1)))
    electrode4 = int(round(x[5] * (NUM_ELE-1)))
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
    max_em_brain = magnitude_modulation(ea, eb)
    return np.array([1 / np.average(max_em_brain[TARGET_POSITION]), np.mean(max_em_brain)])

    
def tis_function6_x(x):

    electrode1 = int(round(x[2] * (NUM_ELE-1)))
    electrode2 = int(round(x[3] * (NUM_ELE-1)))
    stimulation1 = np.zeros(NUM_ELE)
    stimulation1[electrode1] = 2 * x[0]
    stimulation1[electrode2] = -(2 * x[0])
    e1 = np.array([np.matmul(lfm[:, :, 0].T, stimulation1), np.matmul(lfm[:, :, 1].T, stimulation1),np.matmul(lfm[:, :, 2].T, stimulation1)]).T /1000

    electrode3 = int(round(x[4] * (NUM_ELE-1)))
    electrode4 = int(round(x[5] * (NUM_ELE-1)))
    stimulation2 = np.zeros(NUM_ELE)

    stimulation2[electrode3] = 2 * x[1]
    stimulation2[electrode4] = -(2 * x[1])
    e2 = np.array([np.matmul(lfm[:, :, 0].T, stimulation2), np.matmul(lfm[:, :, 1].T, stimulation2),np.matmul(lfm[:, :, 2].T, stimulation2)]).T / 1000
    eam = envelop(e1,e2)

    return np.array([1 / np.average(np.abs(eam[TARGET_POSITION])), np.mean(eam)])  


# for avoidance
def tis_function6_t(x):

    electrode1 = int(round(x[2] * (NUM_ELE-1)))
    electrode2 = int(round(x[3] * (NUM_ELE-1)))
    stimulation1 = np.zeros(NUM_ELE)
    stimulation1[electrode1] = 2 * x[0]
    stimulation1[electrode2] = -(2 * x[0])
    e1 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T /1000
    electrode3 = int(round(x[4] * (NUM_ELE-1)))
    electrode4 = int(round(x[5] * (NUM_ELE-1)))
    stimulation2 = np.zeros(NUM_ELE)
    stimulation2[electrode3] = 2 * x[1]
    stimulation2[electrode4] = -(2 * x[1])
    e2 = np.array([np.matmul(lfm[:,TARGET_POSITION, 0].T, stimulation2), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000
    target = envelop(e1,e2)
     

    electrode1 = int(round(x[2] * (NUM_ELE-1)))
    electrode2 = int(round(x[3] * (NUM_ELE-1)))
    stimulation1 = np.zeros(NUM_ELE)
    stimulation1[electrode1] = 2 * x[0]
    stimulation1[electrode2] = -(2 * x[0])
    e1 = np.array([np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1), np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1),np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)]).T /1000

    electrode3 = int(round(x[4] * (NUM_ELE-1)))
    electrode4 = int(round(x[5] * (NUM_ELE-1)))
 
    stimulation2 = np.zeros(NUM_ELE)

    stimulation2[electrode3] = 2 * x[1]
    stimulation2[electrode4] = -(2 * x[1])
    e2 = np.array([np.matmul(lfm[:,AVOID_POSITION, 0].T, stimulation2), np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation2),np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation2)]).T / 1000
    eam = envelop(e1,e2)

    return np.array([1 / np.average(np.abs(target)), np.average(np.abs(eam))])   


def mti(x):
        x = x/2
        x[np.where(abs(x)<0.01)] = 0
        e1 = np.array([np.matmul(lfm[:, :, 0].T, x[:NUM_ELE]), np.matmul(lfm[:, :, 1].T, x[:NUM_ELE]),np.matmul(lfm[:, :, 2].T, x[:NUM_ELE])]).T /1000
        e2 = np.array([np.matmul(lfm[:, :, 0].T, x[NUM_ELE:]), np.matmul(lfm[:, :, 1].T, x[NUM_ELE:]),np.matmul(lfm[:, :, 2].T, x[NUM_ELE:])]).T /1000
        eam = envelop(e1,e2)
        return np.array([1/np.mean(eam[TARGET_POSITION]),np.mean(eam)])

def mti_avoid(x):
        x = x/2
        x[np.where(abs(x)<0.01)] = 0
        e1 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, x[:NUM_ELE]), np.matmul(lfm[:, TARGET_POSITION, 1].T, x[:NUM_ELE]),np.matmul(lfm[:, TARGET_POSITION, 2].T, x[:NUM_ELE])]).T /1000
        e2 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, x[NUM_ELE:]), np.matmul(lfm[:, TARGET_POSITION, 1].T, x[NUM_ELE:]),np.matmul(lfm[:, TARGET_POSITION, 2].T, x[NUM_ELE:])]).T /1000
        field1 = envelop(e1,e2)
        e1 = np.array([np.matmul(lfm[:, AVOID_POSITION, 0].T, x[:NUM_ELE]), np.matmul(lfm[:, AVOID_POSITION, 1].T, x[:NUM_ELE]),np.matmul(lfm[:, AVOID_POSITION, 2].T, x[:NUM_ELE])]).T /1000
        e2 = np.array([np.matmul(lfm[:, AVOID_POSITION, 0].T, x[NUM_ELE:]), np.matmul(lfm[:, AVOID_POSITION, 1].T, x[NUM_ELE:]),np.matmul(lfm[:, AVOID_POSITION, 2].T, x[NUM_ELE:])]).T /1000
        field2 = envelop(e1,e2)
        return np.array([1/np.mean(field1),np.mean(field2)])


# tdcs 

# def function1_s(x):
#     x = np.array(x)
#     x[abs(x) < 0.01] = 0
#     return np.abs(1000/(np.matmul(lfm[:, TARGET_POSITION, 0].T, x))[0])

def tdcs_function(s):
    s[abs(s[:]) < 0.01] = 0
    eam = ((np.matmul(lfm[:, :, 0].T, s)) ** 2 + (np.matmul(lfm[:, :, 1].T, s)) ** 2 + (
        np.matmul(lfm[:, :, 2].T, s)) ** 2) ** 0.5 /1000
    return np.array([1 / np.average(np.abs(eam[TARGET_POSITION])), np.mean(eam)])  

def tdcs_function1(x):
    x = np.array(x)
    x[abs(x[:]) < 0.01] = 0
    return 1000 / np.average(((np.matmul(lfm[:, TARGET_POSITION, 0].T, x)) ** 2 + (
        np.matmul(lfm[:, TARGET_POSITION, 1].T, x)) ** 2 + (np.matmul(lfm[:, TARGET_POSITION, 2].T, x)) ** 2) ** 0.5)

# constaints for tdcs
def function3(x):
    x = np.array(x)
    x[abs(x) < 0.01] = 0

    result = abs(np.sum(x))
    if result <= 1:
        return 0
    else:
        return result - 1

def function4(x):
    x = np.array(x)
    x[abs(x) < 0.01] = 0
    result = np.sum(abs(x)) + abs(np.sum(x))
    if result <= 4:
        return 0
    else:
        return result - 4

    
def h_tdcs(x):
    #print("constraint violation:", function3(x) + 1/4 * function4(x))
    return function3(x) + 1 / 4 * function4(x)
 
def h_mti(x):
    x = x/2
    x = np.array(x)
    x[abs(x) < 0.01] = 0
    result1 = np.sum(abs(x[:NUM_ELE])) + abs(np.sum(x[:NUM_ELE]))  # without reference
    result2 = np.sum(abs(x[NUM_ELE:])) + abs(np.sum(x[NUM_ELE:]))
    if result1 <= 2 and result2 <= 2:
        return 0
    else:
        return np.maximum(result1-2,0)+np.maximum(result2-2,0)


def function0(s):
    return np.sum(np.abs(s) > 0.01)

#r0.5 matrices
# def function2roastt(s):
#     field = ((np.matmul(lfm[:, 0, :], s)) ** 2 + (np.matmul(lfm[:, 1, :], s)) ** 2 + (
#         np.matmul(lfm[:, 2, :], s)) ** 2) ** 0.5
#     r = 0
#     field_target = function1(s)
#     for f in field:
#         if f > 0.5 * field_target:
#             r = r + 1
#     return r


