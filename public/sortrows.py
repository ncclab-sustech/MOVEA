import numpy as np

def sortrows(Matrix, order = "ascend"):
    Matrix_temp = Matrix[:, ::-1] 
    Matrix_row = Matrix_temp.T
    if order == "ascend":
        rank = np.lexsort(Matrix_row)
    elif order == "descend":
        rank = np.lexsort(-Matrix_row)
    Sorted_Matrix = Matrix[rank,:] 
    return Sorted_Matrix, rank
