import numpy as np


def MSE(f_matrix, fs_matrix):
    m = len(f_matrix)
    n = len(f_matrix[0])
    res = 0
    for i in range(m):
        for j in range(n):
           res += (f_matrix[i][j] - fs_matrix[i][j]) ** 2
    return res / (m*n)

def SNR(f_matrix, fs_matrix):
    m = len(f_matrix)
    n = len(f_matrix[0])
    res1 = 0
    res2 = 0
    for i in range(m):
        for j in range(n):
           res1 +=  fs_matrix[i][j] ** 2
           res2 += (f_matrix[i][j] - fs_matrix[i][j]) ** 2
    return res1 / res2

if __name__ == '__main__':
    f = [[1,2], [2,3]]
    fs = [[2,2], [4,6]]
    print(np.mean((np.array(f)-np.array(fs))**2))
    print(np.sum(np.array(fs)**2)/np.sum((np.array(f)-np.array(fs))**2))

    print(MSE(f, fs))
    print(SNR(f, fs))