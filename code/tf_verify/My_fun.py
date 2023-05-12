import math
import numpy as np
from config import config
import scipy.spatial as sp
# from scipy.spatial import qhull as qh
from numpy import matlib
import subprocess32 as subprocess
from math import *
import scipy
import multiprocessing
import sys
from numpy.linalg import matrix_rank
from scipy import linalg
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from enum import Enum
# a = np.array([1,3])
# a = np.max(np.abs(a) +a, axis=-1)
# a = a

def uniqm(A, t=1e-13):
    """
    Return input matrix with duplicate entries removed.
     A - [matrix] input matrix (possibly containing duplicates)
     t - [float]  tolerance (default=1e-13)
    """
    Nrows = A.shape[0]
    uniquerows = [r1 for r1 in range(Nrows)
                  if not any(np.all(abs(A[r1, :] - A[r2, :]) < t)
                             for r2 in range(r1 + 1, Nrows))]
    return A[uniquerows, :].copy()

def qhullstr(V):
    """
    generate string qhull input format.

    yields a newline separated string of format:
        dimensions (columns of V)
        number of points (rows of V)
        one string for each row of V
    """
    V = np.array(V)
    return "%i\n%i\n" % (V.shape[1], V.shape[0]) \
           + "\n".join(" ".join(str(e) for e in row) for row in V)

def q_hull(V, qstring):
    """
    Use qhull to determine convex hull / volume / normals.
     V - [matrix] vertices
     qstring - [string] arguments to pass to qhull
    """
    # try:
    # qhullp = subprocess.Popen(["qhull", qstring],
    #                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    # Vc = qhullp.communicate(qhullstr(V))[0]  # qhull output to Vc
    Vc = qhullstr(V)
    if qstring == "FS":  # calc area and volume
        ks = Vc.split('\n')[-2]
        Vol = float(ks.split(' ')[-2])  # get volume of D-hull
        return Vol
    elif qstring == "Ft":  # calc vertices and facets
        ks = Vc.split('\n')
        # fms = int(ks[1].split(' ')[1])  # get size of facet matrix
        fms = int(ks[1])
        fmat = ks[2:fms+2]
        fmat = ';'.join(fmat)
        fmat = np.mat(fmat)  # generate matrix
        fmatv = fmat[:, 1:]  # vertices on facets
        return np.array(fmatv)
    elif qstring == "n":  # calc convex hull and get normals
        ks = ';'.join(Vc.split('\n')[2:])  # remove leading dimension output
        k = np.mat(ks[:-1])  # convert to martrix with vertices
        return np.array(k)
    else:
        exit(1)
    # except:
    #     raise NameError('QhullError')

def get_total_eposilon(epsilon_list):
    epsilon = 0.5
    epsilon_star = 0.0
    interval = 0.0
    l = len(epsilon_list)

    epsilon_max_1 = 0
    epsilon_max_2 = 0

    for i in range(1, l):
        x1_lb = np.array(epsilon_list[i][0])
        x1_ub = np.array(epsilon_list[i][1])
        for j in range(i+1, l):
            x2_lb = np.array(epsilon_list[j][0])
            x2_ub = np.array(epsilon_list[j][1])

            epsilon_max_1 = max(np.max(np.abs(x1_ub - x2_lb), axis=-1), epsilon_max_1)
            epsilon_max_2 = max(np.max(np.abs(x2_ub - x1_lb), axis=-1), epsilon_max_2)
    interval = max(epsilon_max_1, epsilon_max_2)
        # interval = max(epsilon_max_1, epsilon_max_2)
        # if interval > epsilon:
        #     epsilon -= pow(0.5, i+1)
        # else:
        #     epsilon += pow(0.5, i+1)
        #
        # if abs(epsilon - epsilon_star) <= 0.0001:
        #     return epsilon
        # else:
        #     epsilon_star = epsilon

    return interval

class BSTNode:
    """
    定义一个二叉树节点类。
    """
    def __init__(self, data, left=None, right=None):
        """
        初始化
        :param data: 节点储存的数据
        :param left: 节点左子树
        :param right: 节点右子树
        """
        self.data = data
        self.left = left
        self.right = right

class BinarySortTree:
    """
    基于BSTNode类的二叉排序树。维护一个根节点的指针。
    """
    def __init__(self):
        self._root = None

    def is_empty(self):
        return self._root is None

    def compare(self, key):
        """
        关键码compare
        :param key: 关键码
        :return: signal 1 or 0
        """
        bt = self._root
        while bt:
            entry = bt.data
            if key < entry:
                bt = bt.left
            else:
                return 0

        return 1

    def insert(self, key):
        """
        插入操作
        :param key:关键码
        :return: 布尔值
        """
        if self.is_empty():
            self._root = BSTNode(key)

        bt = self._root

        while True:
            entry = bt.data

            if key < entry:
                if bt.left is None:
                    bt.left = BSTNode(key)
                bt = bt.left
            elif key > entry:
                if bt.right is None:
                    bt.right = BSTNode(key)
                bt = bt.right
            else:
                bt.data = key
                return

    def delete(self, key):
        """
        二叉排序树最复杂的方法
        :param key: 关键码
        :return: 布尔值
        """
        p, q = None, self._root # 维持p为q的父节点，用于后面的链接操作
        if not q:
            print("空树！")
            return
        while q and q.data != key:
            p = q
            if key < q.data:
                q = q.left
            else:
                q = q.right
            if not q: # 当树中没有关键码key时，结束退出。
                return
        # 上面已将找到了要删除的节点，用q引用。而p则是q的父节点或者None（q为根节点时）。
        if not q.left:
            if p is None:
                self._root = q.right
            elif q is p.left:
                p.left = q.right
            else:
                p.right = q.right
            return
        # 查找节点q的左子树的最右节点，将q的右子树链接为该节点的右子树
        # 该方法可能会增大树的深度，效率并不算高。可以设计其它的方法。
        r = q.left
        while r.right:
            r = r.right
        r.right = q.right
        if p is None:
            self._root = q.left
        elif p.left is q:
            p.left = q.left
        else:
            p.right = q.left

def bSearch(array,element):
    low = 0
    high = len(array) - 1
    while low <= high:
        k = (low + high)//2
        if array[k] <= element:
            return 0
        else:
            high = k-1
    return 1

def EveryStrandIsN(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]

def con2vert(A, b):
    """
    Convert sets of constraints to a list of vertices (of the feasible region).
    If the shape is open, con2vert returns False for the closed property.
    """

    c = np.linalg.lstsq(np.mat(A), np.mat(b))[0]
    btmp = np.mat(b)-np.mat(A)*c
    # D = np.mat(A)/matlib.repmat(btmp, 1, A.shape[1])
    arrayA = np.mat(A)
    arrayB = matlib.repmat(btmp, 1, A.shape[1])
    D = np.divide(arrayA, arrayB, out=np.zeros_like(arrayA, dtype=np.float64), where=arrayB!=0)

    fmatv = q_hull(D, "Ft") #vertices on facets

    G = np.zeros((fmatv.shape[0], D.shape[1]))
    for ix in range(0, fmatv.shape[0]):

        F = D.squeeze()
        # F = D[fmatv[ix, :], :].squeeze()

        G[ix, :] = np.linalg.lstsq(F[ix], np.ones((F[ix].shape[0], 1)))[0].transpose()

    V = G + np.matlib.repmat(c.transpose(), G.shape[0], 1)
    # ux = uniqm(V)

    # eps = 1e-13
    # Av = np.dot(A, ux.T)
    # bv = np.tile(b, (1, ux.shape[0]))
    # closed = np.sciall(Av - bv <= eps)

    return V

def vert2con(V):
    """
    vert2con function maps vertices of a convex polygon
    to linear constraints stored in matrices A and b. And it works.
    """
    k = ConvexHull(V).vertices
    u = np.roll(k,-1)

    k = np.vstack((k,u)).T
    c = np.mean(V[k[:,0]], axis = 0)

    V = V - matlib.repmat(c, V.shape[0],1)
    # V = V - np.matlib.repmat(c, V.shape[0], 1)

    A = np.zeros((k.shape[0], V.shape[1]))
    A[:] = np.nan
    rc = 0

    for ix in range(k.shape[0]):
        F = V[k[ix]]
        if matrix_rank(F, 1e-5) == F.shape[0]:
            rc = rc+1
            # A[rc-1,:] = np.linalg.solve(F,np.ones((F.shape[0])))
            A[rc-1,:] = np.linalg.lstsq(F, np.ones((F.shape[0])), rcond=None)[0]
    A = A[0:rc,:]
    b = np.ones((A.shape[0], 1))
    b = b.T + A @ c.T

    return(A, b)

def sum_abs(a):
    sum = 0
    for i in a:
        sum += abs(i)
    return sum

def cal_tao(A, b):
    tao = float('inf')
    for i in range(len(A)):
        s = sum_abs(A[i,:])
        if s > 0:
            r = b[i, 0] / s
            if tao > r:
                tao = r
        else:
            tao = float('inf')

    return tao

# def InnerBox(A,b):
#     # calculate lamda for x0 if no multi-inner, ignore
#
#
#     # find a point x0 in convex_hull(A,b)
#     x0 = np.zeros((2 * config.k,  1))
#     # initialization
#     k = 0
#     A_k = A
#     b_k = b - A @ x0
#     l_k = np.zeros((1, 2 * config.k))
#     u_k = np.zeros((1, 2 * config.k))
#     L_k = []
#     U_k = []
#     D = list(range(2 * config.k))
#
#     while L_k != D or U_k != D:
#         # calculate tao for equations 13
#         tao = cal_tao(A_k, b_k)
#
#         # equations 13 for iteration
#
#         # deal with equation12 and aij bi in equation 13
#         L_r = []
#         U_r = []
#         for i in range(len(A)):
#
#             # deal with Li_k & Ui_k
#             Li_k = []
#             Ui_k = []
#             for j in range(2 * config.k):
#                 if A_k[i,j] < 0 and ~( (A_k[:, j] >= 0).all() ):
#                     Li_k.append(j)
#                 elif A_k[i,j] > 0 and ~( (A_k[:, j] <= 0 ).all() ):
#                     Ui_k.append(j)
#
#             Li_k = list(set(Li_k))
#             Ui_k = list(set(Ui_k))
#             R = Li_k.copy()
#             R.extend(Ui_k)
#             R = list(set(R))
#             # deal with bi_k+1
#             sum = 0
#             for j in range(2 * config.k):
#                 if j in R:
#                     sum += abs(A_k[i,j])
#             b_k[i,0] = b_k[i,0] - tao * sum
#
#             # deal with a(ij)_k+1
#             for j in range(2 * config.k):
#                 if j in R:
#                     A_k[i,j] = 0
#
#
#             if L_r == []:
#                 L_r = Li_k.copy()
#             else:
#                 L_r.extend(Li_k)
#                 L_r = list(set(L_r))
#             if U_r == []:
#                 U_r = Ui_k.copy()
#             else:
#                 U_k.extend(Ui_k)
#                 U_r = list(set(U_r))
#
#         # deal with l_k+1 & u_k+1 in equation 13 by L_r & U_r
#         for j in range(2 * config.k):
#             if j in L_r:
#                 l_k[0,j] = -tao
#             if j in U_r:
#                 u_k[0,j] = tao
#
#         # deal with L_k+1 and U_k+1
#         if L_k == []:
#             L_k = L_r.copy()
#         else:
#             L_k.extend(L_r)
#             L_k = list(set(L_k))
#         if U_k == []:
#             U_k = U_r.copy()
#         else:
#             U_k.extend(U_r)
#             U_k = list(set(U_k))
#
#
#
#     return V_lower

def pai(u, P, I, zero_d1, zero_1d):
    U = np.mat(np.diag((np.array(u).T)[0]))
    P_u = P * u

    result = np.column_stack( (np.row_stack(( P*U*P.T , P_u.T)), np.row_stack((P_u, np.mat([1])))) )

    return result
def pai_ni(u, P, I, zero_d1, zero_1d):
    # U = np.array(u).T
    U = np.mat(np.diag((np.array(u).T)[0]))
    P_u = P*u

    pai_ni_left = np.column_stack( (np.row_stack((I, -(P_u.T))), np.row_stack((zero_d1, np.mat([1])))) )
    pai_ni_middle = np.column_stack( (np.row_stack(( np.linalg.inv(P*U*P.T - P_u*P_u.T) , zero_1d)), np.row_stack((zero_d1, np.mat([1])))) )
    pai_ni_right = np.column_stack( (np.row_stack((I, zero_1d)), np.row_stack((-(P_u), np.mat([1])))) )
    result = pai_ni_left * pai_ni_middle * pai_ni_right

    return result

def fz(a):
    return a[::-1]

def FZ(mat):
    return np.array(fz(list(map(fz,mat))))

def conv_kernel2matrix(input_shape, out_shape, kernel):

    row_k = kernel.shape[0]
    matrix_record = []
    kernel = FZ(kernel)
    kernel = kernel.reshape(kernel.shape[2] * kernel.shape[3], kernel.shape[0], kernel.shape[1])

    try:
        H = np.zeros( (input_shape[1] + kernel.shape[1]-1, input_shape[1] + kernel.shape[1]-1 ) )
    except:
        a = out_shape[1]
        b = a

    sum_conv_matrix = np.zeros((out_shape[1] * out_shape[1], input_shape[0] * input_shape[1]))

    for k in range(1):

        try:
            H[H.shape[0]-row_k:H.shape[0] , 0:row_k] = kernel[k]
        except:
            a = kernel[k]
            b= H[out_shape[1]-row_k:out_shape[1] , 0:row_k]
            c = a


        for h in H[::-1, :]:
            r = np.zeros( (1, input_shape[1]) )
            r[0,0] = h[0]
            # for idx in range(1, len(r)):
            #     r[0][idx] = h[-idx]
            matrix_record.append( scipy.linalg.toeplitz(h, r) )

        conv_matrix = np.zeros((out_shape[1] * out_shape[1], input_shape[0] * input_shape[1]))
        i = 0
        j = 0

        for F in matrix_record:
            while i <= np.size(conv_matrix, 0) - F.shape[0] and j <= np.size(conv_matrix, 1) - F.shape[1]:
                conv_matrix[i:i+F.shape[0], j:j+F.shape[1]] = F
                i += F.shape[0]
                j += F.shape[1]

        sum_conv_matrix += conv_matrix

    return sum_conv_matrix/kernel.shape[0]

def im2col(input_data, filter_h, filter_w, stride, pad):
    """

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    H, W, C = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    N = H * W * C // 8

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((filter_h, filter_w, C, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[y, x, :, :, :] = img[y:y_max:stride, x:x_max:stride, :]

    col = col.transpose(4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def Vover(S,epsilon):
    """
    calculate the V_over under k = k
    :param S: set of points(1 x k), point reserved as matricx type 以行向量形式存储
    :param epsilon: converged value
    :return: u1 used to calculate MVEE(S)
    """
    # Step 0

    # initial
    k = S.shape[1]
    n = len(S)
    u = np.mat((1 / n) * np.ones(shape=[n, 1]))
    I = np.mat(np.eye(k))
    zero_d1 = np.mat(np.zeros(shape=[k,1]))
    zero_1d = np.mat(np.zeros(shape=[1,k]))
    u1 = np.mat((float('inf')) * np.ones(shape=[n, 1]))
    # k to 2k+1 dimension
    # for idx in range(n):
    #     arr = np.mat(np.int64(np.array(S[idx]) > 0) * np.array(S[idx]))
    #     if idx == 0:
    #         arr_add = arr
    #     else:
    #         arr_add = np.row_stack((arr_add, arr))

    # S = np.column_stack((S, arr_add))
    S = np.mat(S)
    P = S.T
    S_q = np.column_stack((S,np.mat(np.ones(shape=[n,1]))))
    # it = 0
    # while np.linalg.norm(u1 - u, 1) > epsilon:


    # Step 3
    m_value = -999
    for idx in range(n):
        value = S_q[idx] * pai_ni(u, P, I, zero_d1, zero_1d) * S_q[idx].T
        if value >= m_value:
            j = idx
            m_value = value

    # Step 4
    m_value = int(m_value)
    beta = (m_value - (k+1))/((k + 1)*(m_value))

    # Step 5
    ej = np.zeros(shape=[n,1])
    ej[j] = 1
    u1 = (1-beta) * u + beta * ej

    V = ( pow(2, k/2)*pow(pi,k/2)/(k) ) * pow(k, k/2) * pow( np.linalg.det(pai(u1, P, I, zero_d1, zero_1d)) , 0.5)

    return V

def Sigmoid(x):
    return 1/(1+exp(-x))

def Tanh(x):
    return (exp(x)-exp(-x))/(exp(x)+exp(-x))



class used:

    rest_points = []
    record = []
    conv_count = 0
    idx = 0
    record_idx = []
    weight_new = []
    bias_new = []
