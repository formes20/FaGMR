"""
作者：LENOVO
日期：2022年06月25日
"""
from scipy.spatial import ConvexHull
import numpy as np
import multiprocessing
import math
from config import *
# points = np.array([[1,0,0],[0,1,0],[0,0,0],[1,1,0],[1,0,1],[0,1,1],[0,0,1],[1,1,100]])  # your points
# volume = ConvexHull(points).volume
#
# c = [[1,1],[2,2]]
# b = [[3,3]]
# a = []
# a = c + b
# a = 1
# 一种随着权重向量中每一个元素正负动态获得上界的向量(array) upper_vector
def get_upper_vector(weight_index, lbi, ubi):
    # weight_index: weight(n*n)中属于index1 or 2的那一条权重向量(array)
    upper_vector = np.zeros(shape = [1,len(ubi[0])])
    m = weight_index[0]
    n = len(weight_index[0])
    for i in range(n):
        if m[i] >= 0:
            upper_vector[0][i] = ubi[0][i]
        else:
            upper_vector[0][i] = lbi[0][i]
    return upper_vector

def get_lower_vector(weight_index, lbi, ubi):
    # weight_index: weight(n*n)中属于index1 or 2的那一条权重向量(array)
    lower_vector = np.zeros(shape = [1,len(ubi[0])])
    m = weight_index[0]
    n = len(weight_index[0])
    for i in range(n):
        if m[i] <= 0:
            lower_vector[0][i] = ubi[0][i]
        else:
            lower_vector[0][i] = lbi[0][i]
    return lower_vector

# 定四条直线的方程（反应在每条直线的截距表达式上，作为输出）
def decision_intercept_no(weight, bias, lbi, ubi, index_fix, index_select):
    """
    :param weight: 当前层所有节点,每一行代表一个作为入度点的节点拥有的权重列向量 n*1
    :param bias: 当前层所有节点拥有的偏置值形成的行向量 1*n
    :param lbi: 当前层所有节点由一轮propagation形成下界的行向量 1*n
    :param ubi: 当前层所有节点由一轮propagation形成上界的行向量 1*n
    :param index_fix: 激活态未定的点的列表split_zero中的某一个点
    :param index_select: 激活态未定的点的列表split_zero中除index_fix的某一个点
    :return: J1,J2,J3,J4: 1~4号直线的截距，在之后的公式中就代表了这条直线的所有需要用到的性质
    """
    switch = 0
    # 求J1、2、3、4
    weight_J_12 = weight[index_fix,:].reshape(1, -1) - weight[index_select,:].reshape(1, -1)
    weight_J_34 = weight[index_fix,:].reshape(1, -1) + weight[index_select,:].reshape(1, -1)
    J1 = (-(np.matmul(weight_J_12, get_upper_vector(weight_J_12, lbi, ubi).transpose())[0]) - bias[0][index_fix] + bias[0][index_select])[0]
    J2 = (-(np.matmul(weight_J_12, get_lower_vector(weight_J_12, lbi, ubi).transpose())[0]) - bias[0][index_fix] + bias[0][index_select])[0]
    J3 = (np.matmul(weight_J_34, get_upper_vector(weight_J_34, lbi, ubi).transpose())[0] + bias[0][index_fix] + bias[0][index_select])[0]
    J4 = (np.matmul(weight_J_34, get_lower_vector(weight_J_34, lbi, ubi).transpose())[0] + bias[0][index_fix] + bias[0][index_select])[0]

    # 根据大小重新排列J1、2、3、4：J2 > J1, J3 > J4
    if J2 < J1:
        switch = J2
        J2 = J1
        J1 = switch

    if J3 < J4:
        switch = J3
        J3 = J4
        J4 = J3

    return J1,J2,J3,J4

# 计算四维凸包之上下界，等价于计算三维体积
def calculate_3D_vol(weight, bias, lbi, ubi, idx_fix, idx_select, prebound_l, prebound_u):
    """
    :param weight: 权重矩阵，每一行是一个rudu点拥有的权重向量array
    :param bias: 偏置向量array
    :param lbi: 该层所有点的下界形成的行矩阵array
    :param ubi: 该层所有点的上界形成的行矩阵array
    :param i: 固定节点在整层中的索引
    :param j: 选择节点在整层中的索引
    :return: V_lower, V_upper:等价于四维凸包的下界、上界
    """
    V_lower = 0
    V_upper = 0

    # 定四条直线的方程（反应在每条直线的截距表达式上，作为输出）
    J1, J2, J3, J4 = decision_intercept_no(weight, bias, prebound_l, prebound_u, idx_fix, idx_select)

    # 先列出所有要用到的二维上的点坐标
    # 2、3交于A；1、3交于B；1、4交于C；2、4交于D
    A = [0.5 * (J3 - J2), 0.5 * (J3 + J2),0]
    B = [0.5 * (J3 - J1), 0.5 * (J3 + J1),0]
    C = [0.5 * (J4 - J1), 0.5 * (J4 + J1),0]
    D = [0.5 * (J4 - J2), 0.5 * (J4 + J2),0]
    # 以下是（草稿上）x3（idx_fix）和x4(idx_select)的下上界
    l3 = lbi[0][idx_fix]
    u3 = ubi[0][idx_fix]
    l4 = lbi[0][idx_select]
    u4 = ubi[0][idx_select]

    # 共16种情况，虽有镜像、对称之巧，然亦须一一列出
    # 记录一些点
    # u4与（2）、x4轴、（1）、（3）、（4）交于H、G、F、A1、A2
    H = [u4 - J2, u4, 0]
    G = [0, u4, 0]
    F = [u4 - J1, u4, 0]
    A1 = [J3 - u4, u4, 0]
    A2 = [J4 - u4, u4, 0]
    # l4与（2）、x4轴、（1）、u3、（4）、(3)分别交于J、K、N、P1、D1、B2
    J = [l4 - J2, l4, 0]
    K = [0, l4, 0]
    N = [l4 - J1, l4, 0]
    P1 = [u3, l4, 0]
    D1 = [J4 - l4, l4, 0]
    B2 = [J3 - l4, l4, 0]
    # l3与（2）、u4、x3轴、l4、（4）、（1）、（3）分别交于L、M、K_p、M_p、L1、L2、L3
    L = [l3, l3 + J2, 0]
    M = [l3, u4, 0]
    K_p = [l3, 0, 0]
    M_p = [l3, l4, 0]
    L1 = [l3, J4 - l3, 0]
    L2 = [l3, J1 + l3, 0]
    L3 = [l3, J3 - l3, 0]
    # u3与（3）、（1）、u4、x3、(2)、(4)交于P、Q、P2、R、H1、H2
    P = [u3, J3 - u3, 0]
    Q = [u3, J1 + u3, 0]
    P2 = [u3, u4, 0]
    R = [u3, 0, 0]
    H1 = [u3, u3 + J2, 0]
    H2 = [u3, J4 - u3, 0]
    # 其他点
    I = [-J2, 0, 0]  # （2）与x3（横轴）的交点
    B1 = [0, J2, 0]  # （2）与x4（纵轴）的交点
    E = [-J1, 0, 0]  # （1）与x3的交点
    C1 = [0, J1, 0]  # （1）与x4的交点
    B3 = [0, J3, 0]
    I1 = [J3, 0, 0]
    C2 = [0, J4, 0]
    D2 = [J4, 0, 0]
    O = [0, 0, 0]

    l3_shang = []
    l3_xia = []
    l3_x3 = []
    u3_shang = []
    u3_xia = []
    u3_x3 = []
    list_lower = []
    list_upper = []
    # sig1象征要做上下镜像了
    sig1 = 0
    # 情况 01
    if A[0] >= 0 and A[1] >= 0 and B[0] >= 0 and B[1] >= 0 and C[0] <= 0 and C[1] <= 0 and D[0] <= 0 and D[1] <= 0:

        if min(B[1], B1[1]) >= u4 > 0 >= l4 > max(D[1], C1[1]):
            # 构建l3 lower 和 upper 模块
            if l3 <= J[0]:
                l3_shang = [G,H]
                l3_xia = [J,K]
                l3_x3 = [I,O]
            elif l3 > J[0] and l3 <= I[0]:
                l3_shang = [G, H]
                l3_xia = [L,M_p,K]
                l3_x3 = [I,O]
            elif l3 > I[0] and l3 <= H[0]:
                l3_shang = [G,H,L]
                l3_xia = [M_p,K]
                l3_x3 = [K_p,O]
            else:
                l3_shang = [G,M]
                l3_xia = [M_p,K]
                l3_x3 = [K_p,O]

            if u3 > F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = [N]
            elif u3 > E[0] and u3 <= F[0]:
                u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = [N]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q,N]
                else:
                    u3_xia = [P1]

        elif 0 < u4 <= min(B[1], B1[1]) and max(D[1], C1[1]) >= l4 > min(D[1], C1[1]):
            # 构建l3 lower 和 upper 模块
            if l3 <= max(J[0], D[0]):
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if C1[1] > D[1]:
                    l3_xia = [J, N, C1]
                else:
                    l3_xia = [D, D1, K]
            elif l3 > max(J[0], D[0]) and l3 <= I[0]:
                l3_shang = [G, H]
                if C1[1] > D[1]:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
                else:
                    if L1[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                l3_xia.append(L)
                l3_x3 = [I, O]
            elif l3 > I[0] and l3 <= H[0]:
                l3_shang = [G, H, L]
                if C1[1] > D[1]:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
                else:
                    if L1[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                l3_x3 = [K_p, O]
            else:
                l3_shang = [G, M]
                if C1[1] > D[1]:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
                else:
                    if L1[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                l3_x3 = [K_p, O]

            if u3 > F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                if C1[1] > D[1]:
                    u3_xia = []
                else:
                    u3_xia = [N]
            elif u3 > E[0] and u3 <= F[0]:
                u3_shang = [P2, Q]
                u3_x3 = [E]
                if C1[1] > D[1]:
                    u3_xia = []
                else:
                    u3_xia = [N]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if C1[1] > D[1]:
                    u3_xia = [Q]
                else:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]

        elif 0 < u4 <= min(B[1], B1[1]) and C[1] < l4 <= min(D[1], C1[1]):
            # 构建l3 lower 和 upper 模块
            if l3 <= D[0]:
                l3_shang = [G,H]
                l3_xia = [D,D1,N,C]
                l3_x3 = [I,O]
            elif l3 > D[0] and l3 <= I[0]:
                l3_shang = [G, H]
                if L1[0] <= D1[0]:
                    l3_xia = [L,C1,L1,D1,N]
                elif L1[0] > D1[0] and L1[0] <= N[0]:
                    l3_xia = [L,C1,M_p,N]
                else:
                    l3_xia = [L,C1,L2]
                l3_x3 = [I,O]
            else:
                l3_shang = [G,H,L]
                if L1[0] <= D1[0]:
                    l3_xia = [L, C1, L1, D1, N]
                elif L1[0] > D1[0] and L1[0] <= N[0]:
                    l3_xia = [L, C1, M_p, N]
                else:
                    l3_xia = [L, C1, L2]
                l3_x3 = [K_p,O]

            if u3 > F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = []
            elif u3 > E[0] and u3 <= F[0]:
                u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif 0 < u4 <= min(B[1], B1[1]) and l4 <= C[1]:
            # 构建l3 lower 和 upper 模块
            if l3 <= D[0]:
                l3_shang = [G,H]
                l3_xia = [D,C1,C]
                l3_x3 = [I,O]
            elif l3 > D[0] and l3 <= I[0]:
                l3_shang = [G, H]
                if L1[0] > C[0]:
                    l3_xia = [L,C1,L2]
                else:
                    l3_xia = [L,L1,C,C1]
                l3_x3 = [I,O]
            else:
                l3_shang = [G,H,L]
                if L1[0] > C[0]:
                    l3_xia = [C1, L2]
                else:
                    l3_xia = [L1, C, C1]
                l3_x3 = [K_p,O]

            if u3 > F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = []
            elif u3 > E[0] and u3 <= F[0]:
                u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif min(B[1], B1[1]) < u4 <= max(B[1], B1[1]) and 0 >= l4 > max(D[1], C1[1]):
            # 构建l3 lower 和 upper 模块
            if l3 <= J[0]:
                if B[1] > B1[1]:
                    l3_shang = [B1]
                    l3_x3 = [I,O]
                else:
                    l3_shang = [G,H]
                    l3_x3 = [I,O]
                l3_xia = [J, K]
            elif l3 > J[0] and l3 <= I[0]:
                if B[1] > B1[1]:
                    l3_shang = [B1]
                    l3_x3 = [I, O]
                else:
                    l3_shang = [G, H]
                    l3_x3 = [I, O]
                l3_xia = [L,M_p,K]
            else:
                if B[1] > B1[1]:
                    l3_shang = [L,B1]
                    l3_x3 = [I, O]
                else:
                    if M[0] > H[0]:
                        l3_shang = [M,G]
                        l3_x3 = [K_p,O]
                    else:
                        l3_shang = [G,H,L]
                        l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > min(B[0],F[0]):
                if B[1] > B1[1]:
                    u3_shang = [F,H]
                    u3_x3 = [E]
                else:
                    u3_shang = [A1,B]
                    u3_x3 = [E]
                u3_xia = [N]
            elif u3 > E[0] and u3 <= min(B[0],F[0]):
                if B[1] > B1[1]:
                    u3_shang = [P2, Q, H]
                    u3_x3 = [E]
                else:
                    if P2[0] > A1[0]:
                        u3_shang = [A1,P,Q]
                    else:
                        u3_shang = [P2,Q]
                    u3_x3 = [E]
                u3_xia = [N]
            else:
                if B[1] > B1[1]:
                    if P2[0] > H[0]:
                        u3_shang = [P2,H]
                    else:
                        u3_shang =[H1]
                    u3_x3 = [R]
                else:
                    u3_x3 = [R]
                    if P2[0] > A1[0]:
                        u3_shang = [A1,P]
                    else:
                        u3_shang = [P2]
                if P1[0] > N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif min(B[1], B1[1]) < u4 <= max(B[1], B1[1]) and max(D[1], C1[1]) >= l4 > min(D[1], C1[1]):
            # 构建l3 lower 和 upper 模块
            if l3 <= max(J[0],D[0]):
                if B[1] > B1[1]:
                    l3_shang = [B1]
                    l3_x3 = [I,O]
                else:
                    l3_shang = [G,H]
                    l3_x3 = [I,O]
                if C1[1] > D[1]:
                    l3_xia = [J, N, C1]
                else:
                    l3_xia = [D,D1,K]
            elif max(J[0], D[0]) < l3 <= I[0]:
                if B[1] > B1[1]:
                    l3_shang = [B1]
                    l3_x3 = [I, O]
                    if C1[1] > D[1]:
                        if M_p[1] > N[1]:
                            l3_xia = [L2, C1]
                        else:
                            l3_xia = [M_p, N, C1]
                    else:
                        if L1[1] > D[1]:
                            l3_xia = [M_p, K]
                        else:
                            l3_xia = [L1, D1, K]
                else:
                    l3_shang = [G, H]
                    l3_x3 = [I, O]
                    if C1[1] > D[1]:
                        if M_p[1] > N[1]:
                            l3_xia = [L2,C1,L]
                        else:
                            l3_xia = [M_p,N,C1,L]
                    else:
                        if L1[1] > D[1]:
                            l3_xia = [M_p, K,L]
                        else:
                            l3_xia = [L1,D1,K,L]
            else:
                if B[1] > B1[1]:
                    l3_shang = [L,B1]
                    l3_x3 = [I, O]
                else:
                    if M[0] > H[0]:
                        l3_shang = [M,G]
                        l3_x3 = [K_p,O]
                    else:
                        l3_shang = [G,H,L]
                        l3_x3 = [K_p, O]
                if C1[1] > D[1]:
                    if M_p[1] > N[1]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
                else:
                    if L1[1] > D[1]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]

            if u3 > min(B[0],F[0]):
                if B[1] > B1[1]:
                    u3_shang = [F,H]
                    u3_x3 = [E]
                else:
                    u3_shang = [A1,B]
                    u3_x3 = [E]
                if C1[1] > D[1]:
                    u3_xia = []
                else:
                    u3_xia = [N]
            elif E[0] < u3 <= min(B[0], F[0]):
                if B[1] > B1[1]:
                    u3_shang = [P2, Q, H]
                    u3_x3 = [E]
                else:
                    if P2[0] > A1[0]:
                        u3_shang = [A1,P,Q]
                    else:
                        u3_shang = [P2,Q]
                    u3_x3 = [E]
                if C1[1] > D[1]:
                    u3_xia = []
                else:
                    u3_xia = [N]
            else:
                if B[1] > B1[1]:
                    if P2[0] > H[0]:
                        u3_shang = [P2,H]
                    else:
                        u3_shang =[H1]
                    u3_x3 = [R]
                else:
                    u3_x3 = [R]
                    if P2[0] > A1[0]:
                        u3_shang = [A1,P]
                    else:
                        u3_shang = [P2]
                if C1[0] > D[0]:
                    u3_xia = [Q]
                else:
                    if P1[0] > N[0]:
                        u3_xia = [Q,N]
                    else:
                        u3_xia = [P1]

        elif min(B[1], B1[1]) < u4 <= max(B[1], B1[1]) and min(D[1], C1[1]) >= l4 > C[1]:
            # 构建l3 lower 和 upper 模块
            if l3 <= D[0]:
                if B[1] > B1[1]:
                    l3_shang = [B1]
                    l3_x3 = [I,O]
                else:
                    l3_shang = [G,H]
                    l3_x3 = [I,O]
                l3_xia = [D,D1,N,C1]
            elif D[0] < l3 <= I[0]:
                if B[1] > B1[1]:
                    l3_shang = [B1]
                    l3_x3 = [I, O]
                else:
                    l3_shang = [G, H]
                    l3_x3 = [I, O]
                if L1[0] <= D1[0]:
                    l3_xia = [L,L1,D,N,C1]
                elif D1[0] < L1[0] <= N[0]:
                    l3_xia = [L,C1,M_p,N]
                else:
                    l3_xia = [L,C1,L2]
            else:
                if B[1] > B1[1]:
                    l3_shang = [L,B1]
                    l3_x3 = [I, O]
                else:
                    if M[0] > H[0]:
                        l3_shang = [M,G]
                        l3_x3 = [K_p,O]
                    else:
                        l3_shang = [G,H,L]
                        l3_x3 = [K_p, O]
                if L1[0] <= D1[0]:
                    l3_xia = [L, L1, D, N, C1]
                elif D1[0] < L1[0] <= N[0]:
                    l3_xia = [L, C1, M_p, N]
                else:
                    l3_xia = [L, C1, L2]

            if u3 > min(B[0],F[0]):
                if B[1] > B1[1]:
                    u3_shang = [F,H]
                    u3_x3 = [E]
                else:
                    u3_shang = [A1,B]
                    u3_x3 = [E]
                u3_xia = []
            elif E[0] < u3 <= min(B[0], F[0]):
                if B[1] > B1[1]:
                    u3_shang = [P2, Q, H]
                    u3_x3 = [E]
                else:
                    if P2[0] > A1[0]:
                        u3_shang = [A1,P,Q]
                    else:
                        u3_shang = [P2,Q]
                    u3_x3 = [E]
                u3_xia = []
            else:
                if B[1] > B1[1]:
                    if P2[0] > H[0]:
                        u3_shang = [P2,H]
                    else:
                        u3_shang =[H1]
                    u3_x3 = [R]
                else:
                    u3_x3 = [R]
                    if P2[0] > A1[0]:
                        u3_shang = [A1,P]
                    else:
                        u3_shang = [P2]
                u3_xia = [Q]

        elif min(B[1], B1[1]) < u4 <= max(B[1], B1[1]) and l4 <= C[1]:
            # 构建l3 lower 和 upper 模块
            if l3 <= D[0]:
                if B[1] > B1[1]:
                    l3_shang = [B1]
                    l3_x3 = [I,O]
                else:
                    l3_shang = [G,H]
                    l3_x3 = [I,O]
                l3_xia = [D,D1,C1]
            elif D[0] < l3 <= I[0]:
                if B[1] > B1[1]:
                    l3_shang = [B1]
                    l3_x3 = [I, O]
                else:
                    l3_shang = [G, H]
                    l3_x3 = [I, O]
                if L1[0] > C[0]:
                    l3_xia = [L,L2,C1]
                else:
                    l3_xia = [L,C1,L1,C]
            else:
                if B[1] > B1[1]:
                    l3_shang = [L,B1]
                    l3_x3 = [I, O]
                else:
                    if M[0] > H[0]:
                        l3_shang = [M,G]
                        l3_x3 = [K_p,O]
                    else:
                        l3_shang = [G,H,L]
                        l3_x3 = [K_p, O]
                if L1[0] > C[0]:
                    l3_xia = [L, L2, C1]
                else:
                    l3_xia = [L, C1, L1, C]

            if u3 > min(B[0],F[0]):
                if B[1] > B1[1]:
                    u3_shang = [F,H]
                    u3_x3 = [E]
                else:
                    u3_shang = [A1,B]
                    u3_x3 = [E]
                u3_xia = []
            elif E[0] < u3 <= min(B[0], F[0]):
                if B[1] > B1[1]:
                    u3_shang = [P2, Q, H]
                    u3_x3 = [E]
                else:
                    if P2[0] > A1[0]:
                        u3_shang = [A1,P,Q]
                    else:
                        u3_shang = [P2,Q]
                    u3_x3 = [E]
                u3_xia = []
            else:
                if B[1] > B1[1]:
                    if P2[0] > H[0]:
                        u3_shang = [P2,H]
                    else:
                        u3_shang =[H1]
                    u3_x3 = [R]
                else:
                    u3_x3 = [R]
                    if P2[0] > A1[0]:
                        u3_shang = [A1,P]
                    else:
                        u3_shang = [P2]
                u3_xia = [Q]

        elif max(B[1],B1[1]) < u4 <= A[1] and 0 >= l4 > max(D[1], C1[1]):
            if l3 <= J[0]:
                l3_shang = [B1]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [B1]
                l3_x3 = [I,O]
                l3_xia = [L,M_p,K]
            else:
                l3_shang = [B1,L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B[0]:
                u3_shang = [H,A1,B]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] < u3 <= B[0]:
                u3_x3 = [E]
                u3_xia = [N]
                if P2[0] > A1[0]:
                    u3_shang = [Q,H,A1,P]
                elif A1[0] >= P2[0] >= H[0]:
                    u3_shang = [Q,H,P2]
                else:
                    u3_shang = [Q,H1]
            else:
                u3_x3 = [R]
                if P2[0] > A1[0]:
                    u3_shang = [H,A1,P]
                elif A1[0] >= P2[0] >= H[0]:
                    u3_shang = [H,P2]
                else:
                    u3_shang = [H1]
                if P1[0] > N[0]:
                    u3_xia = [Q,N]
                else:
                    u3_xia = [P1]

        elif max(B[1], B1[1]) < u4 <= A[1] and max(D[1],C1[1]) >= l4 > min(D[1], C1[1]):
            if l3 <= min(J[0],D[0]):
                l3_shang = [B1]
                l3_x3 = [I, O]
                if C1[1] > D[1]:
                    l3_xia = [J, N, C1]
                else:
                    l3_xia = [D,D1,K]
            elif min(J[0],D[0]) < l3 <= I[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                if C1[1] > D[1]:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p,N,C1]
                else:
                    if L[0] > D[0]:
                        l3_xia = [M_p,K]
                    else:
                        l3_xia = [L1,D1,K]
            else:
                l3_shang = [B1, L]
                l3_x3 = [K_p, O]
                if C1[1] > D[1]:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
                else:
                    if L1[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]

            if u3 > B[0]:
                u3_shang = [H, A1, B]
                u3_x3 = [E]
                if C1[1] > D[1]:
                    u3_xia = []
                else:
                    u3_xia = [N]
            elif E[0] < u3 <= B[0]:
                u3_x3 = [E]
                if C1[1] > D[1]:
                    u3_xia = []
                else:
                    u3_xia = [N]
                if P2[0] > A1[0]:
                    u3_shang = [Q,H, A1, P]
                elif A1[0] >= P2[0] >= H[0]:
                    u3_shang = [Q,H, P2]
                else:
                    u3_shang = [H1,Q]
            else:
                u3_x3 = [R]
                if P2[0] > A1[0]:
                    u3_shang = [H, A1, P]
                elif A1[0] >= P2[0] >= H[0]:
                    u3_shang = [H, P2]
                else:
                    u3_shang = [H1]
                if C1[1] > D[1]:
                    u3_xia = [Q]
                else:
                    if P1[0] > N[0]:
                        u3_xia = [Q,N]
                    else:
                        u3_xia = [P1]

        elif max(B[1], B1[1]) < u4 <= A[1] and min(D[1],C1[1]) >= l4 > C[1]:
            if l3 <= D[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                if L1[0] <= D[0]:
                    l3_xia = [L1,D,N,C1,L]
                elif D1[0] < L1[0] < N[0]:
                    l3_xia = [M_p,N,C1,L]
                else:
                    l3_xia = [L2,C1,L]
            else:
                l3_shang = [B1, L]
                l3_x3 = [K_p, O]
                if L1[0] <= D[0]:
                    l3_xia = [L1, D, N, C1]
                elif D1[0] < L1[0] < N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]

            if u3 > B[0]:
                u3_shang = [H, A1, B]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] < u3 <= B[0]:
                u3_x3 = [E]
                u3_xia = []
                if P2[0] > A1[0]:
                    u3_shang = [Q,H, A1, P]
                elif A1[0] >= P2[0] >= H[0]:
                    u3_shang = [Q,H, P2]
                else:
                    u3_shang = [H1,Q]
            else:
                u3_x3 = [R]
                if P2[0] > A1[0]:
                    u3_shang = [H, A1, P]
                elif A1[0] >= P2[0] >= H[0]:
                    u3_shang = [H, P2]
                else:
                    u3_shang = [H1]
                u3_xia = [Q]

        elif max(B[1], B1[1]) < u4 <= A[1] and l4 <= C[1]:
            if l3 <= D[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                l3_xia = [D,C,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                if L1[0] > C[0]:
                    l3_xia = [L2,C1,L]
                else:
                    l3_xia = [L1,C1,L,C]
            else:
                l3_shang = [B1, L]
                l3_x3 = [K_p, O]
                if L1[0] > C[0]:
                    l3_xia = [L2,C1]
                else:
                    l3_xia = [L1,C1,C]

            if u3 > B[0]:
                u3_shang = [H, A1, B]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] < u3 <= B[0]:
                u3_x3 = [E]
                u3_xia = []
                if P2[0] > A1[0]:
                    u3_shang = [Q,H, A1, P]
                elif A1[0] >= P2[0] >= H[0]:
                    u3_shang = [Q,H, P2]
                else:
                    u3_shang = [H1,Q]
            else:
                u3_x3 = [R]
                if P2[0] > A1[0]:
                    u3_shang = [H, A1, P]
                elif A1[0] >= P2[0] >= H[0]:
                    u3_shang = [H, P2]
                else:
                    u3_shang = [H1]
                u3_xia = [Q]

        elif u4 > A[1] and l4 > max(D[1],C1[1]):
            if l3 <= J[0]:
                l3_shang = [B1]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                l3_xia = [L,M_p,K]
            else:
                l3_shang = [B1,L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B[0]:
                u3_shang = [A,B]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] < u3 <= B[0]:
                if H1[0] > A[0]:
                    u3_shang = [A, P,Q]
                else:
                    u3_shang = [H1,Q]
                u3_x3 = [E]
                u3_xia = [N]
            else:
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q,N]
                else:
                    u3_xia = [P1]
                if H1[0] > A[0]:
                    u3_shang = [A, P]
                else:
                    u3_shang = [H1]

        elif u4 > A[1] and min(D[1], C1[1]) < l4 <= max(D[1], C1[1]):
            if l3 <= max(J[0],D[0]):
                l3_shang = [B1]
                l3_x3 = [I, O]
                if C1[1] > D[1]:
                    l3_xia = [J, N, C1]
                else:
                    l3_xia = [D, D1, K]
            elif max(J[0],D[0]) < l3 <= I[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                if C1[1] > D[1]:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
                else:
                    if L[0] > D[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
            else:
                l3_shang = [B1, L]
                l3_x3 = [K_p, O]
                if C1[1] > D[1]:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
                else:
                    if L1[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]

            if u3 > B[0]:
                u3_shang = [A, B]
                u3_x3 = [E]
                if C1[1] > D[1]:
                    u3_xia = []
                else:
                    u3_xia = [N]
            elif E[0] < u3 <= B[0]:
                if H1[0] > A[0]:
                    u3_shang = [A, P, Q]
                else:
                    u3_shang = [H1, Q]
                u3_x3 = [E]
                if C1[1] > D[1]:
                    u3_xia = []
                else:
                    u3_xia = [N]
            else:
                u3_x3 = [R]
                if C1[1] > D[1]:
                    u3_xia = [Q]
                else:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                if H1[0] > A[0]:
                    u3_shang = [A, P]
                else:
                    u3_shang = [H1]

        elif u4 > A[1] and C[1] < l4 <= min(D[1], C1[1]):
            if l3 <= D[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                if L1[0] <= D[0]:
                    l3_xia = [L1, D, N, C1, L]
                elif D1[0] < L1[0] < N[0]:
                    l3_xia = [M_p, N, C1, L]
                else:
                    l3_xia = [L2, C1, L]
            else:
                l3_shang = [B1, L]
                l3_x3 = [K_p, O]
                if L1[0] <= D[0]:
                    l3_xia = [L1, D, N, C1]
                elif D1[0] < L1[0] < N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]

            if u3 > B[0]:
                u3_shang = [A, B]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] < u3 <= B[0]:
                if H1[0] > A[0]:
                    u3_shang = [A, P, Q]
                else:
                    u3_shang = [H1, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_x3 = [R]
                u3_xia = [Q]
                if H1[0] > A[0]:
                    u3_shang = [A, P]
                else:
                    u3_shang = [H1]

        else:
            if l3 <= D[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                l3_xia = [D,C,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                if L1[0] > C[0]:
                    l3_xia = [L2, C1, L]
                else:
                    l3_xia = [L1, C1, L, C]
            else:
                l3_shang = [B1, L]
                l3_x3 = [K_p, O]
                if L1[0] > C[0]:
                    l3_xia = [L2, C1]
                else:
                    l3_xia = [L1, C1, C]

            if u3 > B[0]:
                u3_shang = [A, B]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] < u3 <= B[0]:
                if H1[0] > A[0]:
                    u3_shang = [A, P, Q]
                else:
                    u3_shang = [H1, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_x3 = [R]
                u3_xia = [Q]
                if H1[0] > A[0]:
                    u3_shang = [A, P]
                else:
                    u3_shang = [H1]

    # 对情况 01 左右镜像！
    elif A[0] < 0 and A[1] > 0 and B[0] > 0 and B[1] < 0 and C[0] > 0 and C[1] < 0 and D[0] < 0 and D[1] > 0:
        # 按左右镜像对偶关系更换各点的值，不换点的名字
        A = [0.5 * (J3 + J2), 0.5 * (J3 - J2),0]
        B = [0.5 * (J4 + J2), 0.5 * (J4 - J2),0]
        C = [0.5 * (J4 + J1), 0.5 * (J4 - J1),0]
        D = [0.5 * (J3 + J1), 0.5 * (J3 - J1),0]
        H = [-u4 + J3, u4, 0]
        G = [0, u4, 0]
        F = [-u4 + J4, u4, 0]
        A1 = [-J2 + u4, u4, 0]
        A2 = [-J1 + u4, u4, 0]
        J = [-l4 + J3, l4, 0]
        K = [0, l4, 0]
        N = [-l4 + J4, l4, 0]
        P1 = [l3, l4, 0]
        D1 = [-J1 + l4, l4, 0]
        B2 = [-J2 + l4, l4, 0]
        L = [u3, -u3 + J3, 0]
        M = [u3, u4, 0]
        K_p = [u3, 0, 0]
        M_p = [u3, l4, 0]
        L1 = [u3, J1 + u3, 0]
        L2 = [u3, J4 - u3, 0]
        L3 = [u3, J2 + u3, 0]
        P = [l3, J2 + l3, 0]
        Q = [l3, J4 - l3, 0]
        P2 = [l3, u4, 0]
        R = [l3, 0, 0]
        H1 = [l3, -l3 + J3, 0]
        H2 = [l3, J1 + l3, 0]
        I = [-J3, 0, 0]
        B1 = [0, J3, 0]
        E = [-J4, 0, 0]
        C1 = [0, J4, 0]
        B3 = [0, J2, 0]
        I1 = [J2, 0, 0]
        C2 = [0, J1, 0]
        D2 = [J1, 0, 0]
        O = [0, 0, 0]

        # 情况 01之左右镜像
        if min(B[1], B1[1]) >= u4 > 0 >= l4 > max(D[1], C1[1]):
            if u3 >= J[0]:
                l3_shang = [G,H]
                l3_xia = [J,K]
                l3_x3 = [I,O]
            elif u3 < J[0] and u3 >= I[0]:
                l3_shang = [G, H]
                l3_xia = [L,M_p,K]
                l3_x3 = [I,O]
            elif u3 < I[0] and u3 >= H[0]:
                l3_shang = [G,H,L]
                l3_xia = [M_p,K]
                l3_x3 = [K_p,O]
            else:
                l3_shang = [G,M]
                l3_xia = [M_p,K]
                l3_x3 = [K_p,O]

            if l3 < F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = [N]
            elif l3 < E[0] and l3 >= F[0]:
                u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = [N]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q,N]
                else:
                    u3_xia = [P1]

        elif 0 < u4 <= min(B[1], B1[1]) and max(D[1], C1[1]) >= l4 > min(D[1], C1[1]):
            # 构建l3 lower 和 upper 模块
            if u3 >= min(J[0], D[0]):
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if C1[1] > D[1]:
                    l3_xia = [J, N, C1]
                else:
                    l3_xia = [D, D1, K]
            elif u3 < min(J[0], D[0]) and u3 >= I[0]:
                l3_shang = [G, H]
                if C1[1] > D[1]:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
                else:
                    if L1[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                l3_xia.append(L)
                l3_x3 = [I, O]
            elif u3 < I[0] and u3 >= H[0]:
                l3_shang = [G, H, L]
                if C1[1] > D[1]:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
                else:
                    if L1[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                l3_x3 = [K_p, O]
            else:
                l3_shang = [G, M]
                if C1[1] > D[1]:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
                else:
                    if L1[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                l3_x3 = [K_p, O]

            if l3 < F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                if C1[1] > D[1]:
                    u3_xia = []
                else:
                    u3_xia = [N]
            elif l3 < E[0] and l3 >= F[0]:
                u3_shang = [P2, Q]
                u3_x3 = [E]
                if C1[1] > D[1]:
                    u3_xia = []
                else:
                    u3_xia = [N]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if C1[1] > D[1]:
                    u3_xia = [Q]
                else:
                    if P1[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]

        elif 0 < u4 <= min(B[1], B1[1]) and C[1] < l4 <= min(D[1], C1[1]):
            # 构建l3 lower 和 upper 模块
            if u3 >= D[0]:
                l3_shang = [G,H]
                l3_xia = [D,D1,N,C]
                l3_x3 = [I,O]
            elif u3 < D[0] and u3 >= I[0]:
                l3_shang = [G, H]
                if L1[0] >= D1[0]:
                    l3_xia = [L,C1,L1,D1,N]
                elif L1[0] < D1[0] and L1[0] >= N[0]:
                    l3_xia = [L,C1,M_p,N]
                else:
                    l3_xia = [L,C1,L2]
                l3_x3 = [I,O]
            else:
                l3_shang = [G,H,L]
                if L1[0] >= D1[0]:
                    l3_xia = [L, C1, L1, D1, N]
                elif L1[0] < D1[0] and L1[0] >= N[0]:
                    l3_xia = [L, C1, M_p, N]
                else:
                    l3_xia = [L, C1, L2]
                l3_x3 = [K_p,O]

            if l3 < F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = []
            elif l3 < E[0] and l3 >= F[0]:
                u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif 0 < u4 <= min(B[1], B1[1]) and l4 <= C[1]:
            # 构建l3 lower 和 upper 模块
            if u3 >= D[0]:
                l3_shang = [G,H]
                l3_xia = [D,C1,C]
                l3_x3 = [I,O]
            elif u3 < D[0] and u3 >= I[0]:
                l3_shang = [G, H]
                if L1[0] < C[0]:
                    l3_xia = [L,C1,L2]
                else:
                    l3_xia = [L,L1,C,C1]
                l3_x3 = [I,O]
            else:
                l3_shang = [G,H,L]
                if L1[0] < C[0]:
                    l3_xia = [C1, L2]
                else:
                    l3_xia = [L1, C, C1]
                l3_x3 = [K_p,O]

            if l3 < F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = []
            elif l3 < E[0] and l3 >= F[0]:
                u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif min(B[1], B1[1]) < u4 <= max(B[1], B1[1]) and 0 >= l4 > max(D[1], C1[1]):
            # 构建l3 lower 和 upper 模块
            if u3 >= J[0]:
                if B[1] > B1[1]:
                    l3_shang = [B1]
                    l3_x3 = [I,O]
                else:
                    l3_shang = [G,H]
                    l3_x3 = [I,O]
                l3_xia = [J, K]
            elif u3 < J[0] and u3 >= I[0]:
                if B[1] > B1[1]:
                    l3_shang = [B1]
                    l3_x3 = [I, O]
                else:
                    l3_shang = [G, H]
                    l3_x3 = [I, O]
                l3_xia = [L,M_p,K]
            else:
                if B[1] > B1[1]:
                    l3_shang = [L,B1]
                    l3_x3 = [I, O]
                else:
                    if M[0] < H[0]:
                        l3_shang = [M,G]
                        l3_x3 = [K_p,O]
                    else:
                        l3_shang = [G,H,L]
                        l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < max(B[0],F[0]):
                if B[1] > B1[1]:
                    u3_shang = [F,H]
                    u3_x3 = [E]
                else:
                    u3_shang = [A1,B]
                    u3_x3 = [E]
                u3_xia = [N]
            elif l3 < E[0] and l3 >= max(B[0],F[0]):
                if B[1] > B1[1]:
                    u3_shang = [P2, Q, H]
                    u3_x3 = [E]
                else:
                    if P2[0] < A1[0]:
                        u3_shang = [A1,P,Q]
                    else:
                        u3_shang = [P2,Q]
                    u3_x3 = [E]
                u3_xia = [N]
            else:
                if B[1] > B1[1]:
                    if P2[0] < H[0]:
                        u3_shang = [P2,H]
                    else:
                        u3_shang =[H1]
                    u3_x3 = [R]
                else:
                    u3_x3 = [R]
                    if P2[0] < A1[0]:
                        u3_shang = [A1,P]
                    else:
                        u3_shang = [P2]
                if P1[0] < N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif min(B[1], B1[1]) < u4 <= max(B[1], B1[1]) and max(D[1], C1[1]) >= l4 > min(D[1], C1[1]):
            # 构建l3 lower 和 upper 模块
            if u3 >= min(J[0],D[0]):
                if B[1] > B1[1]:
                    l3_shang = [B1]
                    l3_x3 = [I,O]
                else:
                    l3_shang = [G,H]
                    l3_x3 = [I,O]
                if C1[1] > D[1]:
                    l3_xia = [J, N, C1]
                else:
                    l3_xia = [D,D1,K]
            elif min(J[0], D[0]) > u3 >= I[0]:
                if B[1] > B1[1]:
                    l3_shang = [B1]
                    l3_x3 = [I, O]
                    if C1[1] > D[1]:
                        if M_p[0] < N[0]:
                            l3_xia = [L2, C1]
                        else:
                            l3_xia = [M_p, N, C1]
                    else:
                        if L1[0] < D[0]:
                            l3_xia = [M_p, K]
                        else:
                            l3_xia = [L1, D1, K]
                else:
                    l3_shang = [G, H]
                    l3_x3 = [I, O]
                    if C1[1] > D[1]:
                        if M_p[1] > N[1]:
                            l3_xia = [L2,C1,L]
                        else:
                            l3_xia = [M_p,N,C1,L]
                    else:
                        if L1[1] > D[1]:
                            l3_xia = [M_p, K,L]
                        else:
                            l3_xia = [L1,D1,K,L]
            else:
                if B[1] > B1[1]:
                    l3_shang = [L,B1]
                    l3_x3 = [I, O]
                else:
                    if M[0] < H[0]:
                        l3_shang = [M,G]
                        l3_x3 = [K_p,O]
                    else:
                        l3_shang = [G,H,L]
                        l3_x3 = [K_p, O]
                if C1[1] > D[1]:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
                else:
                    if L1[0] < D[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]

            if l3 < max(B[0],F[0]):
                if B[1] > B1[1]:
                    u3_shang = [F,H]
                    u3_x3 = [E]
                else:
                    u3_shang = [A1,B]
                    u3_x3 = [E]
                if C1[1] > D[1]:
                    u3_xia = []
                else:
                    u3_xia = [N]
            elif E[0] > l3 >= max(B[0], F[0]):
                if B[1] > B1[1]:
                    u3_shang = [P2, Q, H]
                    u3_x3 = [E]
                else:
                    if P2[0] < A1[0]:
                        u3_shang = [A1,P,Q]
                    else:
                        u3_shang = [P2,Q]
                    u3_x3 = [E]
                if C1[1] > D[1]:
                    u3_xia = []
                else:
                    u3_xia = [N]
            else:
                if B[1] > B1[1]:
                    if P2[0] < H[0]:
                        u3_shang = [P2,H]
                    else:
                        u3_shang =[H1]
                    u3_x3 = [R]
                else:
                    u3_x3 = [R]
                    if P2[0] < A1[0]:
                        u3_shang = [A1,P]
                    else:
                        u3_shang = [P2]
                if C1[1] > D[1]:
                    u3_xia = [Q]
                else:
                    if P1[0] < N[0]:
                        u3_xia = [Q,N]
                    else:
                        u3_xia = [P1]

        elif min(B[1], B1[1]) < u4 <= max(B[1], B1[1]) and min(D[1], C1[1]) >= l4 > C[1]:
            # 构建l3 lower 和 upper 模块
            if u3 >= D[0]:
                if B[1] > B1[1]:
                    l3_shang = [B1]
                    l3_x3 = [I,O]
                else:
                    l3_shang = [G,H]
                    l3_x3 = [I,O]
                l3_xia = [D,D1,N,C1]
            elif D[0] > u3 >= I[0]:
                if B[1] > B1[1]:
                    l3_shang = [B1]
                    l3_x3 = [I, O]
                else:
                    l3_shang = [G, H]
                    l3_x3 = [I, O]
                if L1[0] >= D1[0]:
                    l3_xia = [L,L1,D,N,C1]
                elif D1[0] > L1[0] >= N[0]:
                    l3_xia = [L,C1,M_p,N]
                else:
                    l3_xia = [L,C1,L2]
            else:
                if B[1] > B1[1]:
                    l3_shang = [L,B1]
                    l3_x3 = [I, O]
                else:
                    if M[0] < H[0]:
                        l3_shang = [M,G]
                        l3_x3 = [K_p,O]
                    else:
                        l3_shang = [G,H,L]
                        l3_x3 = [K_p, O]
                if L1[0] >= D1[0]:
                    l3_xia = [L, L1, D, N, C1]
                elif D1[0] > L1[0] >= N[0]:
                    l3_xia = [L, C1, M_p, N]
                else:
                    l3_xia = [L, C1, L2]

            if l3 > max(B[0],F[0]):
                if B[1] > B1[1]:
                    u3_shang = [F,H]
                    u3_x3 = [E]
                else:
                    u3_shang = [A1,B]
                    u3_x3 = [E]
                u3_xia = []
            elif E[0] > l3 >= max(B[0], F[0]):
                if B[1] > B1[1]:
                    u3_shang = [P2, Q, H]
                    u3_x3 = [E]
                else:
                    if P2[0] < A1[0]:
                        u3_shang = [A1,P,Q]
                    else:
                        u3_shang = [P2,Q]
                    u3_x3 = [E]
                u3_xia = []
            else:
                if B[1] > B1[1]:
                    if P2[0] < H[0]:
                        u3_shang = [P2,H]
                    else:
                        u3_shang =[H1]
                    u3_x3 = [R]
                else:
                    u3_x3 = [R]
                    if P2[0] < A1[0]:
                        u3_shang = [A1,P]
                    else:
                        u3_shang = [P2]
                u3_xia = [Q]

        elif min(B[1], B1[1]) < u4 <= max(B[1], B1[1]) and l4 <= C[1]:
            # 构建l3 lower 和 upper 模块
            if u3 >= D[0]:
                if B[1] > B1[1]:
                    l3_shang = [B1]
                    l3_x3 = [I,O]
                else:
                    l3_shang = [G,H]
                    l3_x3 = [I,O]
                l3_xia = [D,D1,C1]
            elif D[0] > u3 >= I[0]:
                if B[1] > B1[1]:
                    l3_shang = [B1]
                    l3_x3 = [I, O]
                else:
                    l3_shang = [G, H]
                    l3_x3 = [I, O]
                if L1[0] < C[0]:
                    l3_xia = [L,L2,C1]
                else:
                    l3_xia = [L,C1,L1,C]
            else:
                if B[1] > B1[1]:
                    l3_shang = [L,B1]
                    l3_x3 = [I, O]
                else:
                    if M[0] < H[0]:
                        l3_shang = [M,G]
                        l3_x3 = [K_p,O]
                    else:
                        l3_shang = [G,H,L]
                        l3_x3 = [K_p, O]
                if L1[0] < C[0]:
                    l3_xia = [L, L2, C1]
                else:
                    l3_xia = [L, C1, L1, C]

            if l3 < max(B[0],F[0]):
                if B[1] > B1[1]:
                    u3_shang = [F,H]
                    u3_x3 = [E]
                else:
                    u3_shang = [A1,B]
                    u3_x3 = [E]
                u3_xia = []
            elif E[0] > l3 >= min(B[0], F[0]):
                if B[1] > B1[1]:
                    u3_shang = [P2, Q, H]
                    u3_x3 = [E]
                else:
                    if P2[0] < A1[0]:
                        u3_shang = [A1,P,Q]
                    else:
                        u3_shang = [P2,Q]
                    u3_x3 = [E]
                u3_xia = []
            else:
                if B[1] > B1[1]:
                    if P2[0] < H[0]:
                        u3_shang = [P2,H]
                    else:
                        u3_shang =[H1]
                    u3_x3 = [R]
                else:
                    u3_x3 = [R]
                    if P2[0] < A1[0]:
                        u3_shang = [A1,P]
                    else:
                        u3_shang = [P2]
                u3_xia = [Q]

        elif max(B[1],B1[1]) < u4 <= A[1] and 0 >= l4 > max(D[1], C1[1]):
            if u3 >= J[0]:
                l3_shang = [B1]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [B1]
                l3_x3 = [I,O]
                l3_xia = [L,M_p,K]
            else:
                l3_shang = [B1,L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B[0]:
                u3_shang = [H,A1,B]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] > l3 >= B[0]:
                u3_x3 = [E]
                u3_xia = [N]
                if P2[0] < A1[0]:
                    u3_shang = [Q,H,A1,P]
                elif A1[0] <= P2[0] <= H[0]:
                    u3_shang = [Q,H,P2]
                else:
                    u3_shang = [Q,H1]
            else:
                u3_x3 = [R]
                if P2[0] < A1[0]:
                    u3_shang = [H,A1,P]
                elif A1[0] <= P2[0] <= H[0]:
                    u3_shang = [H,P2]
                else:
                    u3_shang = [H1]
                if P1[0] < N[0]:
                    u3_xia = [Q,N]
                else:
                    u3_xia = [P1]

        elif max(B[1], B1[1]) < u4 <= A[1] and max(D[1],C1[1]) >= l4 > min(D[1], C1[1]):
            if u3 >= max(J[0],D[0]):
                l3_shang = [B1]
                l3_x3 = [I, O]
                if C1[1] > D[1]:
                    l3_xia = [J, N, C1]
                else:
                    l3_xia = [D,D1,K]
            elif max(J[0],D[0]) > u3 >= I[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                if C1[1] > D[1]:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p,N,C1]
                else:
                    if L[0] < D[0]:
                        l3_xia = [M_p,K]
                    else:
                        l3_xia = [L1,D1,K]
            else:
                l3_shang = [B1, L]
                l3_x3 = [K_p, O]
                if C1[1] > D[1]:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
                else:
                    if L1[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]

            if l3 < B[0]:
                u3_shang = [H, A1, B]
                u3_x3 = [E]
                if C1[1] > D[1]:
                    u3_xia = []
                else:
                    u3_xia = [N]
            elif E[0] > l3 >= B[0]:
                u3_x3 = [E]
                if C1[1] > D[1]:
                    u3_xia = []
                else:
                    u3_xia = [N]
                if P2[0] < A1[0]:
                    u3_shang = [Q,H, A1, P]
                elif A1[0] <= P2[0] <= H[0]:
                    u3_shang = [Q,H, P2]
                else:
                    u3_shang = [H1,Q]
            else:
                u3_x3 = [R]
                if P2[0] < A1[0]:
                    u3_shang = [H, A1, P]
                elif A1[0] <= P2[0] <= H[0]:
                    u3_shang = [H, P2]
                else:
                    u3_shang = [H1]
                if C1[1] > D[1]:
                    u3_xia = [Q]
                else:
                    if P1[0] < N[0]:
                        u3_xia = [Q,N]
                    else:
                        u3_xia = [P1]

        elif max(B[1], B1[1]) < u4 <= A[1] and min(D[1],C1[1]) >= l4 > C[1]:
            if u3 >= D[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                if L1[0] >= D[0]:
                    l3_xia = [L1,D,N,C1,L]
                elif D1[0] > L1[0] > N[0]:
                    l3_xia = [M_p,N,C1,L]
                else:
                    l3_xia = [L2,C1,L]
            else:
                l3_shang = [B1, L]
                l3_x3 = [K_p, O]
                if L1[0] >= D[0]:
                    l3_xia = [L1, D, N, C1]
                elif D1[0] > L1[0] > N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]

            if l3 < B[0]:
                u3_shang = [H, A1, B]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] > l3 >= B[0]:
                u3_x3 = [E]
                u3_xia = []
                if P2[0] < A1[0]:
                    u3_shang = [Q,H, A1, P]
                elif A1[0] <= P2[0] <= H[0]:
                    u3_shang = [Q,H, P2]
                else:
                    u3_shang = [H1,Q]
            else:
                u3_x3 = [R]
                if P2[0] < A1[0]:
                    u3_shang = [H, A1, P]
                elif A1[0] <= P2[0] <= H[0]:
                    u3_shang = [H, P2]
                else:
                    u3_shang = [H1]
                u3_xia = [Q]

        elif max(B[1], B1[1]) < u4 <= A[1] and l4 <= C[1]:
            if u3 >= D[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                l3_xia = [D,C,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                if L1[0] < C[0]:
                    l3_xia = [L2,C1,L]
                else:
                    l3_xia = [L1,C1,L,C]
            else:
                l3_shang = [B1, L]
                l3_x3 = [K_p, O]
                if L1[0] < C[0]:
                    l3_xia = [L2,C1]
                else:
                    l3_xia = [L1,C1,C]

            if l3 < B[0]:
                u3_shang = [H, A1, B]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] > l3 >= B[0]:
                u3_x3 = [E]
                u3_xia = []
                if P2[0] < A1[0]:
                    u3_shang = [Q,H, A1, P]
                elif A1[0] <= P2[0] <= H[0]:
                    u3_shang = [Q,H, P2]
                else:
                    u3_shang = [H1,Q]
            else:
                u3_x3 = [R]
                if P2[0] < A1[0]:
                    u3_shang = [H, A1, P]
                elif A1[0] <= P2[0] <= H[0]:
                    u3_shang = [H, P2]
                else:
                    u3_shang = [H1]
                u3_xia = [Q]

        elif u4 > A[1] and l4 > max(D[1],C1[1]):
            if u3 >= J[0]:
                l3_shang = [B1]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                l3_xia = [L,M_p,K]
            else:
                l3_shang = [B1,L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B[0]:
                u3_shang = [A,B]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] > l3 >= B[0]:
                if H1[0] > A[0]:
                    u3_shang = [A, P,Q]
                else:
                    u3_shang = [H1,Q]
                u3_x3 = [E]
                u3_xia = [N]
            else:
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q,N]
                else:
                    u3_xia = [P1]
                if H1[0] < A[0]:
                    u3_shang = [A, P]
                else:
                    u3_shang = [H1]

        elif u4 > A[1] and min(D[1], C1[1]) < l4 <= max(D[1], C1[1]):
            if u3 >= min(J[0],D[0]):
                l3_shang = [B1]
                l3_x3 = [I, O]
                if C1[1] > D[1]:
                    l3_xia = [J, N, C1]
                else:
                    l3_xia = [D, D1, K]
            elif min(J[0],D[0]) > u3 >= I[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                if C1[1] > D[1]:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
                else:
                    if L[0] < D[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
            else:
                l3_shang = [B1, L]
                l3_x3 = [K_p, O]
                if C1[1] > D[1]:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
                else:
                    if L1[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]

            if l3 < B[0]:
                u3_shang = [A, B]
                u3_x3 = [E]
                if C1[1] > D[1]:
                    u3_xia = []
                else:
                    u3_xia = [N]
            elif E[0] > l3 >= B[0]:
                if H1[0] < A[0]:
                    u3_shang = [A, P, Q]
                else:
                    u3_shang = [H1, Q]
                u3_x3 = [E]
                if C1[1] > D[1]:
                    u3_xia = []
                else:
                    u3_xia = [N]
            else:
                u3_x3 = [R]
                if C1[1] > D[1]:
                    u3_xia = [Q]
                else:
                    if P1[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                if H1[0] < A[0]:
                    u3_shang = [A, P]
                else:
                    u3_shang = [H1]

        elif u4 > A[1] and C[1] < l4 <= min(D[1], C1[1]):
            if u3 >= D[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                if L1[0] >= D[0]:
                    l3_xia = [L1, D, N, C1, L]
                elif D1[0] > L1[0] > N[0]:
                    l3_xia = [M_p, N, C1, L]
                else:
                    l3_xia = [L2, C1, L]
            else:
                l3_shang = [B1, L]
                l3_x3 = [K_p, O]
                if L1[0] >= D[0]:
                    l3_xia = [L1, D, N, C1]
                elif D1[0] > L1[0] > N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]

            if l3 < B[0]:
                u3_shang = [A, B]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] > l3 >= B[0]:
                if H1[0] < A[0]:
                    u3_shang = [A, P, Q]
                else:
                    u3_shang = [H1, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_x3 = [R]
                u3_xia = [Q]
                if H1[0] < A[0]:
                    u3_shang = [A, P]
                else:
                    u3_shang = [H1]

        else:
            if u3 >= D[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                l3_xia = [D,C,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B1]
                l3_x3 = [I, O]
                if L1[0] < C[0]:
                    l3_xia = [L2, C1, L]
                else:
                    l3_xia = [L1, C1, L, C]
            else:
                l3_shang = [B1, L]
                l3_x3 = [K_p, O]
                if L1[0] < C[0]:
                    l3_xia = [L2, C1]
                else:
                    l3_xia = [L1, C1, C]

            if l3 < B[0]:
                u3_shang = [A, B]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] > l3 >= B[0]:
                if H1[0] < A[0]:
                    u3_shang = [A, P, Q]
                else:
                    u3_shang = [H1, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_x3 = [R]
                u3_xia = [Q]
                if H1[0] < A[0]:
                    u3_shang = [A, P]
                else:
                    u3_shang = [H1]

    # 情况 02
    elif A[0] <= 0 and A[1] >= 0 and B[0] >= 0 and B[1] <= 0 and C[0] >= 0 and C[1] <= 0 and D[0] <= 0 and D[1] <= 0:

        if u4 <= B3[1] and l4 > D[1]:
            if l3 <= J[0]:
                l3_shang = [G,H]
                l3_x3 = [O,I]
                l3_xia = [J,K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [H,G]
                l3_x3 = [I,O]
                l3_xia = [L,K,M_p]
            elif I[0] < l3 <= H[0]:
                l3_shang = [H,G,L]
                l3_x3 = [K_p,O]
                l3_xia = [M_p,K]
            else:
                l3_shang = [M,G]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I1[0] < u3 <= B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [P,P1]
            elif A1[0] < u3 <= I1[0]:
                u3_shang = [A1,P]
                u3_x3 = [R]
                u3_xia = [P1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [P1]

        elif u4 <= B3[1] and max(C2[1],B[1]) < l4 <= D[1]:
            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [O, I]
                l3_xia = [K,D,D1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [H, G]
                l3_x3 = [I, O]
                if L1[1] > M[1]:
                    l3_xia = [L1,L,K,D1]
                else:
                    l3_xia = [M_p,L,K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [H, G, L]
                l3_x3 = [K_p, O]
                if L1[1] > M[1]:
                    l3_xia = [L1, K, D1]
                else:
                    l3_xia = [M_p, K, D1]
            else:
                l3_shang = [M, G]
                l3_x3 = [K_p, O]
                if L1[1] > M[1]:
                    l3_xia = [L1, K, D1]
                else:
                    l3_xia = [M_p, K]

            if u3 > B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I1[0] < u3 <= B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [P, P1]
            elif A1[0] < u3 <= I1[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                u3_xia = [P1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [P1]

        elif u4 <= B3[1] and min(C2[1], B[1]) < l4 <= max(C2[1], B[1]):

            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [O, I]
                if B[1] <= C2[1]:
                    l3_xia = [K,D,C2]
                else:
                    l3_xia = [K,D,D1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [H, G]
                l3_x3 = [I, O]
                if B[1] > C2[1]:
                    if L1[0] > M[0]:
                        l3_xia = [L1,L,K,D1]
                    else:
                        l3_xia = [M_p,L,K]
                else:
                    l3_xia = [L,L1,C2]
            elif I[0] < l3 <= H[0]:
                l3_shang = [H, G, L]
                l3_x3 = [K_p, O]
                if B[1] > C2[1]:
                    if L1[0] > M[0]:
                        l3_xia = [L1, K, D1]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]
            else:
                l3_shang = [M, G]
                l3_x3 = [K_p, O]
                if B[1] > C2[1]:
                    if L1[0] > M[0]:
                        l3_xia = [L1, K, D1]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]

            if u3 > min(B[0],B2[0]):
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I1[0] < u3 <= min(B[0],B2[0]):
                u3_shang = [A1]
                u3_x3 = [I1]
                if B[1] > C2[1]:
                    if Q[0] > N[0]:
                        u3_xia = [P,Q,N]
                    else:
                        u3_xia = [P,P1]
                else:
                    if P1[0] > D1[0]:
                        u3_xia = [P,P1,D1]
                    else:
                        u3_xia = [P,H2]
            elif A1[0] < u3 <= I1[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                if B[1] > C2[1]:
                    if Q[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    if P1[0] > D1[0]:
                        u3_xia = [P1, D1]
                    else:
                        u3_xia = [H2]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if B[1] > C2[1]:
                    if Q[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    if P1[0] > D1[0]:
                        u3_xia = [P1, D1]
                    else:
                        u3_xia = [H2]

        elif u4 <= B3[1] and C[1] < l4 <= min(C2[1], B[1]):

            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [O, I]
                l3_xia = [D,C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [H, G]
                l3_x3 = [I, O]
                l3_xia = [L,L1,C2]
            elif I[0] < l3 <= H[0]:
                l3_shang = [H, G, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [M, G]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if u3 > B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B,N,D1]
            elif I1[0] < u3 <= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                if P1[0] > N[0]:
                    u3_xia = [Q,N,D1,P]
                elif D[0] < P1[0] <= N[0]:
                    u3_xia = [P1,D1,P]
                else:
                    u3_xia = [P1,P]
            elif A1[0] < u3 <= I1[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q, N, D1]
                elif D[0] < P1[0] <= N[0]:
                    u3_xia = [P1, D1]
                else:
                    u3_xia = [P1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q, N, D1]
                elif D[0] < P1[0] <= N[0]:
                    u3_xia = [P1, D1]
                else:
                    u3_xia = [P1]

        elif u4 <= B3[1] and l4 <= C[1]:

            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [O, I]
                l3_xia = [D,C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [H, G]
                l3_x3 = [I, O]
                l3_xia = [L,L1,C2]
            elif I[0] < l3 <= H[0]:
                l3_shang = [H, G, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [M, G]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if u3 > B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B,C]
            elif I1[0] < u3 <= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                if Q[0] > C[0]:
                    u3_xia = [Q,C,P]
                else:
                    u3_xia = [H2,P]
            elif A1[0] < u3 <= I1[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                if Q[0] > C[0]:
                    u3_xia = [Q, C]
                else:
                    u3_xia = [H2]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if Q[0] > C[0]:
                    u3_xia = [Q, C]
                else:
                    u3_xia = [H2]

        elif B3[1] < u4 <= A[1] and 0 >= l4 > D[1]:
            if l3 <= J[0]:
                l3_shang = [B3,A1, H]
                l3_x3 = [O, I]
                l3_xia = [J, K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [B3,A1,H]
                l3_x3 = [I, O]
                l3_xia = [L, K, M_p]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3,A1,H,L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            elif H[0] < l3 < A1[0]:
                l3_shang = [B3,A1,M]
                l3_x3 = [K_p,O]
                l3_xia = [M_p,K]
            else:
                l3_shang = [B3,L3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I1[0] < u3 <= B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P, P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif B3[1] < u4 <= A[1] and D[1] >= l4 > max(C2[1],B[1]):

            if l3 <= D[0]:
                l3_shang = [B3,A1, H]
                l3_x3 = [O, I]
                l3_xia = [D,D1, K]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3,A1,H]
                l3_x3 = [I, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L, D1, M_p]
                else:
                    l3_xia = [M_p, L]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3,A1,H,L]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [D1, M_p]
                else:
                    l3_xia = [M_p]
            elif H[0] < l3 < A1[0]:
                l3_shang = [B3,A1,M]
                l3_x3 = [K_p,O]
                if M_p[0] <= D1[0]:
                    l3_xia = [D1, M_p]
                else:
                    l3_xia = [M_p]
            else:
                l3_shang = [B3,L3]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [D1, M_p]
                else:
                    l3_xia = [M_p]

            if u3 > B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I1[0] < u3 <= B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P, P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif B3[1] < u4 <= A[1] and max(C2[1], B[1]) >= l4 > min(C2[1], B[1]):
            if l3 <= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [O, I]
                if B[1] > C2[1]:
                    l3_xia = [D,D1]
                else:
                    l3_xia = [D, C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if B[1] > C2[1]:
                    if M_p[0] <= D1[0]:
                        l3_xia = [L1, D1, K,L]
                    else:
                        l3_xia = [M_p, K,L]
                else:
                    l3_xia = [L1, C2, L]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if B[1] > C2[1]:
                    if M_p[0] <= D1[0]:
                        l3_xia = [L1, D1, K]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]
            elif H[0] < l3 < A1[0]:
                l3_shang = [B3, A1, M]
                l3_x3 = [K_p, O]
                if B[1] > C2[1]:
                    if M_p[0] <= D1[0]:
                        l3_xia = [L1, D1, K]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                if B[1] > C2[1]:
                    if M_p[0] <= D1[0]:
                        l3_xia = [L1, D1, K]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]

            if u3 > min(B[0],B2[0]):
                u3_shang = []
                u3_x3 = [I1]
                if B[1] > C2[1]:
                    u3_xia = [B,N]
                else:
                    u3_xia = [B2,D1]
            elif I1[0] < u3 <= min(B[0],B2[0]):
                u3_shang = []
                u3_x3 = [I1]
                if B[1] > C2[1]:
                    if Q[0] > N[0]:
                        u3_xia = [Q, N,P]
                    else:
                        u3_xia = [P1,P]
                else:
                    if P1[0] > D1[0]:
                        u3_xia = [P1, D1,P]
                    else:
                        u3_xia = [H2,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if B[1] > C2[1]:
                    if Q[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    if P1[0] > D1[0]:
                        u3_xia = [P1, D1]
                    else:
                        u3_xia = [H2]

        elif B3[1] < u4 <= A[1] and min(C2[1], B[1]) >= l4 > C[1]:

            if l3 <= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [O, I]
                l3_xia = [D, C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [L1, C2, L]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            elif H[0] < l3 < A1[0]:
                l3_shang = [B3, A1, M]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B,D1,N]
            elif I1[0] < u3 <= B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if Q[0] > N[0]:
                    u3_xia = [Q,N,D1,P]
                elif P1[0] <= Q[0] <= N[0]:
                    u3_xia = [P1,D1,P]
                else:
                    u3_xia = [H2,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if Q[0] > N[0]:
                    u3_xia = [Q, N, D1]
                elif P1[0] <= Q[0] <= N[0]:
                    u3_xia = [P1, D1]
                else:
                    u3_xia = [H2]

        elif B3[1] < u4 <= A[1] and l4 <= C[1]:

            if l3 <= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [O, I]
                l3_xia = [D, C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [L1, C2, L]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            elif H[0] < l3 < A1[0]:
                l3_shang = [B3, A1, M]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B, C]
            elif I1[0] < u3 <= B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if Q[0] > C[0]:
                    u3_xia = [Q, C, P]
                else:
                    u3_xia = [H2, P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if Q[0] > C[0]:
                    u3_xia = [Q, C]
                else:
                    u3_xia = [H2]

        elif u4 > A[1] and 0 >= l4 > D[1]:
            if l3 <= J[0]:
                l3_shang = [B3,A]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L,M_p,K]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [B3, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I[0] < u3 <= B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P,P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P2,K]

        elif u4 > A[1] and D[1] >= l4 > max(C2[1],B[1]):
            if l3 <= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,D1, K]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [M_p, D1, L]
                else:
                    l3_xia = [M_p, L]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [M_p, D1]
                else:
                    l3_xia = [M_p]
            else:
                l3_shang = [B3, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [M_p, D1]
                else:
                    l3_xia = [M_p]

            if u3 > B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I[0] < u3 <= B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P, P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif u4 > A[1] and max(C2[1],B[1]) >= l4 > min(C2[1],B[1]):
            if l3 <= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if B[1] > C2[1]:
                    l3_xia = [D, D1]
                else:
                    l3_xia = [D, C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if B[1] > C2[1]:
                    if M_p[0] <= D1[0]:
                        l3_xia = [D1, L1, L, K]
                    else:
                        l3_xia = [M_p, L, K]
                else:
                    l3_xia = [L,L1,C2]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if B[1] > C2[1]:
                    if M_p[0] <= D1[0]:
                        l3_xia = [D1, L1, K]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L]
                l3_x3 = [K_p, O]
                if B[1] > C2[1]:
                    if M_p[0] <= D1[0]:
                        l3_xia = [D1, L1, K]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]

            if u3 > min(B[0],B2[0]):
                u3_shang = []
                u3_x3 = [I1]
                if B[1] > C2[1]:
                    u3_xia = [B,N]
                else:
                    u3_xia = [B2,D1]
            elif I[0] < u3 <= min(B[0],B2[0]):
                u3_shang = []
                u3_x3 = [I1]
                if B[1] > C2[1]:
                    if Q[0] > N[0]:
                        u3_xia = [Q, N, P]
                    else:
                        u3_xia = [P1, P]
                else:
                    if P1[0] > D1[0]:
                        u3_xia = [P1, D1, P]
                    else:
                        u3_xia = [H2, P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if B[1] > C2[1]:
                    if Q[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    if P1[0] > D1[0]:
                        u3_xia = [P1, D1]
                    else:
                        u3_xia = [H2]

        elif u4 > A[1] and min(C2[1],B[1]) >= l4 > C[1]:
            if l3 <= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L1, L, C2]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B,N,D1]
            elif I[0] < u3 <= B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if Q[0] > N[0]:
                    u3_xia = [N,Q,D1,P]
                elif D1[0] >= Q[0] >= N[0]:
                    u3_xia = [P1, D1, P]
                else:
                    u3_xia = [H2,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if Q[0] > N[0]:
                    u3_xia = [N, Q, D1]
                elif D1[0] >= Q[0] >= N[0]:
                    u3_xia = [P1, D1]
                else:
                    u3_xia = [H2]

        else:
            if l3 <= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L1, L, C2]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B,C]
            elif I[0] < u3 <= B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if Q[0] > C[0]:
                    u3_xia = [C,Q,P]
                else:
                    u3_xia = [H2,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if Q[0] > C[0]:
                    u3_xia = [C, Q]
                else:
                    u3_xia = [H2]

    # 对情况 02 左右镜像
    elif A[0] >= 0 and A[1] >= 0 and B[0] >= 0 and B[1] <= 0 and C[0] <= 0 and C[1] <= 0 and D[0] <= 0 and D[1] <= 0:
        A = [0.5 * (J3 + J2), 0.5 * (J3 - J2),0]
        B = [0.5 * (J4 + J2), 0.5 * (J4 - J2),0]
        C = [0.5 * (J4 + J1), 0.5 * (J4 - J1),0]
        D = [0.5 * (J3 + J1), 0.5 * (J3 - J1),0]
        H = [-u4 + J3, u4, 0]
        G = [0, u4, 0]
        F = [-u4 + J4, u4, 0]
        A1 = [-J2 + u4, u4, 0]
        A2 = [-J1 + u4, u4, 0]
        J = [-l4 + J3, l4, 0]
        K = [0, l4, 0]
        N = [-l4 + J4, l4, 0]
        P1 = [l3, l4, 0]
        D1 = [-J1 + l4, l4, 0]
        B2 = [-J2 + l4, l4, 0]
        L = [u3, -u3 + J3, 0]
        M = [u3, u4, 0]
        K_p = [u3, 0, 0]
        M_p = [u3, l4, 0]
        L1 = [u3, J1 + u3, 0]
        L2 = [u3, J4 - u3, 0]
        L3 = [u3, J2 + u3, 0]
        P = [l3, J2 + l3, 0]
        Q = [l3, J4 - l3, 0]
        P2 = [l3, u4, 0]
        R = [l3, 0, 0]
        H1 = [l3, -l3 + J3, 0]
        H2 = [l3, J1 + l3, 0]
        I = [-J3, 0, 0]
        B1 = [0, J3, 0]
        E = [-J4, 0, 0]
        C1 = [0, J4, 0]
        B3 = [0, J2, 0]
        I1 = [J2, 0, 0]
        C2 = [0, J1, 0]
        D2 = [J1, 0, 0]
        O = [0, 0, 0]

        if u4 <= B3[1] and l4 > D[1]:
            if u3 >= J[0]:
                l3_shang = [G,H]
                l3_x3 = [O,I]
                l3_xia = [J,K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [H,G]
                l3_x3 = [I,O]
                l3_xia = [L,K,M_p]
            elif I[0] > u3 >= H[0]:
                l3_shang = [H,G,L]
                l3_x3 = [K_p,O]
                l3_xia = [M_p,K]
            else:
                l3_shang = [M,G]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I1[0] > l3 >= B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [P,P1]
            elif A1[0] > l3 >= I1[0]:
                u3_shang = [A1,P]
                u3_x3 = [R]
                u3_xia = [P1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [P1]

        elif u4 <= B3[1] and max(C2[1],B[1]) < l4 <= D[1]:
            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [O, I]
                l3_xia = [K,D,D1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [H, G]
                l3_x3 = [I, O]
                if L1[1] > M[1]:
                    l3_xia = [L1,L,K,D1]
                else:
                    l3_xia = [M_p,L,K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [H, G, L]
                l3_x3 = [K_p, O]
                if L1[1] > M[1]:
                    l3_xia = [L1, K, D1]
                else:
                    l3_xia = [M_p, K, D1]
            else:
                l3_shang = [M, G]
                l3_x3 = [K_p, O]
                if L1[1] > M[1]:
                    l3_xia = [L1, K, D1]
                else:
                    l3_xia = [M_p, K]

            if l3 < B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I1[0] > l3 >= B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [P, P1]
            elif A1[0] > l3 >= I1[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                u3_xia = [P1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [P1]

        elif u4 <= B3[1] and min(C2[1], B[1]) < l4 <= max(C2[1], B[1]):

            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [O, I]
                if B[1] <= C2[1]:
                    l3_xia = [K,D,C2]
                else:
                    l3_xia = [K,D,D1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [H, G]
                l3_x3 = [I, O]
                if B[1] > C2[1]:
                    if L1[0] < M[0]:
                        l3_xia = [L1,L,K,D1]
                    else:
                        l3_xia = [M_p,L,K]
                else:
                    l3_xia = [L,L1,C2]
            elif I[0] > u3 >= H[0]:
                l3_shang = [H, G, L]
                l3_x3 = [K_p, O]
                if B[1] > C2[1]:
                    if L1[0] < M[0]:
                        l3_xia = [L1, K, D1]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]
            else:
                l3_shang = [M, G]
                l3_x3 = [K_p, O]
                if B[1] > C2[1]:
                    if L1[0] < M[0]:
                        l3_xia = [L1, K, D1]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]

            if l3 < max(B[0],B2[0]):
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I1[0] > l3 >= max(B[0],B2[0]):
                u3_shang = [A1]
                u3_x3 = [I1]
                if B[1] > C2[1]:
                    if Q[0] < N[0]:
                        u3_xia = [P,Q,N]
                    else:
                        u3_xia = [P,P1]
                else:
                    if P1[0] < D1[0]:
                        u3_xia = [P,P1,D1]
                    else:
                        u3_xia = [P,H2]
            elif A1[0] > l3 >= I1[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                if B[1] > C2[1]:
                    if Q[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    if P1[0] < D1[0]:
                        u3_xia = [P1, D1]
                    else:
                        u3_xia = [H2]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if B[1] > C2[1]:
                    if Q[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    if P1[0] < D1[0]:
                        u3_xia = [P1, D1]
                    else:
                        u3_xia = [H2]

        elif u4 <= B3[1] and C[1] < l4 <= min(C2[1], B[1]):

            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [O, I]
                l3_xia = [D,C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [H, G]
                l3_x3 = [I, O]
                l3_xia = [L,L1,C2]
            elif I[0] > u3 >= H[0]:
                l3_shang = [H, G, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [M, G]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if l3 < B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B,N,D1]
            elif I1[0] > l3 >= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                if P1[0] < N[0]:
                    u3_xia = [Q,N,D1,P]
                elif D[0] > P1[0] >= N[0]:
                    u3_xia = [P1,D1,P]
                else:
                    u3_xia = [P1,P]
            elif A1[0] > l3 >= I1[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q, N, D1]
                elif D[0] > P1[0] >= N[0]:
                    u3_xia = [P1, D1]
                else:
                    u3_xia = [P1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q, N, D1]
                elif D[0] > P1[0] >= N[0]:
                    u3_xia = [P1, D1]
                else:
                    u3_xia = [P1]

        elif u4 <= B3[1] and l4 <= C[1]:

            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [O, I]
                l3_xia = [D,C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [H, G]
                l3_x3 = [I, O]
                l3_xia = [L,L1,C2]
            elif I[0] > u3 >= H[0]:
                l3_shang = [H, G, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [M, G]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if l3 < B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B,C]
            elif I1[0] > l3 >= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                if Q[0] < C[0]:
                    u3_xia = [Q,C,P]
                else:
                    u3_xia = [H2,P]
            elif A1[0] > l3 >= I1[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                if Q[0] < C[0]:
                    u3_xia = [Q, C]
                else:
                    u3_xia = [H2]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if Q[0] < C[0]:
                    u3_xia = [Q, C]
                else:
                    u3_xia = [H2]

        elif B3[1] < u4 <= A[1] and 0 >= l4 > D[1]:
            if u3 >= J[0]:
                l3_shang = [B3,A1, H]
                l3_x3 = [O, I]
                l3_xia = [J, K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [B3,A1,H]
                l3_x3 = [I, O]
                l3_xia = [L, K, M_p]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3,A1,H,L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            elif H[0] > u3 > A1[0]:
                l3_shang = [B3,A1,M]
                l3_x3 = [K_p,O]
                l3_xia = [M_p,K]
            else:
                l3_shang = [B3,L3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I1[0] > l3 >= B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P, P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif B3[1] < u4 <= A[1] and D[1] >= l4 > max(C2[1],B[1]):

            if u3 >= D[0]:
                l3_shang = [B3,A1, H]
                l3_x3 = [O, I]
                l3_xia = [D,D1, K]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3,A1,H]
                l3_x3 = [I, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L, D1, M_p]
                else:
                    l3_xia = [M_p, L]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3,A1,H,L]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [D1, M_p]
                else:
                    l3_xia = [M_p]
            elif H[0] > u3 > A1[0]:
                l3_shang = [B3,A1,M]
                l3_x3 = [K_p,O]
                if M_p[0] >= D1[0]:
                    l3_xia = [D1, M_p]
                else:
                    l3_xia = [M_p]
            else:
                l3_shang = [B3,L3]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [D1, M_p]
                else:
                    l3_xia = [M_p]

            if l3 < B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I1[0] > l3 >= B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P, P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif B3[1] < u4 <= A[1] and max(C2[1], B[1]) >= l4 > min(C2[1], B[1]):
            if u3 >= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [O, I]
                if B[1] > C2[1]:
                    l3_xia = [D,D1]
                else:
                    l3_xia = [D, C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if B[1] > C2[1]:
                    if M_p[0] >= D1[0]:
                        l3_xia = [L1, D1, K,L]
                    else:
                        l3_xia = [M_p, K,L]
                else:
                    l3_xia = [L1, C2, L]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if B[1] > C2[1]:
                    if M_p[0] >= D1[0]:
                        l3_xia = [L1, D1, K]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]
            elif H[0] > u3 > A1[0]:
                l3_shang = [B3, A1, M]
                l3_x3 = [K_p, O]
                if B[1] > C2[1]:
                    if M_p[0] >= D1[0]:
                        l3_xia = [L1, D1, K]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                if B[1] > C2[1]:
                    if M_p[0] >= D1[0]:
                        l3_xia = [L1, D1, K]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]

            if l3 < max(B[0],B2[0]):
                u3_shang = []
                u3_x3 = [I1]
                if B[1] > C2[1]:
                    u3_xia = [B,N]
                else:
                    u3_xia = [B2,D1]
            elif I1[0] > l3 >= max(B[0],B2[0]):
                u3_shang = []
                u3_x3 = [I1]
                if B[1] > C2[1]:
                    if Q[0] < N[0]:
                        u3_xia = [Q, N,P]
                    else:
                        u3_xia = [P1,P]
                else:
                    if P1[0] < D1[0]:
                        u3_xia = [P1, D1,P]
                    else:
                        u3_xia = [H2,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if B[1] > C2[1]:
                    if Q[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    if P1[0] < D1[0]:
                        u3_xia = [P1, D1]
                    else:
                        u3_xia = [H2]

        elif B3[1] < u4 <= A[1] and min(C2[1], B[1]) >= l4 > C[1]:

            if u3 >= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [O, I]
                l3_xia = [D, C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [L1, C2, L]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            elif H[0] > u3 > A1[0]:
                l3_shang = [B3, A1, M]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B,D1,N]
            elif I1[0] > l3 >= B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if Q[0] < N[0]:
                    u3_xia = [Q,N,D1,P]
                elif P1[0] >= Q[0] >= N[0]:
                    u3_xia = [P1,D1,P]
                else:
                    u3_xia = [H2,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if Q[0] < N[0]:
                    u3_xia = [Q, N, D1]
                elif P1[0] >= Q[0] >= N[0]:
                    u3_xia = [P1, D1]
                else:
                    u3_xia = [H2]

        elif B3[1] < u4 <= A[1] and l4 <= C[1]:

            if u3 >= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [O, I]
                l3_xia = [D, C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [L1, C2, L]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            elif H[0] > u3 > A1[0]:
                l3_shang = [B3, A1, M]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B, C]
            elif I1[0] > l3 >= B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if Q[0] < C[0]:
                    u3_xia = [Q, C, P]
                else:
                    u3_xia = [H2, P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if Q[0] < C[0]:
                    u3_xia = [Q, C]
                else:
                    u3_xia = [H2]

        elif u4 > A[1] and 0 >= l4 > D[1]:
            if u3 >= J[0]:
                l3_shang = [B3,A]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L,M_p,K]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [B3, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I[0] > l3 >= B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P,P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P2,K]

        elif u4 > A[1] and D[1] >= l4 > max(C2[1],B[1]):
            if u3 >= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,D1, K]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [M_p, D1, L]
                else:
                    l3_xia = [M_p, L]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [M_p, D1]
                else:
                    l3_xia = [M_p]
            else:
                l3_shang = [B3, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [M_p, D1]
                else:
                    l3_xia = [M_p]

            if l3 < B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I[0] > l3 >= B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P, P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif u4 > A[1] and max(C2[1],B[1]) >= l4 > min(C2[1],B[1]):
            if u3 >= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if B[1] > C2[1]:
                    l3_xia = [D, D1]
                else:
                    l3_xia = [D, C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if B[1] > C2[1]:
                    if M_p[0] >= D1[0]:
                        l3_xia = [D1, L1, L, K]
                    else:
                        l3_xia = [M_p, L, K]
                else:
                    l3_xia = [L,L1,C2]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if B[1] > C2[1]:
                    if M_p[0] >= D1[0]:
                        l3_xia = [D1, L1, K]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L]
                l3_x3 = [K_p, O]
                if B[1] > C2[1]:
                    if M_p[0] >= D1[0]:
                        l3_xia = [D1, L1, K]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]

            if l3 < max(B[0],B2[0]):
                u3_shang = []
                u3_x3 = [I1]
                if B[1] > C2[1]:
                    u3_xia = [B,N]
                else:
                    u3_xia = [B2,D1]
            elif I[0] > l3 >= max(B[0],B2[0]):
                u3_shang = []
                u3_x3 = [I1]
                if B[1] > C2[1]:
                    if Q[0] < N[0]:
                        u3_xia = [Q, N, P]
                    else:
                        u3_xia = [P1, P]
                else:
                    if P1[0] < D1[0]:
                        u3_xia = [P1, D1, P]
                    else:
                        u3_xia = [H2, P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if B[1] > C2[1]:
                    if Q[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    if P1[0] < D1[0]:
                        u3_xia = [P1, D1]
                    else:
                        u3_xia = [H2]

        elif u4 > A[1] and min(C2[1],B[1]) >= l4 > C[1]:
            if u3 >= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L1, L, C2]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B,N,D1]
            elif I[0] > l3 >= B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if Q[0] < N[0]:
                    u3_xia = [N,Q,D1,P]
                elif D1[0] <= Q[0] <= N[0]:
                    u3_xia = [P1, D1, P]
                else:
                    u3_xia = [H2,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if Q[0] < N[0]:
                    u3_xia = [N, Q, D1]
                elif D1[0] <= Q[0] <= N[0]:
                    u3_xia = [P1, D1]
                else:
                    u3_xia = [H2]

        else:
            if u3 >= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L1, L, C2]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B,C]
            elif I[0] > l3 >= B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if Q[0] < C[0]:
                    u3_xia = [C,Q,P]
                else:
                    u3_xia = [H2,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if Q[0] < C[0]:
                    u3_xia = [C, Q]
                else:
                    u3_xia = [H2]

    # 对情况 02 上下镜像
    elif A[0] >= 0 and A[1] >= 0 and B[0] >= 0 and B[1] >= 0 and C[0] <= 0 and C[1] <= 0 and D[0] <= 0 and D[1] >= 0:
        A = [0.5 * (J4 + J1), 0.5 * (J4 - J1),0]
        B = [0.5 * (J3 + J1), 0.5 * (J3 - J1),0]
        C = [0.5 * (J3 + J2), 0.5 * (J3 - J2),0]
        D = [0.5 * (J4 + J2), 0.5 * (J4 - J2),0]
        H = [-l4 + J4, l4, 0]
        G = [0, l4, 0]
        F = [-l4 + J3, l4, 0]
        A1 = [-J1 + l4, l4, 0]
        A2 = [-J2 + l4, l4, 0]
        J = [-u4 + J4, u4, 0]
        K = [0, u4, 0]
        N = [-u4 + J3, u4, 0]
        P1 = [u3, u4, 0]
        D1 = [-J2 + u4, u4, 0]
        B2 = [-J1 + u4, u4, 0]
        L = [l3, -l3 + J4, 0]
        M = [l3, l4, 0]
        K_p = [l3, 0, 0]
        M_p = [l3, u4, 0]
        L1 = [l3, J2 + l3, 0]
        L2 = [l3, J3 - l3, 0]
        L3 = [l3, J1 + l3, 0]
        P = [u3, J1 + u3, 0]
        Q = [u3, J3 - u3, 0]
        P2 = [u3, l4, 0]
        R = [u3, 0, 0]
        H1 = [u3, -u3 + J4, 0]
        H2 = [u3, J2 + u3, 0]
        I = [-J4, 0, 0]
        B1 = [0, J4, 0]
        E = [-J3, 0, 0]
        C1 = [0, J3, 0]
        B3 = [0, J1, 0]
        I1 = [J1, 0, 0]
        C2 = [0, J2, 0]
        D2 = [J2, 0, 0]
        O = [0, 0, 0]
        sig1 = 1
        if l4 >= B3[1] and u4 < D[1]:
            if l3 <= J[0]:
                l3_shang = [G,H]
                l3_x3 = [O,I]
                l3_xia = [J,K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [H,G]
                l3_x3 = [I,O]
                l3_xia = [L,K,M_p]
            elif I[0] < l3 <= H[0]:
                l3_shang = [H,G,L]
                l3_x3 = [K_p,O]
                l3_xia = [M_p,K]
            else:
                l3_shang = [M,G]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I1[0] < u3 <= B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [P,P1]
            elif A1[0] < u3 <= I1[0]:
                u3_shang = [A1,P]
                u3_x3 = [R]
                u3_xia = [P1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [P1]

        elif l4 >= B3[1] and min(C2[1],B[1]) > u4 >= D[1]:
            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [O, I]
                l3_xia = [K,D,D1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [H, G]
                l3_x3 = [I, O]
                if L1[1] < M[1]:
                    l3_xia = [L1,L,K,D1]
                else:
                    l3_xia = [M_p,L,K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [H, G, L]
                l3_x3 = [K_p, O]
                if L1[1] < M[1]:
                    l3_xia = [L1, K, D1]
                else:
                    l3_xia = [M_p, K, D1]
            else:
                l3_shang = [M, G]
                l3_x3 = [K_p, O]
                if L1[1] < M[1]:
                    l3_xia = [L1, K, D1]
                else:
                    l3_xia = [M_p, K]

            if u3 > B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I1[0] < u3 <= B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [P, P1]
            elif A1[0] < u3 <= I1[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                u3_xia = [P1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [P1]

        elif l4 >= B3[1] and max(C2[1], B[1]) > u4 >= min(C2[1], B[1]):

            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [O, I]
                if B[1] >= C2[1]:
                    l3_xia = [K,D,C2]
                else:
                    l3_xia = [K,D,D1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [H, G]
                l3_x3 = [I, O]
                if B[1] < C2[1]:
                    if L1[0] > M[0]:
                        l3_xia = [L1,L,K,D1]
                    else:
                        l3_xia = [M_p,L,K]
                else:
                    l3_xia = [L,L1,C2]
            elif I[0] < l3 <= H[0]:
                l3_shang = [H, G, L]
                l3_x3 = [K_p, O]
                if B[1] < C2[1]:
                    if L1[0] > M[0]:
                        l3_xia = [L1, K, D1]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]
            else:
                l3_shang = [M, G]
                l3_x3 = [K_p, O]
                if B[1] < C2[1]:
                    if L1[0] > M[0]:
                        l3_xia = [L1, K, D1]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]

            if u3 > min(B[0],B2[0]):
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I1[0] < u3 <= min(B[0],B2[0]):
                u3_shang = [A1]
                u3_x3 = [I1]
                if B[1] < C2[1]:
                    if Q[0] > N[0]:
                        u3_xia = [P,Q,N]
                    else:
                        u3_xia = [P,P1]
                else:
                    if P1[0] > D1[0]:
                        u3_xia = [P,P1,D1]
                    else:
                        u3_xia = [P,H2]
            elif A1[0] < u3 <= I1[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                if B[1] < C2[1]:
                    if Q[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    if P1[0] > D1[0]:
                        u3_xia = [P1, D1]
                    else:
                        u3_xia = [H2]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if B[1] < C2[1]:
                    if Q[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    if P1[0] > D1[0]:
                        u3_xia = [P1, D1]
                    else:
                        u3_xia = [H2]

        elif l4 >= B3[1] and C[1] > u4 >= max(C2[1], B[1]):

            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [O, I]
                l3_xia = [D,C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [H, G]
                l3_x3 = [I, O]
                l3_xia = [L,L1,C2]
            elif I[0] < l3 <= H[0]:
                l3_shang = [H, G, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [M, G]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if u3 > B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B,N,D1]
            elif I1[0] < u3 <= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                if P1[0] > N[0]:
                    u3_xia = [Q,N,D1,P]
                elif D[0] < P1[0] <= N[0]:
                    u3_xia = [P1,D1,P]
                else:
                    u3_xia = [P1,P]
            elif A1[0] < u3 <= I1[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q, N, D1]
                elif D[0] < P1[0] <= N[0]:
                    u3_xia = [P1, D1]
                else:
                    u3_xia = [P1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q, N, D1]
                elif D[0] < P1[0] <= N[0]:
                    u3_xia = [P1, D1]
                else:
                    u3_xia = [P1]

        elif l4 >= B3[1] and u4 >= C[1]:

            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [O, I]
                l3_xia = [D,C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [H, G]
                l3_x3 = [I, O]
                l3_xia = [L,L1,C2]
            elif I[0] < l3 <= H[0]:
                l3_shang = [H, G, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [M, G]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if u3 > B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B,C]
            elif I1[0] < u3 <= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                if Q[0] > C[0]:
                    u3_xia = [Q,C,P]
                else:
                    u3_xia = [H2,P]
            elif A1[0] < u3 <= I1[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                if Q[0] > C[0]:
                    u3_xia = [Q, C]
                else:
                    u3_xia = [H2]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if Q[0] > C[0]:
                    u3_xia = [Q, C]
                else:
                    u3_xia = [H2]

        elif B3[1] > l4 >= A[1] and 0 <= u4 < D[1]:
            if l3 <= J[0]:
                l3_shang = [B3,A1, H]
                l3_x3 = [O, I]
                l3_xia = [J, K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [B3,A1,H]
                l3_x3 = [I, O]
                l3_xia = [L, K, M_p]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3,A1,H,L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            elif H[0] < l3 < A1[0]:
                l3_shang = [B3,A1,M]
                l3_x3 = [K_p,O]
                l3_xia = [M_p,K]
            else:
                l3_shang = [B3,L3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I1[0] < u3 <= B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P, P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif B3[1] > l4 >= A[1] and D[1] <= u4 < min(C2[1],B[1]):

            if l3 <= D[0]:
                l3_shang = [B3,A1, H]
                l3_x3 = [O, I]
                l3_xia = [D,D1, K]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3,A1,H]
                l3_x3 = [I, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L, D1, M_p]
                else:
                    l3_xia = [M_p, L]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3,A1,H,L]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [D1, M_p]
                else:
                    l3_xia = [M_p]
            elif H[0] < l3 < A1[0]:
                l3_shang = [B3,A1,M]
                l3_x3 = [K_p,O]
                if M_p[0] <= D1[0]:
                    l3_xia = [D1, M_p]
                else:
                    l3_xia = [M_p]
            else:
                l3_shang = [B3,L3]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [D1, M_p]
                else:
                    l3_xia = [M_p]

            if u3 > B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I1[0] < u3 <= B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P, P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif B3[1] > l4 >= A[1] and min(C2[1], B[1]) <= u4 < max(C2[1], B[1]):
            if l3 <= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [O, I]
                if B[1] < C2[1]:
                    l3_xia = [D,D1]
                else:
                    l3_xia = [D, C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if B[1] < C2[1]:
                    if M_p[0] <= D1[0]:
                        l3_xia = [L1, D1, K,L]
                    else:
                        l3_xia = [M_p, K,L]
                else:
                    l3_xia = [L1, C2, L]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if B[1] < C2[1]:
                    if M_p[0] <= D1[0]:
                        l3_xia = [L1, D1, K]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]
            elif H[0] < l3 < A1[0]:
                l3_shang = [B3, A1, M]
                l3_x3 = [K_p, O]
                if B[1] < C2[1]:
                    if M_p[0] <= D1[0]:
                        l3_xia = [L1, D1, K]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                if B[1] < C2[1]:
                    if M_p[0] <= D1[0]:
                        l3_xia = [L1, D1, K]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]

            if u3 > min(B[0],B2[0]):
                u3_shang = []
                u3_x3 = [I1]
                if B[1] < C2[1]:
                    u3_xia = [B,N]
                else:
                    u3_xia = [B2,D1]
            elif I1[0] < u3 <= min(B[0],B2[0]):
                u3_shang = []
                u3_x3 = [I1]
                if B[1] < C2[1]:
                    if Q[0] > N[0]:
                        u3_xia = [Q, N,P]
                    else:
                        u3_xia = [P1,P]
                else:
                    if P1[0] > D1[0]:
                        u3_xia = [P1, D1,P]
                    else:
                        u3_xia = [H2,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if B[1] < C2[1]:
                    if Q[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    if P1[0] > D1[0]:
                        u3_xia = [P1, D1]
                    else:
                        u3_xia = [H2]

        elif B3[1] > l4 >= A[1] and max(C2[1], B[1]) <= u4 < C[1]:

            if l3 <= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [O, I]
                l3_xia = [D, C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [L1, C2, L]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            elif H[0] < l3 < A1[0]:
                l3_shang = [B3, A1, M]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B,D1,N]
            elif I1[0] < u3 <= B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if Q[0] > N[0]:
                    u3_xia = [Q,N,D1,P]
                elif P1[0] <= Q[0] <= N[0]:
                    u3_xia = [P1,D1,P]
                else:
                    u3_xia = [H2,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if Q[0] > N[0]:
                    u3_xia = [Q, N, D1]
                elif P1[0] <= Q[0] <= N[0]:
                    u3_xia = [P1, D1]
                else:
                    u3_xia = [H2]

        elif B3[1] > l4 >= A[1] and u4 >= C[1]:

            if l3 <= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [O, I]
                l3_xia = [D, C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [L1, C2, L]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            elif H[0] < l3 < A1[0]:
                l3_shang = [B3, A1, M]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B, C]
            elif I1[0] < u3 <= B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if Q[0] > C[0]:
                    u3_xia = [Q, C, P]
                else:
                    u3_xia = [H2, P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if Q[0] > C[0]:
                    u3_xia = [Q, C]
                else:
                    u3_xia = [H2]

        elif l4 < A[1] and 0 <= u4 < D[1]:
            if l3 <= J[0]:
                l3_shang = [B3,A]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L,M_p,K]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [B3, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I[0] < u3 <= B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P,P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P2,K]

        elif l4 < A[1] and D[1] <= u4 < min(C2[1],B[1]):
            if l3 <= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,D1, K]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [M_p, D1, L]
                else:
                    l3_xia = [M_p, L]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [M_p, D1]
                else:
                    l3_xia = [M_p]
            else:
                l3_shang = [B3, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [M_p, D1]
                else:
                    l3_xia = [M_p]

            if u3 > B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I[0] < u3 <= B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P, P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif l4 < A[1] and min(C2[1],B[1]) <= u4 < max(C2[1],B[1]):
            if l3 <= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if B[1] < C2[1]:
                    l3_xia = [D, D1]
                else:
                    l3_xia = [D, C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if B[1] < C2[1]:
                    if M_p[0] <= D1[0]:
                        l3_xia = [D1, L1, L, K]
                    else:
                        l3_xia = [M_p, L, K]
                else:
                    l3_xia = [L,L1,C2]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if B[1] < C2[1]:
                    if M_p[0] <= D1[0]:
                        l3_xia = [D1, L1, K]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L]
                l3_x3 = [K_p, O]
                if B[1] < C2[1]:
                    if M_p[0] <= D1[0]:
                        l3_xia = [D1, L1, K]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]

            if u3 > min(B[0],B2[0]):
                u3_shang = []
                u3_x3 = [I1]
                if B[1] < C2[1]:
                    u3_xia = [B,N]
                else:
                    u3_xia = [B2,D1]
            elif I[0] < u3 <= min(B[0],B2[0]):
                u3_shang = []
                u3_x3 = [I1]
                if B[1] < C2[1]:
                    if Q[0] > N[0]:
                        u3_xia = [Q, N, P]
                    else:
                        u3_xia = [P1, P]
                else:
                    if P1[0] > D1[0]:
                        u3_xia = [P1, D1, P]
                    else:
                        u3_xia = [H2, P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if B[1] < C2[1]:
                    if Q[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    if P1[0] > D1[0]:
                        u3_xia = [P1, D1]
                    else:
                        u3_xia = [H2]

        elif l4 < A[1] and max(C2[1],B[1]) <= u4 < C[1]:
            if l3 <= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L1, L, C2]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B,N,D1]
            elif I[0] < u3 <= B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if Q[0] > N[0]:
                    u3_xia = [N,Q,D1,P]
                elif D1[0] >= Q[0] >= N[0]:
                    u3_xia = [P1, D1, P]
                else:
                    u3_xia = [H2,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if Q[0] > N[0]:
                    u3_xia = [N, Q, D1]
                elif D1[0] >= Q[0] >= N[0]:
                    u3_xia = [P1, D1]
                else:
                    u3_xia = [H2]

        else:
            if l3 <= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L1, L, C2]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B,C]
            elif I[0] < u3 <= B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if Q[0] > C[0]:
                    u3_xia = [C,Q,P]
                else:
                    u3_xia = [H2,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if Q[0] > C[0]:
                    u3_xia = [C, Q]
                else:
                    u3_xia = [H2]

    # 对情况 02 中心对称镜像（在左右镜像基础上进行上下镜像）
    elif A[0] <= 0 and A[1] >= 0 and B[0] >= 0 and B[1] >= 0 and C[0] >= 0 and C[1] <= 0 and D[0] <= 0 and D[1] >= 0:
        sig1 = 1
        A = [0.5 * (J3 + J1), 0.5 * (J3 - J1),0]
        B = [0.5 * (J4 + J2), 0.5 * (J4 - J2),0]
        C = [0.5 * (J4 + J1), 0.5 * (J4 - J1),0]
        D = [0.5 * (J3 + J2), 0.5 * (J3 - J2),0]
        H = [l4 - J1, l4, 0]
        G = [0, l4, 0]
        F = [l4 - J2, l4, 0]
        A1 = [J4 - l4, l4, 0]
        A2 = [J3 - l4, l4, 0]
        J = [u4 - J1, u4, 0]
        K = [0, u4, 0]
        N = [u4 - J2, u4, 0]
        P1 = [l3, u4, 0]
        D1 = [J3 - u4, u4, 0]
        B2 = [J4 - u4, u4, 0]
        L = [u3, u3 + J1, 0]
        M = [u3, l4, 0]
        K_p = [u3, 0, 0]
        M_p = [u3, u4, 0]
        L1 = [u3, J3 - u3, 0]
        L2 = [u3, J2 + u3, 0]
        L3 = [u3, J4 - u3, 0]
        P = [l3, J4 - l3, 0]
        Q = [l3, J2 + l3, 0]
        P2 = [l3, l4, 0]
        R = [l3, 0, 0]
        H1 = [l3, l3 + J1, 0]
        H2 = [l3, J3 - l3, 0]
        I = [-J1, 0, 0]
        B1 = [0, J1, 0]
        E = [-J2, 0, 0]
        C1 = [0, J2, 0]
        B3 = [0, J4, 0]
        I1 = [J4, 0, 0]
        C2 = [0, J3, 0]
        D2 = [J3, 0, 0]
        O = [0, 0, 0]

        if l4 >= B3[1] and u4 < D[1]:
            if u3 >= J[0]:
                l3_shang = [G,H]
                l3_x3 = [O,I]
                l3_xia = [J,K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [H,G]
                l3_x3 = [I,O]
                l3_xia = [L,K,M_p]
            elif I[0] > u3 >= H[0]:
                l3_shang = [H,G,L]
                l3_x3 = [K_p,O]
                l3_xia = [M_p,K]
            else:
                l3_shang = [M,G]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I1[0] > l3 >= B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [P,P1]
            elif A1[0] > l3 >= I1[0]:
                u3_shang = [A1,P]
                u3_x3 = [R]
                u3_xia = [P1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [P1]

        elif l4 >= B3[1] and min(C2[1],B[1]) > u4 >= D[1]:
            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [O, I]
                l3_xia = [K,D,D1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [H, G]
                l3_x3 = [I, O]
                if L1[1] < M[1]:
                    l3_xia = [L1,L,K,D1]
                else:
                    l3_xia = [M_p,L,K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [H, G, L]
                l3_x3 = [K_p, O]
                if L1[1] < M[1]:
                    l3_xia = [L1, K, D1]
                else:
                    l3_xia = [M_p, K, D1]
            else:
                l3_shang = [M, G]
                l3_x3 = [K_p, O]
                if L1[1] < M[1]:
                    l3_xia = [L1, K, D1]
                else:
                    l3_xia = [M_p, K]

            if l3 < B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I1[0] > l3 >= B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [P, P1]
            elif A1[0] > l3 >= I1[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                u3_xia = [P1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [P1]

        elif l4 >= B3[1] and max(C2[1], B[1]) > u4 >= min(C2[1], B[1]):

            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [O, I]
                if B[1] >= C2[1]:
                    l3_xia = [K,D,C2]
                else:
                    l3_xia = [K,D,D1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [H, G]
                l3_x3 = [I, O]
                if B[1] < C2[1]:
                    if L1[0] < M[0]:
                        l3_xia = [L1,L,K,D1]
                    else:
                        l3_xia = [M_p,L,K]
                else:
                    l3_xia = [L,L1,C2]
            elif I[0] > u3 >= H[0]:
                l3_shang = [H, G, L]
                l3_x3 = [K_p, O]
                if B[1] < C2[1]:
                    if L1[0] < M[0]:
                        l3_xia = [L1, K, D1]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]
            else:
                l3_shang = [M, G]
                l3_x3 = [K_p, O]
                if B[1] < C2[1]:
                    if L1[0] < M[0]:
                        l3_xia = [L1, K, D1]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]

            if l3 < max(B[0],B2[0]):
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I1[0] > l3 >= max(B[0],B2[0]):
                u3_shang = [A1]
                u3_x3 = [I1]
                if B[1] < C2[1]:
                    if Q[0] < N[0]:
                        u3_xia = [P,Q,N]
                    else:
                        u3_xia = [P,P1]
                else:
                    if P1[0] < D1[0]:
                        u3_xia = [P,P1,D1]
                    else:
                        u3_xia = [P,H2]
            elif A1[0] > l3 >= I1[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                if B[1] < C2[1]:
                    if Q[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    if P1[0] < D1[0]:
                        u3_xia = [P1, D1]
                    else:
                        u3_xia = [H2]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if B[1] < C2[1]:
                    if Q[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    if P1[0] < D1[0]:
                        u3_xia = [P1, D1]
                    else:
                        u3_xia = [H2]

        elif l4 >= B3[1] and C[1] > u4 >= max(C2[1], B[1]):

            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [O, I]
                l3_xia = [D,C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [H, G]
                l3_x3 = [I, O]
                l3_xia = [L,L1,C2]
            elif I[0] > u3 >= H[0]:
                l3_shang = [H, G, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [M, G]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if l3 < B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B,N,D1]
            elif I1[0] > l3 >= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                if P1[0] < N[0]:
                    u3_xia = [Q,N,D1,P]
                elif D[0] > P1[0] >= N[0]:
                    u3_xia = [P1,D1,P]
                else:
                    u3_xia = [P1,P]
            elif A1[0] > l3 >= I1[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q, N, D1]
                elif D[0] > P1[0] >= N[0]:
                    u3_xia = [P1, D1]
                else:
                    u3_xia = [P1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q, N, D1]
                elif D[0] > P1[0] >= N[0]:
                    u3_xia = [P1, D1]
                else:
                    u3_xia = [P1]

        elif l4 >= B3[1] and u4 >= C[1]:

            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [O, I]
                l3_xia = [D,C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [H, G]
                l3_x3 = [I, O]
                l3_xia = [L,L1,C2]
            elif I[0] > u3 >= H[0]:
                l3_shang = [H, G, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [M, G]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if l3 < B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B,C]
            elif I1[0] > l3 >= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                if Q[0] < C[0]:
                    u3_xia = [Q,C,P]
                else:
                    u3_xia = [H2,P]
            elif A1[0] > l3 >= I1[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                if Q[0] < C[0]:
                    u3_xia = [Q, C]
                else:
                    u3_xia = [H2]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if Q[0] < C[0]:
                    u3_xia = [Q, C]
                else:
                    u3_xia = [H2]

        elif B3[1] > l4 >= A[1] and 0 <= u4 < D[1]:
            if u3 >= J[0]:
                l3_shang = [B3,A1, H]
                l3_x3 = [O, I]
                l3_xia = [J, K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [B3,A1,H]
                l3_x3 = [I, O]
                l3_xia = [L, K, M_p]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3,A1,H,L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            elif H[0] > u3 > A1[0]:
                l3_shang = [B3,A1,M]
                l3_x3 = [K_p,O]
                l3_xia = [M_p,K]
            else:
                l3_shang = [B3,L3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I1[0] > l3 >= B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P, P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif B3[1] > l4 >= A[1] and D[1] <= u4 < min(C2[1],B[1]):

            if u3 >= D[0]:
                l3_shang = [B3,A1, H]
                l3_x3 = [O, I]
                l3_xia = [D,D1, K]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3,A1,H]
                l3_x3 = [I, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L, D1, M_p]
                else:
                    l3_xia = [M_p, L]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3,A1,H,L]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [D1, M_p]
                else:
                    l3_xia = [M_p]
            elif H[0] > u3 > A1[0]:
                l3_shang = [B3,A1,M]
                l3_x3 = [K_p,O]
                if M_p[0] >= D1[0]:
                    l3_xia = [D1, M_p]
                else:
                    l3_xia = [M_p]
            else:
                l3_shang = [B3,L3]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [D1, M_p]
                else:
                    l3_xia = [M_p]

            if l3 < B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I1[0] > l3 >= B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P, P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif B3[1] > l4 >= A[1] and min(C2[1], B[1]) <= u4 < max(C2[1], B[1]):
            if u3 >= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [O, I]
                if B[1] < C2[1]:
                    l3_xia = [D,D1]
                else:
                    l3_xia = [D, C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if B[1] < C2[1]:
                    if M_p[0] >= D1[0]:
                        l3_xia = [L1, D1, K,L]
                    else:
                        l3_xia = [M_p, K,L]
                else:
                    l3_xia = [L1, C2, L]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if B[1] < C2[1]:
                    if M_p[0] >= D1[0]:
                        l3_xia = [L1, D1, K]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]
            elif H[0] > u3 > A1[0]:
                l3_shang = [B3, A1, M]
                l3_x3 = [K_p, O]
                if B[1] < C2[1]:
                    if M_p[0] >= D1[0]:
                        l3_xia = [L1, D1, K]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                if B[1] < C2[1]:
                    if M_p[0] >= D1[0]:
                        l3_xia = [L1, D1, K]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]

            if l3 < max(B[0],B2[0]):
                u3_shang = []
                u3_x3 = [I1]
                if B[1] < C2[1]:
                    u3_xia = [B,N]
                else:
                    u3_xia = [B2,D1]
            elif I1[0] > l3 >= max(B[0],B2[0]):
                u3_shang = []
                u3_x3 = [I1]
                if B[1] < C2[1]:
                    if Q[0] < N[0]:
                        u3_xia = [Q, N,P]
                    else:
                        u3_xia = [P1,P]
                else:
                    if P1[0] < D1[0]:
                        u3_xia = [P1, D1,P]
                    else:
                        u3_xia = [H2,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if B[1] < C2[1]:
                    if Q[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    if P1[0] < D1[0]:
                        u3_xia = [P1, D1]
                    else:
                        u3_xia = [H2]

        elif B3[1] > l4 >= A[1] and max(C2[1], B[1]) <= u4 < C[1]:

            if u3 >= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [O, I]
                l3_xia = [D, C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [L1, C2, L]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            elif H[0] > u3 > A1[0]:
                l3_shang = [B3, A1, M]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B,D1,N]
            elif I1[0] > l3 >= B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if Q[0] < N[0]:
                    u3_xia = [Q,N,D1,P]
                elif P1[0] >= Q[0] >= N[0]:
                    u3_xia = [P1,D1,P]
                else:
                    u3_xia = [H2,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if Q[0] < N[0]:
                    u3_xia = [Q, N, D1]
                elif P1[0] >= Q[0] >= N[0]:
                    u3_xia = [P1, D1]
                else:
                    u3_xia = [H2]

        elif B3[1] > l4 >= A[1] and u4 >= C[1]:

            if u3 >= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [O, I]
                l3_xia = [D, C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [L1, C2, L]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            elif H[0] > u3 > A1[0]:
                l3_shang = [B3, A1, M]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B, C]
            elif I1[0] > l3 >= B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if Q[0] < C[0]:
                    u3_xia = [Q, C, P]
                else:
                    u3_xia = [H2, P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if Q[0] < C[0]:
                    u3_xia = [Q, C]
                else:
                    u3_xia = [H2]

        elif l4 < A[1] and 0 <= u4 < D[1]:
            if u3 >= J[0]:
                l3_shang = [B3,A]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L,M_p,K]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [B3, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I[0] > l3 >= B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P,P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P2,K]

        elif l4 < A[1] and D[1] <= u4 < min(C2[1],B[1]):
            if u3 >= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,D1, K]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [M_p, D1, L]
                else:
                    l3_xia = [M_p, L]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [M_p, D1]
                else:
                    l3_xia = [M_p]
            else:
                l3_shang = [B3, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [M_p, D1]
                else:
                    l3_xia = [M_p]

            if l3 < B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I[0] > l3 >= B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P, P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif l4 < A[1] and min(C2[1],B[1]) <= u4 < max(C2[1],B[1]):
            if u3 >= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if B[1] < C2[1]:
                    l3_xia = [D, D1]
                else:
                    l3_xia = [D, C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if B[1] < C2[1]:
                    if M_p[0] >= D1[0]:
                        l3_xia = [D1, L1, L, K]
                    else:
                        l3_xia = [M_p, L, K]
                else:
                    l3_xia = [L,L1,C2]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if B[1] < C2[1]:
                    if M_p[0] >= D1[0]:
                        l3_xia = [D1, L1, K]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L]
                l3_x3 = [K_p, O]
                if B[1] < C2[1]:
                    if M_p[0] >= D1[0]:
                        l3_xia = [D1, L1, K]
                    else:
                        l3_xia = [M_p, K]
                else:
                    l3_xia = [L1, C2]

            if l3 < max(B[0],B2[0]):
                u3_shang = []
                u3_x3 = [I1]
                if B[1] < C2[1]:
                    u3_xia = [B,N]
                else:
                    u3_xia = [B2,D1]
            elif I[0] > l3 >= max(B[0],B2[0]):
                u3_shang = []
                u3_x3 = [I1]
                if B[1] < C2[1]:
                    if Q[0] < N[0]:
                        u3_xia = [Q, N, P]
                    else:
                        u3_xia = [P1, P]
                else:
                    if P1[0] < D1[0]:
                        u3_xia = [P1, D1, P]
                    else:
                        u3_xia = [H2, P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if B[1] < C2[1]:
                    if Q[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    if P1[0] < D1[0]:
                        u3_xia = [P1, D1]
                    else:
                        u3_xia = [H2]

        elif l4 < A[1] and max(C2[1],B[1]) <= u4 < C[1]:
            if u3 >= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L1, L, C2]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B,N,D1]
            elif I[0] > l3 >= B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if Q[0] < N[0]:
                    u3_xia = [N,Q,D1,P]
                elif D1[0] <= Q[0] <= N[0]:
                    u3_xia = [P1, D1, P]
                else:
                    u3_xia = [H2,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if Q[0] < N[0]:
                    u3_xia = [N, Q, D1]
                elif D1[0] <= Q[0] <= N[0]:
                    u3_xia = [P1, D1]
                else:
                    u3_xia = [H2]

        else:
            if u3 >= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L1, L, C2]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [B3, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B,C]
            elif I[0] > l3 >= B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if Q[0] < C[0]:
                    u3_xia = [C,Q,P]
                else:
                    u3_xia = [H2,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if Q[0] < C[0]:
                    u3_xia = [C, Q]
                else:
                    u3_xia = [H2]

    # 情况 03
    elif A[0] <= 0 and A[1] >= 0 and B[0] >= 0 and B[1] >= 0 and C[0] <= 0 and C[1] <= 0 and D[0] <= 0 and D[1] <= 0:

        if B[1] >= u4 > 0 >= l4 > max(D[1], C1[1]):
            if l3 <= J[0]:
                l3_shang = [G,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L,M_p, K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] < u3 <= F[0]:
                u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = [N]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q,N]
                else:
                    u3_xia = [P1]

        elif B[1] >= u4 and max(D[1], C1[1]) >= l4 > min(D[1], C1[1]):
            if l3 <= max(J[0],D[0]):
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    l3_xia = [D,D1, K]
                else:
                    l3_xia = [J,N,C1]
            elif max(J[0],D[0]) < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1,D1,L,K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if u3 > F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                if D[1] > C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            elif E[0] < u3 <= F[0]:
                u3_shang = [P2, Q]
                u3_x3 = [E]
                if D[1] > C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if D[1] > C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif B[1] >= u4 and min(D[1], C1[1]) >= l4 > C[1]:
            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1,D1,N,L,C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p,N,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if u3 > F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] < u3 <= F[0]:
                u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif B[1] >= u4 and l4 <= C[1]:

            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1,C,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2]

            if u3 > F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] < u3 <= F[0]:
                u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif B3[1] >= u4 > B[1] and 0 >= l4 > max(D[1], C1[1]):
            if l3 <= J[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > F[0]:
                u3_shang = [A1,B]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] < u3 <= F[0]:
                if P2[0] > A[0]:
                    u3_shang = [A1,P,Q]
                else:
                    u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = [N]
            elif 0 < u3 <= E[0]:
                if P2[0] > A[0]:
                    u3_shang = [A1, P]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif B3[1] >= u4 > B[1] and max(D[1], C1[1]) >= l4 > min(D[1], C1[1]):
            if l3 <= max(J[0],D[0]):
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    l3_xia = [D,D1, K]
                else:
                    l3_xia = [J,N,C1]
            elif max(J[0],D[0]) < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1,D1,L,K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if u3 > F[0]:
                u3_shang = [A1, B]
                u3_x3 = [E]
                if D[1] > C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            elif E[0] < u3 <= F[0]:
                if P2[0] > A[0]:
                    u3_shang = [A1, P, Q]
                else:
                    u3_shang = [P2, Q]
                u3_x3 = [E]
                if D[1] > C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            elif 0 < u3 <= E[0]:
                if P2[0] > A[0]:
                    u3_shang = [A1, P]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                if D[1] > C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif B3[1] >= u4 > B[1] and min(D[1], C1[1]) >= l4 > C[1]:

            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1,D1,N,L,C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p,N,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if u3 > F[0]:
                u3_shang = [A1, B]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] < u3 <= F[0]:
                if P2[0] > A[0]:
                    u3_shang = [A1, P, Q]
                else:
                    u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = []
            elif 0 < u3 <= E[0]:
                if P2[0] > A[0]:
                    u3_shang = [A1, P]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif B3[1] >= u4 > B[1] and l4 <= C[1]:

            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1,C,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2]

            if u3 > F[0]:
                u3_shang = [A1, B]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] < u3 <= F[0]:
                if P2[0] > A[0]:
                    u3_shang = [A1, P, Q]
                else:
                    u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = []
            elif 0 < u3 <= E[0]:
                if P2[0] > A[0]:
                    u3_shang = [A1, P]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif A[1] >= u4 > B3[1] and 0 >= l4 > max(D[1], C1[1]):

            if l3 <= J[0]:
                l3_shang = [B3,A1,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                if M[0] > A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M,A1,B3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > F[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [N]
            elif F[0] >= u3 > E[0]:
                u3_shang = [P,Q]
                u3_x3 = [E]
                u3_xia = [N]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q,N]
                else:
                    u3_xia = [P1]

        elif A[1] >= u4 > B3[1] and max(D[1], C1[1]) >= l4 > min(D[1], C1[1]):

            if l3 <= max(J[0],D[0]):
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    l3_xia = [D, D1, K]
                else:
                    l3_xia = [J, N, C1]
            elif max(J[0],D[0]) < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1, D1, L, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                if M[0] > A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if u3 > F[0]:
                u3_shang = [B]
                u3_x3 = [E]
                if D[1] > C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            elif F[0] >= u3 > E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                if D[1] > C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if D[1] > C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif A[1] >= u4 > B3[1] and min(D[1], C1[1]) >= l4 > C[1]:

            if l3 <= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, L, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                if M[0] > A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if u3 > B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = []
            elif B[0] >= u3 > E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        elif A[1] >= u4 > B3[1] and l4 <= C[1]:

            if l3 <= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2, C1]
            else:
                if M[0] > A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2]

            if u3 > B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = []
            elif B[0] >= u3 > E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        elif A[1] <= u4 and 0 >= l4 > max(D[1], C1[1]):

            if l3 <= J[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > F[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [N]
            elif F[0] >= u3 > E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                u3_xia = [N]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif A[1] <= u4 and max(D[1], C1[1]) >= l4 > min(D[1], C1[1]):

            if l3 <= max(J[0],D[0]):
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    l3_xia = [D, D1, K]
                else:
                    l3_xia = [J, N, C1]
            elif max(J[0],D[0]) < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1, D1, L, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                l3_shang = [L3, B3]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if u3 > F[0]:
                u3_shang = [B]
                u3_x3 = [E]
                if D[1] > C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            elif F[0] >= u3 > E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                if D[1] > C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if D[1] > C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif A[1] <= u4 and min(D[1], C1[1]) >= l4 > C[1]:

            if l3 <= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, L, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if u3 > B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = []
            elif B[0] >= u3 > E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        else:

            if l3 <= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2]

            if u3 > B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = []
            elif B[0] >= u3 > E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

    # 对情况 03 左右镜像
    elif A[0] >= 0 and A[1] >= 0 and B[0] >= 0 and B[1] <= 0 and C[0] >= 0 and C[1] <= 0 and D[0] <= 0 and D[1] >= 0:
        A = [0.5 * (J3 + J2), 0.5 * (J3 - J2),0]
        B = [0.5 * (J4 + J2), 0.5 * (J4 - J2),0]
        C = [0.5 * (J4 + J1), 0.5 * (J4 - J1),0]
        D = [0.5 * (J3 + J1), 0.5 * (J3 - J1),0]
        H = [-u4 + J3, u4, 0]
        G = [0, u4, 0]
        F = [-u4 + J4, u4, 0]
        A1 = [-J2 + u4, u4, 0]
        A2 = [-J1 + u4, u4, 0]
        J = [-l4 + J3, l4, 0]
        K = [0, l4, 0]
        N = [-l4 + J4, l4, 0]
        P1 = [l3, l4, 0]
        D1 = [-J1 + l4, l4, 0]
        B2 = [-J2 + l4, l4, 0]
        L = [u3, -u3 + J3, 0]
        M = [u3, u4, 0]
        K_p = [u3, 0, 0]
        M_p = [u3, l4, 0]
        L1 = [u3, J1 + u3, 0]
        L2 = [u3, J4 - u3, 0]
        L3 = [u3, J2 + u3, 0]
        P = [l3, J2 + l3, 0]
        Q = [l3, J4 - l3, 0]
        P2 = [l3, u4, 0]
        R = [l3, 0, 0]
        H1 = [l3, -l3 + J3, 0]
        H2 = [l3, J1 + l3, 0]
        I = [-J3, 0, 0]
        B1 = [0, J3, 0]
        E = [-J4, 0, 0]
        C1 = [0, J4, 0]
        B3 = [0, J2, 0]
        I1 = [J2, 0, 0]
        C2 = [0, J1, 0]
        D2 = [J1, 0, 0]
        O = [0, 0, 0]

        if B[1] >= u4 > 0 >= l4 > max(D[1], C1[1]):
            if u3 >= J[0]:
                l3_shang = [G,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L,M_p, K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] > l3 >= F[0]:
                u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = [N]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q,N]
                else:
                    u3_xia = [P1]

        elif B[1] >= u4 and max(D[1], C1[1]) >= l4 > min(D[1], C1[1]):
            if u3 >= min(J[0],D[0]):
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    l3_xia = [D,D1, K]
                else:
                    l3_xia = [J,N,C1]
            elif min(J[0],D[0]) > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1,D1,L,K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if l3 < F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                if D[1] > C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            elif E[0] > l3 >= F[0]:
                u3_shang = [P2, Q]
                u3_x3 = [E]
                if D[1] > C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if D[1] > C1[1]:
                    if P1[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif B[1] >= u4 and min(D[1], C1[1]) >= l4 > C[1]:
            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1,D1,N,L,C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p,N,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if l3 < F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] > l3 >= F[0]:
                u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif B[1] >= u4 and l4 <= C[1]:

            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1,C,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2]

            if l3 < F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] > l3 >= F[0]:
                u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif B3[1] >= u4 > B[1] and 0 >= l4 > max(D[1], C1[1]):
            if u3 >= J[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < F[0]:
                u3_shang = [A1,B]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] > l3 >= F[0]:
                if P2[0] < A[0]:
                    u3_shang = [A1,P,Q]
                else:
                    u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = [N]
            elif 0 > l3 >= E[0]:
                if P2[0] < A[0]:
                    u3_shang = [A1, P]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif B3[1] >= u4 > B[1] and max(D[1], C1[1]) >= l4 > min(D[1], C1[1]):
            if u3 >= min(J[0],D[0]):
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    l3_xia = [D,D1, K]
                else:
                    l3_xia = [J,N,C1]
            elif min(J[0],D[0]) > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1,D1,L,K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if l3 < F[0]:
                u3_shang = [A1, B]
                u3_x3 = [E]
                if D[1] > C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            elif E[0] > l3 >= F[0]:
                if P2[0] < A[0]:
                    u3_shang = [A1, P, Q]
                else:
                    u3_shang = [P2, Q]
                u3_x3 = [E]
                if D[1] > C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            elif 0 > l3 >= E[0]:
                if P2[0] < A[0]:
                    u3_shang = [A1, P]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                if D[1] > C1[1]:
                    if P1[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif B3[1] >= u4 > B[1] and min(D[1], C1[1]) >= l4 > C[1]:

            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1,D1,N,L,C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p,N,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if l3 < F[0]:
                u3_shang = [A1, B]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] > l3 >= F[0]:
                if P2[0] < A[0]:
                    u3_shang = [A1, P, Q]
                else:
                    u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = []
            elif 0 > l3 >= E[0]:
                if P2[0] < A[0]:
                    u3_shang = [A1, P]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif B3[1] >= u4 > B[1] and l4 <= C[1]:

            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1,C,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2]

            if l3 < F[0]:
                u3_shang = [A1, B]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] > l3 >= F[0]:
                if P2[0] < A[0]:
                    u3_shang = [A1, P, Q]
                else:
                    u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = []
            elif 0 > l3 >= E[0]:
                if P2[0] < A[0]:
                    u3_shang = [A1, P]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif A[1] >= u4 > B3[1] and 0 >= l4 > max(D[1], C1[1]):

            if u3 >= J[0]:
                l3_shang = [B3,A1,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                if M[0] < A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M,A1,B3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < F[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [N]
            elif F[0] <= l3 < E[0]:
                u3_shang = [P,Q]
                u3_x3 = [E]
                u3_xia = [N]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q,N]
                else:
                    u3_xia = [P1]

        elif A[1] >= u4 > B3[1] and max(D[1], C1[1]) >= l4 > min(D[1], C1[1]):

            if u3 >= min(J[0],D[0]):
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    l3_xia = [D, D1, K]
                else:
                    l3_xia = [J, N, C1]
            elif min(J[0],D[0]) > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1, D1, L, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                if M[0] < A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if l3 < F[0]:
                u3_shang = [B]
                u3_x3 = [E]
                if D[1] > C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            elif F[0] <= l3 < E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                if D[1] > C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if D[1] > C1[1]:
                    if P1[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif A[1] >= u4 > B3[1] and min(D[1], C1[1]) >= l4 > C[1]:

            if u3 >= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, L, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                if M[0] < A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if l3 < B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = []
            elif B[0] <= l3 < E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        elif A[1] >= u4 > B3[1] and l4 <= C[1]:

            if u3 >= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2, C1]
            else:
                if M[0] < A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2]

            if l3 < B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = []
            elif B[0] <= l3 < E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        elif A[1] <= u4 and 0 >= l4 > max(D[1], C1[1]):

            if u3 >= J[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < F[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [N]
            elif F[0] <= l3 < E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                u3_xia = [N]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif A[1] <= u4 and max(D[1], C1[1]) >= l4 > min(D[1], C1[1]):

            if u3 >= min(J[0],D[0]):
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    l3_xia = [D, D1, K]
                else:
                    l3_xia = [J, N, C1]
            elif min(J[0],D[0]) > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1, D1, L, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                l3_shang = [L3, B3]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if l3 < F[0]:
                u3_shang = [B]
                u3_x3 = [E]
                if D[1] > C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            elif F[0] <= l3 < E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                if D[1] > C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if D[1] > C1[1]:
                    if P1[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif A[1] <= u4 and min(D[1], C1[1]) >= l4 > C[1]:

            if u3 >= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, L, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if l3 < B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = []
            elif B[0] <= l3 < E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        else:

            if u3 >= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2]

            if l3 < B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = []
            elif B[0] <= l3 < E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

    # 对情况 03 上下镜像
    elif A[0] <= 0 and A[1] >= 0 and B[0] >= 0 and B[1] <= 0 and C[0] <= 0 and C[1] <= 0 and D[0] <= 0 and D[1] >= 0:
        A = [0.5 * (J4 + J1), 0.5 * (J4 - J1),0]
        B = [0.5 * (J3 + J1), 0.5 * (J3 - J1),0]
        C = [0.5 * (J3 + J2), 0.5 * (J3 - J2),0]
        D = [0.5 * (J4 + J2), 0.5 * (J4 - J2),0]
        H = [-l4 + J4, l4, 0]
        G = [0, l4, 0]
        F = [-l4 + J3, l4, 0]
        A1 = [-J1 + l4, l4, 0]
        A2 = [-J2 + l4, l4, 0]
        J = [-u4 + J4, u4, 0]
        K = [0, u4, 0]
        N = [-u4 + J3, u4, 0]
        P1 = [u3, u4, 0]
        D1 = [-J2 + u4, u4, 0]
        B2 = [-J1 + u4, u4, 0]
        L = [l3, -l3 + J4, 0]
        M = [l3, l4, 0]
        K_p = [l3, 0, 0]
        M_p = [l3, u4, 0]
        L1 = [l3, J2 + l3, 0]
        L2 = [l3, J3 - l3, 0]
        L3 = [l3, J1 + l3, 0]
        P = [u3, J1 + u3, 0]
        Q = [u3, J3 - u3, 0]
        P2 = [u3, l4, 0]
        R = [u3, 0, 0]
        H1 = [u3, -u3 + J4, 0]
        H2 = [u3, J2 + u3, 0]
        I = [-J4, 0, 0]
        B1 = [0, J4, 0]
        E = [-J3, 0, 0]
        C1 = [0, J3, 0]
        B3 = [0, J1, 0]
        I1 = [J1, 0, 0]
        C2 = [0, J2, 0]
        D2 = [J2, 0, 0]
        O = [0, 0, 0]
        sig1 = 1
        if B[1] <= l4 < 0 <= u4 < min(D[1], C1[1]):
            if l3 <= J[0]:
                l3_shang = [G,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L,M_p, K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] < u3 <= F[0]:
                u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = [N]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q,N]
                else:
                    u3_xia = [P1]

        elif B[1] <= l4 and min(D[1], C1[1]) <= u4 < max(D[1], C1[1]):
            if l3 <= max(J[0],D[0]):
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    l3_xia = [D,D1, K]
                else:
                    l3_xia = [J,N,C1]
            elif max(J[0],D[0]) < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1,D1,L,K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if u3 > F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                if D[1] < C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            elif E[0] < u3 <= F[0]:
                u3_shang = [P2, Q]
                u3_x3 = [E]
                if D[1] < C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if D[1] < C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif B[1] <= l4 and max(D[1], C1[1]) <= u4 < C[1]:
            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1,D1,N,L,C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p,N,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if u3 > F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] < u3 <= F[0]:
                u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif B[1] <= l4 and u4 >= C[1]:

            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1,C,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2]

            if u3 > F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] < u3 <= F[0]:
                u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif B3[1] <= l4 < B[1] and 0 <= u4 < min(D[1], C1[1]):
            if l3 <= J[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > F[0]:
                u3_shang = [A1,B]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] < u3 <= F[0]:
                if P2[0] > A[0]:
                    u3_shang = [A1,P,Q]
                else:
                    u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = [N]
            elif 0 < u3 <= E[0]:
                if P2[0] > A[0]:
                    u3_shang = [A1, P]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif B3[1] <= l4 < B[1] and min(D[1], C1[1]) <= u4 < max(D[1], C1[1]):
            if l3 <= max(J[0],D[0]):
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    l3_xia = [D,D1, K]
                else:
                    l3_xia = [J,N,C1]
            elif max(J[0],D[0]) < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1,D1,L,K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if u3 > F[0]:
                u3_shang = [A1, B]
                u3_x3 = [E]
                if D[1] < C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            elif E[0] < u3 <= F[0]:
                if P2[0] > A[0]:
                    u3_shang = [A1, P, Q]
                else:
                    u3_shang = [P2, Q]
                u3_x3 = [E]
                if D[1] < C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            elif 0 < u3 <= E[0]:
                if P2[0] > A[0]:
                    u3_shang = [A1, P]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                if D[1] < C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif B3[1] <= l4 < B[1] and max(D[1], C1[1]) <= u4 < C[1]:

            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1,D1,N,L,C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p,N,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if u3 > F[0]:
                u3_shang = [A1, B]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] < u3 <= F[0]:
                if P2[0] > A[0]:
                    u3_shang = [A1, P, Q]
                else:
                    u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = []
            elif 0 < u3 <= E[0]:
                if P2[0] > A[0]:
                    u3_shang = [A1, P]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif B3[1] <= l4 < B[1] and u4 >= C[1]:

            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1,C,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2]

            if u3 > F[0]:
                u3_shang = [A1, B]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] < u3 <= F[0]:
                if P2[0] > A[0]:
                    u3_shang = [A1, P, Q]
                else:
                    u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = []
            elif 0 < u3 <= E[0]:
                if P2[0] > A[0]:
                    u3_shang = [A1, P]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif A[1] <= l4 < B3[1] and 0 <= u4 < min(D[1], C1[1]):

            if l3 <= J[0]:
                l3_shang = [B3,A1,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                if M[0] > A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M,A1,B3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > F[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [N]
            elif F[0] >= u3 > E[0]:
                u3_shang = [P,Q]
                u3_x3 = [E]
                u3_xia = [N]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q,N]
                else:
                    u3_xia = [P1]

        elif A[1] <= l4 < B3[1] and min(D[1], C1[1]) <= u4 < max(D[1], C1[1]):

            if l3 <= max(J[0],D[0]):
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    l3_xia = [D, D1, K]
                else:
                    l3_xia = [J, N, C1]
            elif max(J[0],D[0]) < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1, D1, L, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                if M[0] > A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if u3 > F[0]:
                u3_shang = [B]
                u3_x3 = [E]
                if D[1] < C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            elif F[0] >= u3 > E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                if D[1] < C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if D[1] < C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif A[1] <= l4 < B3[1] and max(D[1], C1[1]) <= u4 < C[1]:

            if l3 <= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, L, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                if M[0] > A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if u3 > B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = []
            elif B[0] >= u3 > E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        elif A[1] <= l4 < B3[1] and u4 >= C[1]:

            if l3 <= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2, C1]
            else:
                if M[0] > A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2]

            if u3 > B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = []
            elif B[0] >= u3 > E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        elif A[1] >= l4 and 0 <= u4 < min(D[1], C1[1]):

            if l3 <= J[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > F[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [N]
            elif F[0] >= u3 > E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                u3_xia = [N]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif A[1] >= l4 and min(D[1], C1[1]) <= u4 < max(D[1], C1[1]):

            if l3 <= max(J[0],D[0]):
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    l3_xia = [D, D1, K]
                else:
                    l3_xia = [J, N, C1]
            elif max(J[0],D[0]) < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1, D1, L, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                l3_shang = [L3, B3]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if u3 > F[0]:
                u3_shang = [B]
                u3_x3 = [E]
                if D[1] < C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            elif F[0] >= u3 > E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                if D[1] < C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if D[1] < C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif A[1] >= l4 and max(D[1], C1[1]) <= u4 < C[1]:

            if l3 <= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, L, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if u3 > B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = []
            elif B[0] >= u3 > E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        else:

            if l3 <= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2]

            if u3 > B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = []
            elif B[0] >= u3 > E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

    # 对情况 03 中心对称镜像
    elif A[0] >= 0 and A[1] >= 0 and B[0] >= 0 and B[1] >= 0 and C[0] >= 0 and C[1] <= 0 and D[0] <= 0 and D[1] <= 0:
        sig1 = 1
        A = [0.5 * (J3 + J1), 0.5 * (J3 - J1),0]
        B = [0.5 * (J4 + J2), 0.5 * (J4 - J2),0]
        C = [0.5 * (J4 + J1), 0.5 * (J4 - J1),0]
        D = [0.5 * (J3 + J2), 0.5 * (J3 - J2),0]
        H = [l4 - J1, l4, 0]
        G = [0, l4, 0]
        F = [l4 - J2, l4, 0]
        A1 = [J4 - l4, l4, 0]
        A2 = [J3 - l4, l4, 0]
        J = [u4 - J1, u4, 0]
        K = [0, u4, 0]
        N = [u4 - J2, u4, 0]
        P1 = [l3, u4, 0]
        D1 = [J3 - u4, u4, 0]
        B2 = [J4 - u4, u4, 0]
        L = [u3, u3 + J1, 0]
        M = [u3, l4, 0]
        K_p = [u3, 0, 0]
        M_p = [u3, u4, 0]
        L1 = [u3, J3 - u3, 0]
        L2 = [u3, J2 + u3, 0]
        L3 = [u3, J4 - u3, 0]
        P = [l3, J4 - l3, 0]
        Q = [l3, J2 + l3, 0]
        P2 = [l3, l4, 0]
        R = [l3, 0, 0]
        H1 = [l3, l3 + J1, 0]
        H2 = [l3, J3 - l3, 0]
        I = [-J1, 0, 0]
        B1 = [0, J1, 0]
        E = [-J2, 0, 0]
        C1 = [0, J2, 0]
        B3 = [0, J4, 0]
        I1 = [J4, 0, 0]
        C2 = [0, J3, 0]
        D2 = [J3, 0, 0]
        O = [0, 0, 0]

        if B[1] <= l4 < 0 <= u4 < min(D[1], C1[1]):
            if u3 >= J[0]:
                l3_shang = [G,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L,M_p, K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] > l3 >= F[0]:
                u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = [N]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q,N]
                else:
                    u3_xia = [P1]

        elif B[1] <= l4 and min(D[1], C1[1]) <= u4 < max(D[1], C1[1]):
            if u3 >= min(J[0],D[0]):
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    l3_xia = [D,D1, K]
                else:
                    l3_xia = [J,N,C1]
            elif min(J[0],D[0]) > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1,D1,L,K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if l3 < F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                if D[1] < C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            elif E[0] > l3 >= F[0]:
                u3_shang = [P2, Q]
                u3_x3 = [E]
                if D[1] < C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if D[1] < C1[1]:
                    if P1[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif B[1] <= l4 and max(D[1], C1[1]) <= u4 < C[1]:
            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1,D1,N,L,C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p,N,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if l3 < F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] > l3 >= F[0]:
                u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif B[1] <= l4 and u4 >= C[1]:

            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1,C,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2]

            if l3 < F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] > l3 >= F[0]:
                u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif B3[1] <= l4 < B[1] and 0 <= u4 < min(D[1], C1[1]):
            if u3 >= J[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < F[0]:
                u3_shang = [A1,B]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] > l3 >= F[0]:
                if P2[0] < A[0]:
                    u3_shang = [A1,P,Q]
                else:
                    u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = [N]
            elif 0 > l3 >= E[0]:
                if P2[0] < A[0]:
                    u3_shang = [A1, P]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif B3[1] <= l4 < B[1] and min(D[1], C1[1]) <= u4 < max(D[1], C1[1]):
            if u3 >= min(J[0],D[0]):
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    l3_xia = [D,D1, K]
                else:
                    l3_xia = [J,N,C1]
            elif min(J[0],D[0]) > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1,D1,L,K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if l3 < F[0]:
                u3_shang = [A1, B]
                u3_x3 = [E]
                if D[1] < C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            elif E[0] > l3 >= F[0]:
                if P2[0] < A[0]:
                    u3_shang = [A1, P, Q]
                else:
                    u3_shang = [P2, Q]
                u3_x3 = [E]
                if D[1] < C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            elif 0 > l3 >= E[0]:
                if P2[0] < A[0]:
                    u3_shang = [A1, P]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                if D[1] < C1[1]:
                    if P1[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif B3[1] <= l4 < B[1] and max(D[1], C1[1]) <= u4 < C[1]:

            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1,D1,N,L,C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p,N,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if l3 < F[0]:
                u3_shang = [A1, B]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] > l3 >= F[0]:
                if P2[0] < A[0]:
                    u3_shang = [A1, P, Q]
                else:
                    u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = []
            elif 0 > l3 >= E[0]:
                if P2[0] < A[0]:
                    u3_shang = [A1, P]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif B3[1] <= l4 < B[1] and u4 >= C[1]:

            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1,C,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2]

            if l3 < F[0]:
                u3_shang = [A1, B]
                u3_x3 = [E]
                u3_xia = []
            elif E[0] > l3 >= F[0]:
                if P2[0] < A[0]:
                    u3_shang = [A1, P, Q]
                else:
                    u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                if P2[0] < A[0]:
                    u3_shang = [A1, P]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif A[1] <= l4 < B3[1] and 0 <= u4 < min(D[1], C1[1]):

            if u3 >= J[0]:
                l3_shang = [B3,A1,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                if M[0] < A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M,A1,B3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < F[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [N]
            elif F[0] <= l3 < E[0]:
                u3_shang = [P,Q]
                u3_x3 = [E]
                u3_xia = [N]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q,N]
                else:
                    u3_xia = [P1]

        elif A[1] <= l4 < B3[1] and min(D[1], C1[1]) <= u4 < max(D[1], C1[1]):

            if u3 >= min(J[0],D[0]):
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    l3_xia = [D, D1, K]
                else:
                    l3_xia = [J, N, C1]
            elif min(J[0],D[0]) > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1, D1, L, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                if M[0] < A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if l3 < F[0]:
                u3_shang = [B]
                u3_x3 = [E]
                if D[1] < C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            elif F[0] <= l3 < E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                if D[1] < C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if D[1] < C1[1]:
                    if P1[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif A[1] <= l4 < B3[1] and max(D[1], C1[1]) <= u4 < C[1]:

            if u3 >= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, L, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                if M[0] < A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if l3 < B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = []
            elif B[0] <= l3 < E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        elif A[1] <= l4 < B3[1] and u4 >= C[1]:

            if u3 >= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2, C1]
            else:
                if M[0] < A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2]

            if l3 < B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = []
            elif B[0] <= l3 < E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        elif A[1] >= l4 and 0 <= u4 < min(D[1], C1[1]):

            if u3 >= J[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < F[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [N]
            elif F[0] <= l3 < E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                u3_xia = [N]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif A[1] >= l4 and min(D[1], C1[1]) <= u4 < max(D[1], C1[1]):

            if u3 >= min(J[0],D[0]):
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    l3_xia = [D, D1, K]
                else:
                    l3_xia = [J, N, C1]
            elif min(J[0],D[0]) > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1, D1, L, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                l3_shang = [L3, B3]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if l3 < F[0]:
                u3_shang = [B]
                u3_x3 = [E]
                if D[1] < C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            elif F[0] <= l3 < E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                if D[1] < C1[1]:
                    u3_xia = [N]
                else:
                    u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if D[1] < C1[1]:
                    if P1[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif A[1] >= l4 and max(D[1], C1[1]) <= u4 < C[1]:

            if u3 >= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, L, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if l3 < B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = []
            elif B[0] <= l3 < E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        else:

            if u3 >= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2]

            if l3 < B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = []
            elif B[0] <= l3 < E[0]:
                u3_shang = [P, Q]
                u3_x3 = [E]
                u3_xia = []
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

    # 情况 04
    elif A[0] <= 0 and A[1] >= 0 and B[0] >= 0 and B[1] <= 0 and C[0] <= 0 and C[1] <= 0 and D[0] <= 0 and D[1] <= 0:

        if B3[1] >= u4 > 0 >= l4 > B[1]:
            if l3 <= J[0]:
                l3_shang = [G,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L,M_p, K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I[0] < u3 <= B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [P,P1]
            elif A1[0] < u3 <= I[0]:
                u3_shang = [A1,P]
                u3_x3 = [R]
                u3_xia = [P1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [P1]

        elif B3[1] >= u4 > 0 and B[1] >= l4 > max(C1[1],D[1]):
            if l3 <= J[0]:
                l3_shang = [G,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L,M_p, K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B,N]
            elif I[0] < u3 <= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                if P1[0] > N[0]:
                    u3_xia = [Q,N,P]
                else:
                    u3_xia = [P1,P]
            elif A1[0] < u3 <= I[0]:
                u3_shang = [A1,P]
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif B3[1] >= u4 > 0 and max(C1[1],D[1]) >= l4 > min(C1[1],D[1]):
            if l3 <= max(J[0], D[0]):
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    l3_xia = [D, D1, K]
                else:
                    l3_xia = [J, N, C1]
            elif max(J[0], D[0]) < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1, D1, L, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if u3 > B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                if D[1] > C1[1]:
                    u3_xia = [N,B]
                else:
                    u3_xia = [B]
            elif I[0] < u3 <= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                if D[1] > C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q,N,P]
                    else:
                        u3_xia = [P1,P]
                else:
                    u3_xia = [Q,P]
            elif A1[0] < u3 <= I[0]:
                u3_shang = [A1,P]
                u3_x3 = [R]
                if D[1] > C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if D[1] > C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif B3[1] >= u4 > 0 and min(C1[1], D[1]) >= l4 > C[1]:
            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1,D1,N,L,C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p,N,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if u3 > B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B]
            elif I[0] < u3 <= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [Q, P]
            elif A1[0] < u3 <= I[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                u3_xia = [Q]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif B3[1] >= u4 > 0 and l4 <= C[1]:

            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1,C,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2]

            if u3 > B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B]
            elif I[0] < u3 <= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [Q, P]
            elif A1[0] < u3 <= I[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                u3_xia = [Q]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif A[1] >= u4 > B3[1] and 0 >= l4 > B[1]:

            if l3 <= J[0]:
                l3_shang = [B3,A1,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                if M[0] > A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M,A1,B3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif B2[0] >= u3 > I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P,P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif A[1] >= u4 > B3[1] and B[1] >= l4 > max(C1[1],D[1]):

            if l3 <= J[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                if M[0] > A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B,N]
            elif B[0] >= u3 > I[0]:
                u3_shang = []
                u3_x3 = [I1]
                if P1[0] > N[0]:
                    u3_xia = [Q,N,P]
                else:
                    u3_xia = [P1,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif A[1] >= u4 > B3[1] and max(C1[1],D[1]) >= l4 > min(C1[1],D[1]):

            if l3 <= max(J[0],D[0]):
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    l3_xia = [D, D1, K]
                else:
                    l3_xia = [J, N, C1]
            elif max(J[0],D[0]) < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1, D1, L, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                if M[0] > A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if D[1] > C1[1]:
                    u3_xia = [N,B]
                else:
                    u3_xia = [B]
            elif B[0] >= u3 > I[0]:
                u3_shang = []
                u3_x3 = [I1]
                if D[1] > C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N, P]
                    else:
                        u3_xia = [P1, P]
                else:
                    u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if D[1] > C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif A[1] >= u4 > B3[1] and min(D[1], C1[1]) >= l4 > C[1]:

            if l3 <= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, L, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                if M[0] > A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B]
            elif B[0] >= u3 > I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        elif A[1] >= u4 > B3[1] and l4 <= C[1]:

            if l3 <= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2, C1]
            else:
                if M[0] > A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B]
            elif B[0] >= u3 > I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        elif u4 > A[1] and 0 >= l4 > B[1]:

            if l3 <= J[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif B2[0] >= u3 > I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P,P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif u4 > A[1] and B[1] >= l4 > max(C1[1],D[1]):

            if l3 <= J[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B,N]
            elif B[0] >= u3 > I[0]:
                u3_shang = []
                u3_x3 = [I1]
                if P1[0] > N[0]:
                    u3_xia = [Q,N,P]
                else:
                    u3_xia = [P1,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif u4 > A[1] and max(C1[1],D[1]) >= l4 > min(C1[1],D[1]):

            if l3 <= max(J[0],D[0]):
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    l3_xia = [D, D1, K]
                else:
                    l3_xia = [J, N, C1]
            elif max(J[0],D[0]) < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1, D1, L, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                l3_shang = [L3, B3]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if D[1] > C1[1]:
                    u3_xia = [N,B]
                else:
                    u3_xia = [B]
            elif B[0] >= u3 > I[0]:
                u3_shang = []
                u3_x3 = [I1]
                if D[1] > C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N, P]
                    else:
                        u3_xia = [P1, P]
                else:
                    u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if D[1] > C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif u4 > A[1] and min(D[1], C1[1]) >= l4 > C[1]:

            if l3 <= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, L, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B]
            elif B[0] >= u3 > I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        else:

            if l3 <= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B]
            elif B[0] >= u3 > I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

    # 对情况 04 左右镜像
    elif A[0] >= 0 and A[1] >= 0 and B[0] >= 0 and B[1] <= 0 and C[0] >= 0 and C[1] <= 0 and D[0] <= 0 and D[1] <= 0:
        A = [0.5 * (J3 + J2), 0.5 * (J3 - J2),0]
        B = [0.5 * (J4 + J2), 0.5 * (J4 - J2),0]
        C = [0.5 * (J4 + J1), 0.5 * (J4 - J1),0]
        D = [0.5 * (J3 + J1), 0.5 * (J3 - J1),0]
        H = [-u4 + J3, u4, 0]
        G = [0, u4, 0]
        F = [-u4 + J4, u4, 0]
        A1 = [-J2 + u4, u4, 0]
        A2 = [-J1 + u4, u4, 0]
        J = [-l4 + J3, l4, 0]
        K = [0, l4, 0]
        N = [-l4 + J4, l4, 0]
        P1 = [l3, l4, 0]
        D1 = [-J1 + l4, l4, 0]
        B2 = [-J2 + l4, l4, 0]
        L = [u3, -u3 + J3, 0]
        M = [u3, u4, 0]
        K_p = [u3, 0, 0]
        M_p = [u3, l4, 0]
        L1 = [u3, J1 + u3, 0]
        L2 = [u3, J4 - u3, 0]
        L3 = [u3, J2 + u3, 0]
        P = [l3, J2 + l3, 0]
        Q = [l3, J4 - l3, 0]
        P2 = [l3, u4, 0]
        R = [l3, 0, 0]
        H1 = [l3, -l3 + J3, 0]
        H2 = [l3, J1 + l3, 0]
        I = [-J3, 0, 0]
        B1 = [0, J3, 0]
        E = [-J4, 0, 0]
        C1 = [0, J4, 0]
        B3 = [0, J2, 0]
        I1 = [J2, 0, 0]
        C2 = [0, J1, 0]
        D2 = [J1, 0, 0]
        O = [0, 0, 0]

        if B3[1] >= u4 > 0 >= l4 > B[1]:
            if u3 >= J[0]:
                l3_shang = [G,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L,M_p, K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I[0] > l3 >= B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [P,P1]
            elif A1[0] > l3 >= I[0]:
                u3_shang = [A1,P]
                u3_x3 = [R]
                u3_xia = [P1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [P1]

        elif B3[1] >= u4 > 0 and B[1] >= l4 > max(C1[1],D[1]):
            if u3 >= J[0]:
                l3_shang = [G,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L,M_p, K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B,N]
            elif I[0] > l3 >= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                if P1[0] < N[0]:
                    u3_xia = [Q,N,P]
                else:
                    u3_xia = [P1,P]
            elif A1[0] > l3 >= I[0]:
                u3_shang = [A1,P]
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif B3[1] >= u4 > 0 and max(C1[1],D[1]) >= l4 > min(C1[1],D[1]):
            if u3 >= min(J[0], D[0]):
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    l3_xia = [D, D1, K]
                else:
                    l3_xia = [J, N, C1]
            elif min(J[0], D[0]) > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1, D1, L, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if u3 > B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                if D[1] > C1[1]:
                    u3_xia = [N,B]
                else:
                    u3_xia = [B]
            elif I[0] < u3 <= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                if D[1] > C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q,N,P]
                    else:
                        u3_xia = [P1,P]
                else:
                    u3_xia = [Q,P]
            elif A1[0] < u3 <= I[0]:
                u3_shang = [A1,P]
                u3_x3 = [R]
                if D[1] > C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if D[1] > C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif B3[1] >= u4 > 0 and min(C1[1], D[1]) >= l4 > C[1]:
            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1,D1,N,L,C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p,N,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if l3 < B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B]
            elif I[0] > l3 >= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [Q, P]
            elif A1[0] > l3 >= I[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                u3_xia = [Q]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif B3[1] >= u4 > 0 and l4 <= C[1]:

            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1,C,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2]

            if l3 < B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B]
            elif I[0] > l3 >= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [Q, P]
            elif A1[0] > l3 >= I[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                u3_xia = [Q]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif A[1] >= u4 > B3[1] and 0 >= l4 > B[1]:

            if u3 >= J[0]:
                l3_shang = [B3,A1,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                if M[0] < A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M,A1,B3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif B2[0] <= l3 < I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P,P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif A[1] >= u4 > B3[1] and B[1] >= l4 > max(C1[1],D[1]):

            if u3 >= J[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                if M[0] < A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B,N]
            elif B[0] <= l3 < I[0]:
                u3_shang = []
                u3_x3 = [I1]
                if P1[0] < N[0]:
                    u3_xia = [Q,N,P]
                else:
                    u3_xia = [P1,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif A[1] >= u4 > B3[1] and max(C1[1],D[1]) >= l4 > min(C1[1],D[1]):

            if u3 >= min(J[0],D[0]):
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    l3_xia = [D, D1, K]
                else:
                    l3_xia = [J, N, C1]
            elif min(J[0],D[0]) > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1, D1, L, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                if M[0] < A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if D[1] > C1[1]:
                    u3_xia = [N,B]
                else:
                    u3_xia = [B]
            elif B[0] <= l3 < I[0]:
                u3_shang = []
                u3_x3 = [I1]
                if D[1] > C1[1]:
                    if P1[0] < N[0]:
                        u3_xia = [Q, N, P]
                    else:
                        u3_xia = [P1, P]
                else:
                    u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if D[1] > C1[1]:
                    if P1[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif A[1] >= u4 > B3[1] and min(D[1], C1[1]) >= l4 > C[1]:

            if u3 >= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, L, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                if M[0] < A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B]
            elif B[0] <= l3 < I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        elif A[1] >= u4 > B3[1] and l4 <= C[1]:

            if u3 >= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2, C1]
            else:
                if M[0] < A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B]
            elif B[0] <= l3 < I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        elif u4 > A[1] and 0 >= l4 > B[1]:

            if u3 >= J[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif B2[0] <= l3 < I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P,P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif u4 > A[1] and B[1] >= l4 > max(C1[1],D[1]):

            if u3 >= J[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B,N]
            elif B[0] <= l3 < I[0]:
                u3_shang = []
                u3_x3 = [I1]
                if P1[0] < N[0]:
                    u3_xia = [Q,N,P]
                else:
                    u3_xia = [P1,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif u4 > A[1] and max(C1[1],D[1]) >= l4 > min(C1[1],D[1]):

            if u3 >= min(J[0],D[0]):
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    l3_xia = [D, D1, K]
                else:
                    l3_xia = [J, N, C1]
            elif min(J[0],D[0]) > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if D[1] > C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1, D1, L, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                l3_shang = [L3, B3]
                l3_x3 = [K_p, O]
                if D[1] > C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if D[1] > C1[1]:
                    u3_xia = [N,B]
                else:
                    u3_xia = [B]
            elif B[0] <= l3 < I[0]:
                u3_shang = []
                u3_x3 = [I1]
                if D[1] > C1[1]:
                    if P1[0] < N[0]:
                        u3_xia = [Q, N, P]
                    else:
                        u3_xia = [P1, P]
                else:
                    u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if D[1] > C1[1]:
                    if P1[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif u4 > A[1] and min(D[1], C1[1]) >= l4 > C[1]:

            if u3 >= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, L, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B]
            elif B[0] <= l3 < I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        else:

            if u3 >= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B]
            elif B[0] <= l3 < I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

    # 对情况 04 上下镜像
    elif A[0] <= 0 and A[1] >= 0 and B[0] >= 0 and B[1] >= 0 and C[0] <= 0 and C[1] <= 0 and D[0] <= 0 and D[1] >= 0:
        A = [0.5 * (J4 + J1), 0.5 * (J4 - J1),0]
        B = [0.5 * (J3 + J1), 0.5 * (J3 - J1),0]
        C = [0.5 * (J3 + J2), 0.5 * (J3 - J2),0]
        D = [0.5 * (J4 + J2), 0.5 * (J4 - J2),0]
        H = [-l4 + J4, l4, 0]
        G = [0, l4, 0]
        F = [-l4 + J3, l4, 0]
        A1 = [-J1 + l4, l4, 0]
        A2 = [-J2 + l4, l4, 0]
        J = [-u4 + J4, u4, 0]
        K = [0, u4, 0]
        N = [-u4 + J3, u4, 0]
        P1 = [u3, u4, 0]
        D1 = [-J2 + u4, u4, 0]
        B2 = [-J1 + u4, u4, 0]
        L = [l3, -l3 + J4, 0]
        M = [l3, l4, 0]
        K_p = [l3, 0, 0]
        M_p = [l3, u4, 0]
        L1 = [l3, J2 + l3, 0]
        L2 = [l3, J3 - l3, 0]
        L3 = [l3, J1 + l3, 0]
        P = [u3, J1 + u3, 0]
        Q = [u3, J3 - u3, 0]
        P2 = [u3, l4, 0]
        R = [u3, 0, 0]
        H1 = [u3, -u3 + J4, 0]
        H2 = [u3, J2 + u3, 0]
        I = [-J4, 0, 0]
        B1 = [0, J4, 0]
        E = [-J3, 0, 0]
        C1 = [0, J3, 0]
        B3 = [0, J1, 0]
        I1 = [J1, 0, 0]
        C2 = [0, J2, 0]
        D2 = [J2, 0, 0]
        O = [0, 0, 0]
        sig1 = 1

        if B3[1] <= l4 < 0 <= u4 < B[1]:
            if l3 <= J[0]:
                l3_shang = [G,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L,M_p, K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I[0] < u3 <= B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [P,P1]
            elif A1[0] < u3 <= I[0]:
                u3_shang = [A1,P]
                u3_x3 = [R]
                u3_xia = [P1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [P1]

        elif B3[1] <= l4 < 0 and B[1] <= u4 < min(C1[1],D[1]):
            if l3 <= J[0]:
                l3_shang = [G,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L,M_p, K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B,N]
            elif I[0] < u3 <= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                if P1[0] > N[0]:
                    u3_xia = [Q,N,P]
                else:
                    u3_xia = [P1,P]
            elif A1[0] < u3 <= I[0]:
                u3_shang = [A1,P]
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif B3[1] <= l4 < 0 and min(C1[1],D[1]) <= u4 < max(C1[1],D[1]):
            if l3 <= max(J[0], D[0]):
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    l3_xia = [D, D1, K]
                else:
                    l3_xia = [J, N, C1]
            elif max(J[0], D[0]) < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1, D1, L, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if u3 > B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                if D[1] < C1[1]:
                    u3_xia = [N,B]
                else:
                    u3_xia = [B]
            elif I[0] < u3 <= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                if D[1] < C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q,N,P]
                    else:
                        u3_xia = [P1,P]
                else:
                    u3_xia = [Q,P]
            elif A1[0] < u3 <= I[0]:
                u3_shang = [A1,P]
                u3_x3 = [R]
                if D[1] < C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if D[1] < C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif B3[1] <= l4 < 0 and max(C1[1], D[1]) <= u4 < C[1]:
            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1,D1,N,L,C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p,N,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if u3 > B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B]
            elif I[0] < u3 <= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [Q, P]
            elif A1[0] < u3 <= I[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                u3_xia = [Q]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif B3[1] <= l4 < 0 and u4 >= C[1]:

            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1,C,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2]

            if u3 > B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B]
            elif I[0] < u3 <= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [Q, P]
            elif A1[0] < u3 <= I[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                u3_xia = [Q]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif A[1] <= l4 < B3[1] and 0 <= u4 < B[1]:

            if l3 <= J[0]:
                l3_shang = [B3,A1,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                if M[0] > A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M,A1,B3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif B2[0] >= u3 > I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P,P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif A[1] <= l4 < B3[1] and B[1] <= u4 < min(C1[1],D[1]):

            if l3 <= J[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                if M[0] > A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B,N]
            elif B[0] >= u3 > I[0]:
                u3_shang = []
                u3_x3 = [I1]
                if P1[0] > N[0]:
                    u3_xia = [Q,N,P]
                else:
                    u3_xia = [P1,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif A[1] <= l4 > B3[1] and min(C1[1],D[1]) <= u4 < max(C1[1],D[1]):

            if l3 <= max(J[0],D[0]):
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    l3_xia = [D, D1, K]
                else:
                    l3_xia = [J, N, C1]
            elif max(J[0],D[0]) < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1, D1, L, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                if M[0] > A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if D[1] < C1[1]:
                    u3_xia = [N,B]
                else:
                    u3_xia = [B]
            elif B[0] >= u3 > I[0]:
                u3_shang = []
                u3_x3 = [I1]
                if D[1] < C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N, P]
                    else:
                        u3_xia = [P1, P]
                else:
                    u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if D[1] < C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif A[1] <= l4 < B3[1] and max(D[1], C1[1]) <= u4 < C[1]:

            if l3 <= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, L, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                if M[0] > A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B]
            elif B[0] >= u3 > I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        elif A[1] <= l4 < B3[1] and u4 >= C[1]:

            if l3 <= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2, C1]
            else:
                if M[0] > A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B]
            elif B[0] >= u3 > I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        elif l4 < A[1] and 0 <= u4 < B[1]:

            if l3 <= J[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif B2[0] >= u3 > I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P,P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif l4 < A[1] and B[1] <= u4 < min(C1[1],D[1]):

            if l3 <= J[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B,N]
            elif B[0] >= u3 > I[0]:
                u3_shang = []
                u3_x3 = [I1]
                if P1[0] > N[0]:
                    u3_xia = [Q,N,P]
                else:
                    u3_xia = [P1,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if P1[0] > N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif l4 < A[1] and min(C1[1],D[1]) <= u4 < max(C1[1],D[1]):

            if l3 <= max(J[0],D[0]):
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    l3_xia = [D, D1, K]
                else:
                    l3_xia = [J, N, C1]
            elif max(J[0],D[0]) < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1, D1, L, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                l3_shang = [L3, B3]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] > D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] > N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if D[1] < C1[1]:
                    u3_xia = [N,B]
                else:
                    u3_xia = [B]
            elif B[0] >= u3 > I[0]:
                u3_shang = []
                u3_x3 = [I1]
                if D[1] < C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N, P]
                    else:
                        u3_xia = [P1, P]
                else:
                    u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if D[1] < C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif l4 < A[1] and max(D[1], C1[1]) <= u4 < C[1]:

            if l3 <= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, L, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                if M_p[0] <= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B]
            elif B[0] >= u3 > I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        else:

            if l3 <= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] < l3 <= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                if M_p[0] <= C[0]:
                    l3_xia = [L1, C]
                elif D[0] <= M_p[0] <= N[0]:
                    l3_xia = [L2]

            if u3 > B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B]
            elif B[0] >= u3 > I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

    # 对情况 04 中心对称镜像
    elif A[0] >= 0 and A[1] >= 0 and B[0] >= 0 and B[1] >= 0 and C[0] >= 0 and C[1] <= 0 and D[0] <= 0 and D[1] >= 0:
        sig1 = 1
        A = [0.5 * (J3 + J1), 0.5 * (J3 - J1),0]
        B = [0.5 * (J4 + J2), 0.5 * (J4 - J2),0]
        C = [0.5 * (J4 + J1), 0.5 * (J4 - J1),0]
        D = [0.5 * (J3 + J2), 0.5 * (J3 - J2),0]
        H = [l4 - J1, l4, 0]
        G = [0, l4, 0]
        F = [l4 - J2, l4, 0]
        A1 = [J4 - l4, l4, 0]
        A2 = [J3 - l4, l4, 0]
        J = [u4 - J1, u4, 0]
        K = [0, u4, 0]
        N = [u4 - J2, u4, 0]
        P1 = [l3, u4, 0]
        D1 = [J3 - u4, u4, 0]
        B2 = [J4 - u4, u4, 0]
        L = [u3, u3 + J1, 0]
        M = [u3, l4, 0]
        K_p = [u3, 0, 0]
        M_p = [u3, u4, 0]
        L1 = [u3, J3 - u3, 0]
        L2 = [u3, J2 + u3, 0]
        L3 = [u3, J4 - u3, 0]
        P = [l3, J4 - l3, 0]
        Q = [l3, J2 + l3, 0]
        P2 = [l3, l4, 0]
        R = [l3, 0, 0]
        H1 = [l3, l3 + J1, 0]
        H2 = [l3, J3 - l3, 0]
        I = [-J1, 0, 0]
        B1 = [0, J1, 0]
        E = [-J2, 0, 0]
        C1 = [0, J2, 0]
        B3 = [0, J4, 0]
        I1 = [J4, 0, 0]
        C2 = [0, J3, 0]
        D2 = [J3, 0, 0]
        O = [0, 0, 0]

        if B3[1] <= l4 < 0 <= u4 < B[1]:
            if u3 >= J[0]:
                l3_shang = [G,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L,M_p, K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B2]
            elif I[0] > l3 >= B2[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [P,P1]
            elif A1[0] > l3 >= I[0]:
                u3_shang = [A1,P]
                u3_x3 = [R]
                u3_xia = [P1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [P1]

        elif B3[1] <= l4 < 0 and B[1] <= u4 < min(C1[1],D[1]):
            if u3 >= J[0]:
                l3_shang = [G,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L,M_p, K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B,N]
            elif I[0] > l3 >= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                if P1[0] < N[0]:
                    u3_xia = [Q,N,P]
                else:
                    u3_xia = [P1,P]
            elif A1[0] > l3 >= I[0]:
                u3_shang = [A1,P]
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif B3[1] <= l4 < 0 and min(C1[1],D[1]) <= u4 < max(C1[1],D[1]):
            if u3 >= min(J[0], D[0]):
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    l3_xia = [D, D1, K]
                else:
                    l3_xia = [J, N, C1]
            elif min(J[0], D[0]) > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1, D1, L, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if u3 > B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                if D[1] < C1[1]:
                    u3_xia = [N,B]
                else:
                    u3_xia = [B]
            elif I[0] < u3 <= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                if D[1] < C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q,N,P]
                    else:
                        u3_xia = [P1,P]
                else:
                    u3_xia = [Q,P]
            elif A1[0] < u3 <= I[0]:
                u3_shang = [A1,P]
                u3_x3 = [R]
                if D[1] < C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if D[1] < C1[1]:
                    if P1[0] > N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif B3[1] <= l4 < 0 and max(C1[1], D[1]) <= u4 < C[1]:
            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1,D1,N,L,C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p,N,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if l3 < B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B]
            elif I[0] > l3 >= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [Q, P]
            elif A1[0] > l3 >= I[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                u3_xia = [Q]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif B3[1] <= l4 < 0 and u4 >= C[1]:

            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1,C,L,C1]
                else:
                    l3_xia = [L2,L,C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2]

            if l3 < B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [B]
            elif I[0] > l3 >= B[0]:
                u3_shang = [A1]
                u3_x3 = [I1]
                u3_xia = [Q, P]
            elif A1[0] > l3 >= I[0]:
                u3_shang = [A1, P]
                u3_x3 = [R]
                u3_xia = [Q]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q]

        elif A[1] <= l4 < B3[1] and 0 <= u4 < B[1]:

            if u3 >= J[0]:
                l3_shang = [B3,A1,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                if M[0] < A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M,A1,B3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif B2[0] <= l3 < I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P,P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif A[1] <= l4 < B3[1] and B[1] <= u4 < min(C1[1],D[1]):

            if u3 >= J[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                if M[0] < A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B,N]
            elif B[0] <= l3 < I[0]:
                u3_shang = []
                u3_x3 = [I1]
                if P1[0] < N[0]:
                    u3_xia = [Q,N,P]
                else:
                    u3_xia = [P1,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif A[1] <= l4 < B3[1] and min(C1[1],D[1]) <= u4 < max(C1[1],D[1]):

            if u3 >= min(J[0],D[0]):
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    l3_xia = [D, D1, K]
                else:
                    l3_xia = [J, N, C1]
            elif min(J[0],D[0]) > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1, D1, L, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                if M[0] < A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if D[1] < C1[1]:
                    u3_xia = [N,B]
                else:
                    u3_xia = [B]
            elif B[0] <= l3 < I[0]:
                u3_shang = []
                u3_x3 = [I1]
                if D[1] < C1[1]:
                    if P1[0] < N[0]:
                        u3_xia = [Q, N, P]
                    else:
                        u3_xia = [P1, P]
                else:
                    u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if D[1] < C1[1]:
                    if P1[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif A[1] <= l4 < B3[1] and max(D[1], C1[1]) <= u4 < C[1]:

            if u3 >= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, L, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                if M[0] < A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B]
            elif B[0] <= l3 < I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        elif A[1] <= l4 < B3[1] and u4 >= C[1]:

            if u3 >= D[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A1, H]
                l3_x3 = [I, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3, A1, H, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2, C1]
            else:
                if M[0] < A1[0]:
                    l3_shang = [B3, L3]
                else:
                    l3_shang = [M, A1, B3]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B]
            elif B[0] <= l3 < I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        elif l4 < A[1] and 0 <= u4 < B[1]:

            if u3 >= J[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B2[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B2]
            elif B2[0] <= l3 < I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [P,P1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif l4 < A[1] and B[1] <= u4 < min(C1[1],D[1]):

            if u3 >= J[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B,N]
            elif B[0] <= l3 < I[0]:
                u3_shang = []
                u3_x3 = [I1]
                if P1[0] < N[0]:
                    u3_xia = [Q,N,P]
                else:
                    u3_xia = [P1,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if P1[0] < N[0]:
                    u3_xia = [Q, N]
                else:
                    u3_xia = [P1]

        elif l4 < A[1] and min(C1[1],D[1]) <= u4 < max(C1[1],D[1]):

            if u3 >= min(J[0],D[0]):
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    l3_xia = [D, D1, K]
                else:
                    l3_xia = [J, N, C1]
            elif min(J[0],D[0]) > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if D[1] < C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, L, K]
                    else:
                        l3_xia = [L1, D1, L, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, L, C1]
                    else:
                        l3_xia = [M_p, N, L, C1]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]
            else:
                l3_shang = [L3, B3]
                l3_x3 = [K_p, O]
                if D[1] < C1[1]:
                    if M_p[0] < D1[0]:
                        l3_xia = [M_p, K]
                    else:
                        l3_xia = [L1, D1, K]
                else:
                    if M_p[0] < N[0]:
                        l3_xia = [L2, C1]
                    else:
                        l3_xia = [M_p, N, C1]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                if D[1] < C1[1]:
                    u3_xia = [N,B]
                else:
                    u3_xia = [B]
            elif B[0] <= l3 < I[0]:
                u3_shang = []
                u3_x3 = [I1]
                if D[1] < C1[1]:
                    if P1[0] < N[0]:
                        u3_xia = [Q, N, P]
                    else:
                        u3_xia = [P1, P]
                else:
                    u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if D[1] < C1[1]:
                    if P1[0] < N[0]:
                        u3_xia = [Q, N]
                    else:
                        u3_xia = [P1]
                else:
                    u3_xia = [Q]

        elif l4 < A[1] and max(D[1], C1[1]) <= u4 < C[1]:

            if u3 >= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,D1,N,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, L, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N, C1]
                else:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                if M_p[0] >= D1[0]:
                    l3_xia = [L1, D1, N]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [M_p, N]
                else:
                    l3_xia = [L2]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B]
            elif B[0] <= l3 < I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

        else:

            if u3 >= D[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                l3_xia = [D,D1,C1]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3, A]
                l3_x3 = [I, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, L, C1]
                else:
                    l3_xia = [L2, L, C1]
            elif I[0] > u3 >= A[0]:
                l3_shang = [B3, A, L]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C, C1]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2, C1]
            else:
                l3_shang = [B3, L3]
                l3_x3 = [K_p, O]
                if M_p[0] >= C[0]:
                    l3_xia = [L1, C]
                elif D[0] >= M_p[0] >= N[0]:
                    l3_xia = [L2]

            if l3 < B[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [B]
            elif B[0] <= l3 < I[0]:
                u3_shang = []
                u3_x3 = [I1]
                u3_xia = [Q,P]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q]

    # 情况 05
    elif A[0] <= 0 and A[1] >= 0 and B[0] >= 0 and B[1] >= 0 and C[0] >= 0 and C[1] <= 0 and D[0] <= 0 and D[1] <= 0:

        if B[1] >= u4 > 0 >= l4 > D[1]:
            if l3 <= J[0]:
                l3_shang = [G,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L,M_p, K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] < u3 <= F[0]:
                u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = [N]
            elif N[0] < u3 <= E[0]:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q,N]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [P1]

        elif B[1] >= u4 and D[1] >= l4 > C2[1]:
            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,K]
            elif D[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L, L1, D1, K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,D1, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [L1,D1, K]

            if u3 > F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] < u3 <= F[0]:
                u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = [N]
            elif N[0] < u3 <= E[0]:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q, N]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [P1]

        elif B[1] >= u4 and C2[1] >= l4 > C[1]:
            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L, L1, C2]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]

            if u3 > F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = [N,D1]
            elif E[0] < u3 <= F[0]:
                u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = [N,D1]
            elif N[0] < u3 <= E[0]:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q, N,D1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] > D1[0]:
                    u3_xia = [P1,D1]
                else:
                    u3_xia = [H2]

        elif B[1] >= u4 and l4 <= C[1]:
            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L, L1, C2]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]

            if u3 > F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = [C]
            elif E[0] < u3 <= F[0]:
                u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = [C]
            elif N[0] < u3 <= E[0]:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q, C]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [H2]

        elif B3[1] >= u4 > B[1] and l4 > D[1]:
            if l3 <= J[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B[0]:
                u3_shang = [F,B]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] < u3 <= B[0]:
                if P2[0] > F[0]:
                    u3_shang = [P,F,Q]
                else:
                    u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = [N]
            elif N[0] < u3 <= E[0]:
                if P2[0] > F[0]:
                    u3_shang = [P, F]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q, N]
            else:
                if P2[0] > F[0]:
                    u3_shang = [P, F]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [P1]

        elif B3[1] >= u4 > B[1] and D[1] >= l4 > C2[1]:
            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,K]
            elif D[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L, L1, D1, K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,D1, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [L1,D1, K]

            if u3 > B[0]:
                u3_shang = [F,B]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] < u3 <= B[0]:
                if P2[0] > F[0]:
                    u3_shang = [P,F,Q]
                else:
                    u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = [N]
            elif N[0] < u3 <= E[0]:
                if P2[0] > F[0]:
                    u3_shang = [P, F]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q, N]
            else:
                if P2[0] > F[0]:
                    u3_shang = [P, F]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [P1]

        elif B3[1] >= u4 > B[1] and C2[1] >= l4 > C[1]:
            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L, L1, C2]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]

            if u3 > B[0]:
                u3_shang = [F,B]
                u3_x3 = [E]
                u3_xia = [N,D1]
            elif E[0] < u3 <= B[0]:
                if P2[0] > F[0]:
                    u3_shang = [P,F,Q]
                else:
                    u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = [N,D1]
            elif N[0] < u3 <= E[0]:
                if P2[0] > F[0]:
                    u3_shang = [P, F]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q, N, D1]
            else:
                if P2[0] > F[0]:
                    u3_shang = [P, F]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] > D1[0]:
                    u3_xia = [P1,D1]
                else:
                    u3_xia = [H2]

        elif B3[1] >= u4 > B[1] and l4 <= C[1]:
            if l3 <= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D, C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L, L1, C2]
            elif I[0] < l3 <= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if u3 > B[0]:
                u3_shang = [F, B]
                u3_x3 = [E]
                u3_xia = [C]
            elif E[0] < u3 <= B[0]:
                if P2[0] > F[0]:
                    u3_shang = [P, F, Q]
                else:
                    u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = [C]
            elif N[0] < u3 <= E[0]:
                if P2[0] > F[0]:
                    u3_shang = [P, F]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q, C]
            else:
                if P2[0] > F[0]:
                    u3_shang = [P, F]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [H2]

        elif A[1] >= u4 > B3[1] and l4 > D[1]:
            if l3 <= J[0]:
                l3_shang = [B3,F, H]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [B3,F, H]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3,F, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                if M[0] > F[0]:
                    l3_shang = [B3,L3]
                else:
                    l3_shang = [M,F]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] < u3 <= B[0]:
                u3_shang = [P,Q]
                u3_x3 = [E]
                u3_xia = [N]
            elif N[0] < u3 <= E[0]:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q, N]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif A[1] >= u4 > B3[1] and D[1] >= l4 > C2[1]:
            if l3 <= D[0]:
                l3_shang = [B3,F, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,K]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3,F, H]
                l3_x3 = [I, O]
                l3_xia = [L, L1, D1, K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3,F, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,D1, K]
            else:
                if M[0] > F[0]:
                    l3_shang = [B3,L3]
                else:
                    l3_shang = [M,F]
                l3_x3 = [K_p, O]
                l3_xia = [L1,D1, K]

            if u3 > B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] < u3 <= B[0]:
                u3_shang = [P,Q]
                u3_x3 = [E]
                u3_xia = [N]
            elif N[0] < u3 <= E[0]:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q, N]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif A[1] >= u4 > B3[1] and C2[1] >= l4 > C[1]:
            if l3 <= D[0]:
                l3_shang = [B3,F, H]
                l3_x3 = [I, O]
                l3_xia = [D,C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3,F, H]
                l3_x3 = [I, O]
                l3_xia = [L, L1, C2]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3,F, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]
            else:
                if M[0] > F[0]:
                    l3_shang = [B3,L3]
                else:
                    l3_shang = [M,F]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]

            if u3 > B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [N,D1]
            elif E[0] < u3 <= B[0]:
                u3_shang = [P,Q]
                u3_x3 = [E]
                u3_xia = [N,D1]
            elif N[0] < u3 <= E[0]:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q, N,D1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if P1[0] > D1[0]:
                    u3_xia = [P1, D1]
                else:
                    u3_xia = [H2]

        elif A[1] >= u4 > B3[1] and l4 <= C[1]:
            if l3 <= D[0]:
                l3_shang = [B3,F, H]
                l3_x3 = [I, O]
                l3_xia = [D,C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3,F, H]
                l3_x3 = [I, O]
                l3_xia = [L, L1, C2]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3,F, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]
            else:
                if M[0] > F[0]:
                    l3_shang = [B3,L3]
                else:
                    l3_shang = [M,F]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]

            if u3 > B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [C]
            elif E[0] < u3 <= B[0]:
                u3_shang = [P,Q]
                u3_x3 = [E]
                u3_xia = [C]
            elif N[0] < u3 <= E[0]:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q, C]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [H2]

        elif u4 > A[1] and l4 > D[1]:
            if l3 <= J[0]:
                l3_shang = [B3,A]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] < l3 <= I[0]:
                l3_shang = [B3,A]
                l3_x3 = [I, O]
                l3_xia = [L,M_p, K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3,A, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [B3,L3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if u3 > B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] < u3 <= B[0]:
                u3_shang = [P,Q]
                u3_x3 = [E]
                u3_xia = [N]
            elif N[0] < u3 <= E[0]:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q, N]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif u4 > A[1] and D[1] >= l4 > C2[1]:
            if l3 <= D[0]:
                l3_shang = [B3,A]
                l3_x3 = [I,O]
                l3_xia = [D,D1,K]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3,A]
                l3_x3 = [I, O]
                l3_xia = [L,L1,D1,K]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3,A, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,D1,K]
            else:
                l3_shang = [B3,L3]
                l3_x3 = [K_p, O]
                l3_xia = [L1,D1, K]

            if u3 > B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] < u3 <= B[0]:
                u3_shang = [P,Q]
                u3_x3 = [E]
                u3_xia = [N]
            elif N[0] < u3 <= E[0]:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q, N]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif u4 > A[1] and C2[1] >= l4 > C[1]:
            if l3 <= D[0]:
                l3_shang = [B3,A]
                l3_x3 = [I,O]
                l3_xia = [D,C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3,A]
                l3_x3 = [I, O]
                l3_xia = [L,L1,C2]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3,A, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]
            else:
                l3_shang = [B3,L3]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]

            if u3 > B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [N,D1]
            elif E[0] < u3 <= B[0]:
                u3_shang = [P,Q]
                u3_x3 = [E]
                u3_xia = [N,D1]
            elif N[0] < u3 <= E[0]:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q, N,D1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if P1[0] > D1[0]:
                    u3_xia = [P1, D1]
                else:
                    u3_xia = [H2]

        else:
            if l3 <= D[0]:
                l3_shang = [B3,A]
                l3_x3 = [I,O]
                l3_xia = [D,C2]
            elif D[0] < l3 <= I[0]:
                l3_shang = [B3,A]
                l3_x3 = [I, O]
                l3_xia = [L,L1,C2]
            elif I[0] < l3 <= H[0]:
                l3_shang = [B3,A, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]
            else:
                l3_shang = [B3,L3]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]

            if u3 > B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [C]
            elif E[0] < u3 <= B[0]:
                u3_shang = [P,Q]
                u3_x3 = [E]
                u3_xia = [C]
            elif N[0] < u3 <= E[0]:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q, C]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [H2]

    # 对情况 05 左右镜像
    elif A[0] >= 0 and A[1] >= 0 and B[0] >= 0 and B[1] <= 0 and C[0] <= 0 and C[1] <= 0 and D[0] <= 0 and D[1] >= 0:
        A = [0.5 * (J3 + J2), 0.5 * (J3 - J2),0]
        B = [0.5 * (J4 + J2), 0.5 * (J4 - J2),0]
        C = [0.5 * (J4 + J1), 0.5 * (J4 - J1),0]
        D = [0.5 * (J3 + J1), 0.5 * (J3 - J1),0]
        H = [-u4 + J3, u4, 0]
        G = [0, u4, 0]
        F = [-u4 + J4, u4, 0]
        A1 = [-J2 + u4, u4, 0]
        A2 = [-J1 + u4, u4, 0]
        J = [-l4 + J3, l4, 0]
        K = [0, l4, 0]
        N = [-l4 + J4, l4, 0]
        P1 = [l3, l4, 0]
        D1 = [-J1 + l4, l4, 0]
        B2 = [-J2 + l4, l4, 0]
        L = [u3, -u3 + J3, 0]
        M = [u3, u4, 0]
        K_p = [u3, 0, 0]
        M_p = [u3, l4, 0]
        L1 = [u3, J1 + u3, 0]
        L2 = [u3, J4 - u3, 0]
        L3 = [u3, J2 + u3, 0]
        P = [l3, J2 + l3, 0]
        Q = [l3, J4 - l3, 0]
        P2 = [l3, u4, 0]
        R = [l3, 0, 0]
        H1 = [l3, -l3 + J3, 0]
        H2 = [l3, J1 + l3, 0]
        I = [-J3, 0, 0]
        B1 = [0, J3, 0]
        E = [-J4, 0, 0]
        C1 = [0, J4, 0]
        B3 = [0, J2, 0]
        I1 = [J2, 0, 0]
        C2 = [0, J1, 0]
        D2 = [J1, 0, 0]
        O = [0, 0, 0]

        if B[1] >= u4 > 0 >= l4 > D[1]:
            if u3 >= J[0]:
                l3_shang = [G,H]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L,M_p, K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] > l3 >= F[0]:
                u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = [N]
            elif N[0] > l3 >= E[0]:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q,N]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [P1]

        elif B[1] >= u4 and D[1] >= l4 > C2[1]:
            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,K]
            elif D[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L, L1, D1, K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,D1, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [L1,D1, K]

            if l3 < F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] > l3 >= F[0]:
                u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = [N]
            elif N[0] > l3 >= E[0]:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q, N]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [P1]

        elif B[1] >= u4 and C2[1] >= l4 > C[1]:
            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L, L1, C2]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]

            if l3 < F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = [N,D1]
            elif E[0] > l3 >= F[0]:
                u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = [N,D1]
            elif N[0] > l3 >= E[0]:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q, N,D1]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] < D1[0]:
                    u3_xia = [P1,D1]
                else:
                    u3_xia = [H2]

        elif B[1] >= u4 and l4 <= C[1]:
            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L, L1, C2]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]

            if l3 < F[0]:
                u3_shang = [F]
                u3_x3 = [E]
                u3_xia = [C]
            elif E[0] > l3 >= F[0]:
                u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = [C]
            elif N[0] > l3 >= E[0]:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q, C]
            else:
                u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [H2]

        elif B3[1] >= u4 > B[1] and l4 > D[1]:
            if u3 >= J[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B[0]:
                u3_shang = [F,B]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] > l3 >= B[0]:
                if P2[0] < F[0]:
                    u3_shang = [P,F,Q]
                else:
                    u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = [N]
            elif N[0] > l3 >= E[0]:
                if P2[0] < F[0]:
                    u3_shang = [P, F]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q, N]
            else:
                if P2[0] < F[0]:
                    u3_shang = [P, F]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [P1]

        elif B3[1] >= u4 > B[1] and D[1] >= l4 > C2[1]:
            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,K]
            elif D[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L, L1, D1, K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,D1, K]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [L1,D1, K]

            if l3 < B[0]:
                u3_shang = [F,B]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] > l3 >= B[0]:
                if P2[0] < F[0]:
                    u3_shang = [P,F,Q]
                else:
                    u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = [N]
            elif N[0] > l3 >= E[0]:
                if P2[0] < F[0]:
                    u3_shang = [P, F]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q, N]
            else:
                if P2[0] < F[0]:
                    u3_shang = [P, F]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [P1]

        elif B3[1] >= u4 > B[1] and C2[1] >= l4 > C[1]:
            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D,C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L, L1, C2]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]

            if l3 < B[0]:
                u3_shang = [F,B]
                u3_x3 = [E]
                u3_xia = [N,D1]
            elif E[0] > l3 >= B[0]:
                if P2[0] < F[0]:
                    u3_shang = [P,F,Q]
                else:
                    u3_shang = [P2,Q]
                u3_x3 = [E]
                u3_xia = [N,D1]
            elif N[0] > l3 >= E[0]:
                if P2[0] < F[0]:
                    u3_shang = [P, F]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q, N, D1]
            else:
                if P2[0] < F[0]:
                    u3_shang = [P, F]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                if P1[0] < D1[0]:
                    u3_xia = [P1,D1]
                else:
                    u3_xia = [H2]

        elif B3[1] >= u4 > B[1] and l4 <= C[1]:
            if u3 >= D[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [D, C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [G, H]
                l3_x3 = [I, O]
                l3_xia = [L, L1, C2]
            elif I[0] > u3 >= H[0]:
                l3_shang = [G, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]
            else:
                l3_shang = [G, M]
                l3_x3 = [K_p, O]
                l3_xia = [L1, C2]

            if l3 < B[0]:
                u3_shang = [F, B]
                u3_x3 = [E]
                u3_xia = [C]
            elif E[0] > l3 >= B[0]:
                if P2[0] < F[0]:
                    u3_shang = [P, F, Q]
                else:
                    u3_shang = [P2, Q]
                u3_x3 = [E]
                u3_xia = [C]
            elif N[0] > l3 >= E[0]:
                if P2[0] < F[0]:
                    u3_shang = [P, F]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [Q, C]
            else:
                if P2[0] < F[0]:
                    u3_shang = [P, F]
                else:
                    u3_shang = [P2]
                u3_x3 = [R]
                u3_xia = [H2]

        elif A[1] >= u4 > B3[1] and l4 > D[1]:
            if u3 >= J[0]:
                l3_shang = [B3,F, H]
                l3_x3 = [I, O]
                l3_xia = [J, K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [B3,F, H]
                l3_x3 = [I, O]
                l3_xia = [L, M_p, K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3,F, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                if M[0] < F[0]:
                    l3_shang = [B3,L3]
                else:
                    l3_shang = [M,F]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] > l3 >= B[0]:
                u3_shang = [P,Q]
                u3_x3 = [E]
                u3_xia = [N]
            elif N[0] > l3 >= E[0]:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q, N]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif A[1] >= u4 > B3[1] and D[1] >= l4 > C2[1]:
            if u3 >= D[0]:
                l3_shang = [B3,F, H]
                l3_x3 = [I, O]
                l3_xia = [D,D1,K]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3,F, H]
                l3_x3 = [I, O]
                l3_xia = [L, L1, D1, K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3,F, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,D1, K]
            else:
                if M[0] < F[0]:
                    l3_shang = [B3,L3]
                else:
                    l3_shang = [M,F]
                l3_x3 = [K_p, O]
                l3_xia = [L1,D1, K]

            if l3 < B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] > l3 >= B[0]:
                u3_shang = [P,Q]
                u3_x3 = [E]
                u3_xia = [N]
            elif N[0] > l3 >= E[0]:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q, N]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif A[1] >= u4 > B3[1] and C2[1] >= l4 > C[1]:
            if u3 >= D[0]:
                l3_shang = [B3,F, H]
                l3_x3 = [I, O]
                l3_xia = [D,C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3,F, H]
                l3_x3 = [I, O]
                l3_xia = [L, L1, C2]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3,F, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]
            else:
                if M[0] < F[0]:
                    l3_shang = [B3,L3]
                else:
                    l3_shang = [M,F]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]

            if l3 < B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [N,D1]
            elif E[0] > l3 <= B[0]:
                u3_shang = [P,Q]
                u3_x3 = [E]
                u3_xia = [N,D1]
            elif N[0] > l3 >= E[0]:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q, N,D1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if P1[0] < D1[0]:
                    u3_xia = [P1, D1]
                else:
                    u3_xia = [H2]

        elif A[1] >= u4 > B3[1] and l4 <= C[1]:
            if u3 >= D[0]:
                l3_shang = [B3,F, H]
                l3_x3 = [I, O]
                l3_xia = [D,C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3,F, H]
                l3_x3 = [I, O]
                l3_xia = [L, L1, C2]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3,F, H, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]
            else:
                if M[0] < F[0]:
                    l3_shang = [B3,L3]
                else:
                    l3_shang = [M,F]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]

            if l3 < B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [C]
            elif E[0] > l3 >= B[0]:
                u3_shang = [P,Q]
                u3_x3 = [E]
                u3_xia = [C]
            elif N[0] > l3 >= E[0]:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q, C]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [H2]

        elif u4 > A[1] and l4 > D[1]:
            if u3 >= J[0]:
                l3_shang = [B3,A]
                l3_x3 = [I,O]
                l3_xia = [J,K]
            elif J[0] > u3 >= I[0]:
                l3_shang = [B3,A]
                l3_x3 = [I, O]
                l3_xia = [L,M_p, K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3,A, L]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]
            else:
                l3_shang = [B3,L3]
                l3_x3 = [K_p, O]
                l3_xia = [M_p, K]

            if l3 < B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] > l3 >= B[0]:
                u3_shang = [P,Q]
                u3_x3 = [E]
                u3_xia = [N]
            elif N[0] > l3 >= E[0]:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q, N]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif u4 > A[1] and D[1] >= l4 > C2[1]:
            if u3 >= D[0]:
                l3_shang = [B3,A]
                l3_x3 = [I,O]
                l3_xia = [D,D1,K]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3,A]
                l3_x3 = [I, O]
                l3_xia = [L,L1,D1,K]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3,A, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,D1,K]
            else:
                l3_shang = [B3,L3]
                l3_x3 = [K_p, O]
                l3_xia = [L1,D1, K]

            if l3 < B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [N]
            elif E[0] > l3 >= B[0]:
                u3_shang = [P,Q]
                u3_x3 = [E]
                u3_xia = [N]
            elif N[0] > l3 >= E[0]:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q, N]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [P1]

        elif u4 > A[1] and C2[1] >= l4 > C[1]:
            if u3 >= D[0]:
                l3_shang = [B3,A]
                l3_x3 = [I,O]
                l3_xia = [D,C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3,A]
                l3_x3 = [I, O]
                l3_xia = [L,L1,C2]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3,A, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]
            else:
                l3_shang = [B3,L3]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]

            if l3 < B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [N,D1]
            elif E[0] > l3 >= B[0]:
                u3_shang = [P,Q]
                u3_x3 = [E]
                u3_xia = [N,D1]
            elif N[0] > l3 >= E[0]:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q, N,D1]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                if P1[0] < D1[0]:
                    u3_xia = [P1, D1]
                else:
                    u3_xia = [H2]

        else:
            if u3 >= D[0]:
                l3_shang = [B3,A]
                l3_x3 = [I,O]
                l3_xia = [D,C2]
            elif D[0] > u3 >= I[0]:
                l3_shang = [B3,A]
                l3_x3 = [I, O]
                l3_xia = [L,L1,C2]
            elif I[0] > u3 >= H[0]:
                l3_shang = [B3,A, L]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]
            else:
                l3_shang = [B3,L3]
                l3_x3 = [K_p, O]
                l3_xia = [L1,C2]

            if l3 < B[0]:
                u3_shang = [B]
                u3_x3 = [E]
                u3_xia = [C]
            elif E[0] > l3 >= B[0]:
                u3_shang = [P,Q]
                u3_x3 = [E]
                u3_xia = [C]
            elif N[0] > l3 >= E[0]:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [Q, C]
            else:
                u3_shang = [P]
                u3_x3 = [R]
                u3_xia = [H2]

    # get_sig1 接受上下镜像信号互换_xia 与 _shang
    if sig1:
        temp = u3_shang
        u3_shang = u3_xia
        u3_xia = temp

        temp = l3_shang
        l3_shang = l3_xia
        l3_xia = temp

    # 对x3>0的点进行翻折
    for List in [u3_shang, u3_x3, u3_xia]:
        for point in List:
            point[2] = point[0]

    # 构造lower凸包之极点
    list_lower = list_lower + l3_shang + l3_x3 + u3_shang + u3_x3

    # 构造upper凸包之极点
    list_upper = list_upper + l3_shang + l3_xia + u3_shang + u3_xia

    try:
        if list_lower and list_upper:
            V_lower = ConvexHull(np.array(list_lower)).volume
            V_upper = ConvexHull(np.array(list_upper)).volume
        else:
            V_lower = 0
            V_upper = 0
    except:
        V_lower = 0
        V_upper = 0
    return V_lower, V_upper


# 方便pool的函数，包含了调用四维凸包计算函数calculate_3D_vol以及换位的一串if,可能得考虑把krelu.py中的全序列kact_args(未加1-relu的时候)加进来帮助使用pool
def target_pool():

    return 0

# main函数，供调用
def M_select_main(nn, layer_depth, lbi, ubi, split_zero, prebound_l, prebound_u):
    """
    :param nn: 目标网络模型
    :param layer_depth: 当前隐藏层是第几层 即是 analyzer.py 174行中的i/2(整除)-1
    :param lbi: 记录了第layer_depth层上节点的下界 -> lbi[-1]  1*neuron_num(layer_depth)
    :param ubi: 记录了第layer_depth层上节点的上界 -> ubi[-1]  1*neuron_num(layer_depth)
    :param split_zero: 记录了第lay_depth层有哪几个激活状态未定的点 list< dtype = int >
    :return: kact_args: k=2下的点对的列表 list< dtype = list >
    :return: rest_point: 暂时没分配到的点组成的列表
    """

    # 权重weight neuron_num(i-1)*neuron_num(i)
    # 偏置bias neuron_num(i)得是个list
    weight = nn.weights[layer_depth]
    bias = nn.biases[lM_select_1.pyayer_depth]

    # 将weight,bias,lbi,ubi从list转为array并统一成行向量
    weight = np.array(weight)
    bias = (np.array(bias)).reshape(1,-1)
    lbi = (np.array(lbi)).reshape(1,-1)
    ubi = (np.array(ubi)).reshape(1,-1)
    prebound_l = (np.array(prebound_l)).reshape(1,-1)
    prebound_u = (np.array(prebound_u)).reshape(1, -1)
    # 维护一个3 * len(split_zero)的array用来通过四维胞积的大小决定哪个点与哪个点配对
    # 第一行，第二行分别记录四维下界、上界，第三行记录其配对目标select_point
    record_list = np.ones(shape=[4,len(split_zero)])
    record_list = -record_list
    # with multiprocessing.Pool(config.numproc) as pool:
    #     res = pool.map(calculate_3D_vol, zip())

    for i in range(len(split_zero)-1):
        idx_fix = split_zero[i]
        for j in range(i+1, len(split_zero)):
            idx_select = split_zero[j]
            # 得到索引i和索引j所代表的节点之间构成的三维凸包(x3 >= 0部分)的体积V_3_0 以及 整体凸包的体积V_upper
            V_lower, V_upper = calculate_3D_vol(weight, bias, lbi, ubi, idx_fix, idx_select, prebound_l, prebound_u)

            if record_list[0][i] == -1:
                if V_upper != 0 and V_lower != 0:
                    record_list[0][i] = V_lower
                    record_list[1][i] = V_upper
                    record_list[2][i] = idx_select
                record_list[3][i] = False
            if record_list[0][j] == -1:
                if V_upper != 0 and V_lower != 0:
                    record_list[0][j] = V_lower
                    record_list[1][j] = V_upper
                    record_list[2][j] = idx_fix
                record_list[3][j] = False
            # if ((V_lower >= record_list[0][i] and V_lower <= record_list[1][i]) or (V_upper >= record_list[0][i] and V_upper <= record_list[1][i])) and record_list[3][i] == True:
            #     if V_lower < record_list[0][i]:
            #         record_list[0][i] = V_lower
            #     record_list[3][i] = False
            #
            #
            # if ((V_lower >= record_list[0][j] and V_lower <= record_list[1][j]) or (V_upper >= record_list[0][j] and V_upper <= record_list[1][j])) and record_list[3][j] == True:
            #     if V_lower < record_list[0][j]:
            #         record_list[0][j] = V_lower
            #     record_list[3][j] = False


            if V_upper <= record_list[0][i] and V_upper != 0:
                record_list[0][i] = V_lower
                record_list[1][i] = V_upper
                record_list[2][i] = idx_select
                record_list[3][i] = True

            if V_upper <= record_list[0][j] and V_upper != 0:
                record_list[0][j] = V_lower
                record_list[1][j] = V_upper
                record_list[2][j] = idx_fix
                record_list[3][j] = True

    # 返回没被选到的点
    rest_point = []
    kact_args = []
    for k in range(len(split_zero)):
        if record_list[3][k] == False:
            rest_point.append(split_zero[k])
        else:
            kact_args.append(tuple([int(split_zero[k]), int(record_list[2][k])]))


    return kact_args, rest_point