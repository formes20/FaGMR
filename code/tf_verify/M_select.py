"""
作者：LENOVO
日期：2022年06月25日
"""
from scipy.spatial import ConvexHull
import numpy as np
# import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import os
import time
import math
from config import *
from My_fun import *
from elina_scalar import *
from elina_dimension import *
from elina_linexpr0 import *
from elina_abstract0 import *
from fppoly import *
from fconv import *
import itertools
import random
from quickhull import *
# points = np.array([[1,0,0],[0,1,0],[0,0,0],[1,1,0],[1,0,1],[0,1,1],[0,0,1],[1,1,100]])  # your points
# volume = ConvexHull(points).volume
#
# c = [[1,1],[2,2]]
# b = [[3,3]]
# a = []
# a = c + b
# a = 1

def generate_linexpr0(offset, varids, coeffs):
    # returns ELINA expression, equivalent to sum_i(varids[i]*coeffs[i])
    assert len(varids) == len(coeffs)
    n = len(varids)

    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_SPARSE, n)
    cst = pointer(linexpr0.contents.cst)
    elina_scalar_set_double(cst.contents.val.scalar, 0)

    for i, (x, coeffx) in enumerate(zip(varids, coeffs)):
        linterm = pointer(linexpr0.contents.p.linterm[i])
        linterm.contents.dim = ElinaDim(offset + x)
        coeff = pointer(linterm.contents.coeff)
        elina_scalar_set_double(coeff.contents.val.scalar, coeffx)

    return linexpr0

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

def target_pool_k(nn, weight, bias, lbi, ubi, lis, split_zero, prebound_l, prebound_u, man, element, layerno):
    # 先写k=3看看
    offset = 0
    total_size = 0
    i = split_zero.index(int(lis[0]))
    count = 0

    for l in range(len(lis)-2):
        idx_fix = lis[l]
        i = i + l
        for j in range(i+1,len(split_zero)-1):
            idx_select_01 = split_zero[j]
            for g in range(j+1, len(split_zero)):
                idx_select_02 = split_zero[g]

                # get box P from ELINA
                total_size = 3 ** config.K - 1
                linexpr0 = elina_linexpr0_array_alloc(total_size)
                ii = 0

                varsid = tuple([idx_fix, idx_select_01, idx_select_02])
                for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
                    if all(c == 0 for c in coeffs):
                        continue
                    linexpr0[ii] = generate_linexpr0(offset, varsid, coeffs)
                    ii = ii + 1
                upper_bound = get_upper_bound_for_linexpr0(man, element, linexpr0, total_size, layerno)

                ii = 0
                a = []
                bi = []

                input_hrep = []
                for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
                    if all(c == 0 for c in coeffs):
                        continue
                    a.append([-c for c in coeffs])
                    bi.append(upper_bound[ii])
                    input_hrep.append([upper_bound[ii]] + [-c for c in coeffs])
                    ii = ii + 1
                #P中的约束： Ax <= b
                A = np.mat(a)
                b = np.mat(bi).transpose()

                ux= con2vert(A, b)
                V = ConvexHull(ux).volume

                u_fix = ubi[0,idx_fix]
                u_select_01 = ubi[0,idx_select_01]
                u_select_02 = ubi[0, idx_select_02]

                l_fix = lbi[0, idx_fix]
                l_select_01 = lbi[0, idx_select_01]
                l_select_02 = lbi[0, idx_select_02]

                for u_idx in range(config.k):
                    uu = max(ux[:,u_idx])
                    if u_idx == 0 and 0 < uu < u_fix:
                        u_fix = uu
                    if u_idx == 1 and 0 < uu < u_select_01:
                        u_select_01 = uu
                    if u_idx == 2 and 0 < uu < u_select_02:
                        u_select_02 = uu

                for l_idx in range(config.k):
                    ll = min(ux[:,l_idx])
                    if l_idx == 0 and ll > l_fix:
                        l_fix = ll
                    if l_idx == 1 and ll > l_select_01:
                        l_select_01 = ll
                    if l_idx == 2 and ll < l_select_02:
                        l_select_02 = ll

                # V = Vover(ux,0.05)
                # try:
                if 'ReLU' in nn.layertypes:
                    V_upper = V * u_fix * u_select_01 * u_select_02
                    V_lower = V_upper * 1/6 * 1/5 * 1/4
                    # V_lower = 1/factorial(2*config.k) * u_fix * u_select_01 * u_select_02 * u_fix * u_select_01 * u_select_02
                elif 'Sigmoid' in nn.layertypes:
                    V_upper = V * (Sigmoid(u_fix)-Sigmoid(l_fix)) * (Sigmoid(u_select_01)-Sigmoid(l_select_01)) * (Sigmoid(u_select_02)-Sigmoid(l_select_02))
                    V_lower = 1/factorial(2*config.k) * V_upper/V * (u_fix-l_fix) * (u_select_01-l_select_01) * (u_select_02-l_select_02)
                    # V_lower = V_upper * 1 / 6 * 1 / 5 * 1 / 4
                elif 'Tanh' in nn.layertypes:
                    V_upper = V * (Tanh(u_fix)-Tanh(l_fix)) * (Tanh(u_select_01)-Tanh(l_select_01)) * (Tanh(u_select_02)-Tanh(l_select_02))
                    V_lower = 1/factorial(2*config.k) * V_upper/V * (u_fix-l_fix) * (u_select_01-l_select_01) * (u_select_02-l_select_02)

                if used.record[0][i] == -1:
                    if V_upper != 0 and V_lower != 0:
                        used.record[0][i] = V_lower
                        used.record[1][i] = V_upper
                        used.record[2][i] = True
                        used.record[3][i].append(tuple([idx_select_01, idx_select_02]))
                        # used.record[3][i] = [tuple([idx_select_01, idx_select_02])]

                if used.record[0][j] == -1:
                    if V_upper != 0 and V_lower != 0:
                        used.record[0][j] = V_lower
                        used.record[1][j] = V_upper
                        used.record[2][j] = True

                if used.record[0][g] == -1:
                    if V_upper != 0 and V_lower != 0:
                        used.record[0][g] = V_lower
                        used.record[1][g] = V_upper
                        used.record[2][g] = True

                if used.record[2][i] == True:
                    # if V_upper < used.record[1][i] and V_lower < used.record[0][i]:
                    #     used.record[3][i] = []
                    #     used.record[0][i] = V_lower
                    #     used.record[1][i] = V_upper
                    #     used.record[3][i].append(tuple([idx_select_01, idx_select_02]))

                    if max(V_lower,used.record[0][i]) <= min(V_upper,used.record[1][i]) and V_upper >= used.record[0][i] and V_lower <= used.record[1][i]:
                        used.record[1][i] = min(V_upper, used.record[1][i])
                        used.record[3][i].append(tuple([idx_select_01,idx_select_02]))
                        used.record[0][i] = max(V_lower,used.record[0][i])
                    elif V_upper <= used.record[0][i]:
                        used.record[3][i] = []
                        used.record[0][i] = V_lower
                        used.record[1][i] = V_upper
                        used.record[3][i].append(tuple([idx_select_01,idx_select_02]))

                if used.record[2][j] == True:
                    if max(V_lower, used.record[0][j]) <= min(V_upper, used.record[1][j]) and V_upper >= used.record[0][j] and V_lower <= used.record[1][j]:
                        used.record[1][j] = min(V_upper, used.record[1][j])
                        used.record[3][j].append(tuple([idx_fix,idx_select_02]))
                        used.record[0][j] = max(V_lower, used.record[0][j])
                    elif V_upper <= used.record[0][j]:
                        used.record[3][j] = []
                        used.record[0][j] = V_lower
                        used.record[1][j] = V_upper
                        used.record[3][j].append(tuple([idx_fix,idx_select_02]))

                if used.record[2][g] == True:
                    if max(V_lower,used.record[0][g]) <= min(V_upper,used.record[1][g]) and V_upper >= used.record[0][g] and V_lower <= used.record[1][g]:
                        used.record[1][g] = min(V_upper, used.record[1][g])
                        used.record[3][g].append(tuple([idx_fix,idx_select_01]))
                        used.record[0][g] = max(V_lower,used.record[0][g])
                    elif V_upper <= used.record[0][g]:
                        used.record[3][g] = []
                        used.record[0][g] = V_lower
                        used.record[1][g] = V_upper
                        used.record[3][g].append(tuple([idx_fix,idx_select_01]))

        count += 1
    return

# def target_pool_k_reverse(weight, bias, lbi, ubi, lis, split_zero, prebound_l, prebound_u, man, element, layerno):
#     # 先写k=3看看
#     offset = 0
#     total_size = 0
#     i = split_zero.index(lis[0])
#     count = 0
#     for l in range(len(lis)-2):
#         idx_fix = lis[l]
#         i = i + l
#         for j in range(i+1,len(split_zero)-1):
#             idx_select_01 = split_zero[j]
#             for g in range(j+1, len(split_zero)):
#                 idx_select_02 = split_zero[g]
#
#                 # get box P from ELINA
#                 total_size = 3 ** config.K - 1
#                 linexpr0 = elina_linexpr0_array_alloc(total_size)
#                 ii = 0
#
#                 varsid = tuple([idx_fix, idx_select_01, idx_select_02])
#                 for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
#                     if all(c == 0 for c in coeffs):
#                         continue
#                     linexpr0[ii] = generate_linexpr0(offset, varsid, coeffs)
#                     ii = ii + 1
#                 upper_bound = get_upper_bound_for_linexpr0(man, element, linexpr0, total_size, layerno)
#
#                 ii = 0
#                 a = []
#                 bi = []
#                 input_hrep = []
#                 for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
#                     if all(c == 0 for c in coeffs):
#                         continue
#                     a.append([-c for c in coeffs])
#                     bi.append(upper_bound[ii])
#                     input_hrep.append([upper_bound[ii]] + [-c for c in coeffs])
#                     ii = ii + 1
#
#                 A = np.mat(a)
#                 b = np.mat(bi).transpose()
#
#                 ux= con2vert(A, b)
#                 # print(ux)
#                 try:
#                     V_upper = Vover(ux,0.5)
#                     # V_upper = ConvexHull(np.array(ux)).volume
#                 except:
#                     V_upper = 0
#
#                 # coff = ubi[0,idx_fix] * ubi[0,idx_select_01] * ubi[0,idx_select_02]
#                 # V_upper = V_upper * coff
#
#                 # when k = 3, exchang x>=-number to x>=0 in b, idx = 13 15 21
#                 # for number in [13,15,21]:
#                 #     b[number][0] = 0
#                 # ux = con2vert(A,b)
#
#                 try:
#                     V_lower = 1/(2*config.k) * V_upper
#                 except:
#                     V_lower = 0
#
#                 # V_lower = V_lower * coff
#
#                 if used.record[0][i] == -1:
#                     if V_upper != 0 and V_lower != 0:
#                         used.record[0][i] = V_lower
#                         used.record[1][i] = V_upper
#                         used.record[2][i] = False
#
#                 if used.record[0][j] == -1:
#                     if V_upper != 0 and V_lower != 0:
#                         used.record[0][j] = V_lower
#                         used.record[1][j] = V_upper
#                         used.record[2][j] = False
#
#
#                 if V_upper <= used.record[0][i] and V_upper != 0 and used.record[2][i] == False:
#                     used.record[0][i] = V_lower
#                     used.record[1][i] = V_upper
#                     used.record[2][i] = True
#                     # kact_args_i.append(tuple([int(split_zero[i]), int(split_zero[j])]))
#
#                 if V_upper <= used.record[0][j] and V_upper != 0 and used.record[2][j] == False:
#                     used.record[0][j] = V_lower
#                     used.record[1][j] = V_upper
#                     used.record[2][j] = True
#                     # kact_args_i.append(tuple([int(split_zero[j]), int(split_zero[i])]))
#
#                 if V_upper <= used.record[0][g] and V_upper != 0 and used.record[2][g] == False:
#                     used.record[0][g] = V_lower
#                     used.record[1][g] = V_upper
#                     used.record[2][g] = True
#
#                 if used.record[2][i] == True:
#                     if max(V_lower,used.record[0][i]) <= min(V_upper,used.record[1][i]) and V_lower <= used.record[1][i]:
#                         used.record[1][i] = min(V_upper, used.record[1][i])
#                         used.record[3][i].append(tuple([idx_select_01,idx_select_02]))
#                         used.record[0][i] = max(V_lower,used.record[0][i])
#                     elif V_upper <= used.record[0][i]:
#                         used.record[3][i] = []
#                         used.record[0][i] = V_lower
#                         used.record[1][i] = V_upper
#                         used.record[3][i].append(tuple([idx_select_01,idx_select_02]))
#
#                 if used.record[2][j] == True:
#                     if max(V_lower, used.record[0][j]) <= min(V_upper, used.record[1][j]) and V_lower <= used.record[1][j]:
#                         used.record[1][j] = min(V_upper, used.record[1][j])
#                         used.record[3][j].append(tuple([idx_fix,idx_select_02]))
#                         used.record[0][j] = max(V_lower, used.record[0][j])
#                     elif V_upper <= used.record[0][j]:
#                         used.record[3][j] = []
#                         used.record[0][j] = V_lower
#                         used.record[1][j] = V_upper
#                         used.record[3][j].append(tuple([idx_fix,idx_select_02]))
#
#                 if used.record[2][g] == True:
#                     if max(V_lower,used.record[0][g]) <= min(V_upper,used.record[1][g]) and used.record[1][g] <= used.record[1][g]:
#                         used.record[1][g] = min(V_upper, used.record[1][g])
#                         used.record[3][g].append(tuple([idx_fix,idx_select_01]))
#                         used.record[0][g] = max(V_lower,used.record[0][g])
#                     elif V_upper <= used.record[0][g]:
#                         used.record[3][g] = []
#                         used.record[0][g] = V_lower
#                         used.record[1][g] = V_upper
#                         used.record[3][g].append(tuple([idx_fix,idx_select_01]))
#
#         count += 1
#     return

def M_select_main(nn, layer_depth, lbi, ubi, split_zero, prebound_l, prebound_u, man, element, layerno):
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
    # weight = nn.weights[layer_depth]
    # bias = nn.biases[layer_depth + nn.layertypes.count('Conv')]
    weight = used.weight_new[layer_depth]
    bias = used.bias_new[layer_depth]

    # 将weight,bias,lbi,ubi从list转为array并统一成行向量
    weight = np.array(weight)
    bias = (np.array(bias)).reshape(1,-1)

    weight = np.around(weight, decimals=4, out=None)
    bias = np.around(bias, decimals=4, out=None)

    lbi = np.around((np.array(lbi)).reshape(1,-1), decimals=4, out = None)
    ubi = np.around((np.array(ubi)).reshape(1,-1), decimals=4, out = None)



    prebound_l = np.around((np.array(prebound_l)).reshape(1,-1), decimals=4, out=None)
    prebound_u = np.around((np.array(prebound_u)).reshape(1, -1), decimals=4, out=None)
    # 维护一个3 * len(split_zero)的array用来通过四维胞积的大小决定哪个点与哪个点配对
    # 第一行，第二行分别记录四维下界、上界，第三行记录其配对目标select_point

    kact_args = []
    # rest_point = []

    used.rest_points = []

    # with multiprocessing.Pool(config.numproc) as pool:
    #     res = pool.map(calculate_3D_vol, zip())

    # 并fa
    num_core = 2
    temp = EveryStrandIsN(split_zero,os.cpu_count())
    used.record = -np.ones(shape=[4,len(split_zero)])
    used.record = used.record.tolist()
    for i in range(len(split_zero)):
        used.record[3][i] = []
        used.record[2][i] = False
    # for i in temp:
    #     print(i)

    pool = ThreadPool(num_core)
    split_zero_reverse = list(reversed(split_zero))
    split_zero_random = split_zero.copy()
    random.shuffle(split_zero_random)
    # pool = ThreadPool(12)
    # result = pool.starmap(target_pool_1,
    #                             [(weight, bias, lbi, ubi, lis, split_zero, prebound_l, prebound_u)for lis in temp] )

    if config.K == 2:
        pool.starmap(target_pool,[(weight, bias, lbi, ubi, lis, split_zero, prebound_l, prebound_u)for lis in temp] )
    else:
        if config.test == 'reverse':
            pool.starmap(target_pool_k_reverse,
                         [(weight, bias, lbi, ubi, lis, split_zero_reverse, prebound_l, prebound_u, man, element, layerno) for
                          lis in temp])
        elif config.test == 'random':
            pool.starmap(target_pool_k,
                         [(weight, bias, lbi, ubi, lis, split_zero_random, prebound_l, prebound_u, man, element, layerno) for
                          lis in temp])
        else:
            pool.starmap(target_pool_k, [(nn,weight, bias, lbi, ubi, lis, split_zero, prebound_l, prebound_u, man, element, layerno) for lis in temp])
    pool.close()
    pool.join()

    # for l in result:
    #     if l == []:
    #         continue
    #     else:
    #         kact_args.extend(l)

    if config.K == 2:
        for i in range(len(split_zero)):
            if used.record[2][i] == False:
                used.rest_points.append(split_zero[i])
            else:
                for j in used.record[3][i]:
                    kact_args.append(tuple([int(split_zero[i]), int(j)]))
    else:
        for i in range(len(split_zero)):
            if used.record[3][i] == []:
                used.rest_points.append(split_zero[i])
            else:
                for j in used.record[3][i]:
                    kact_args.append(tuple([split_zero[i]])+j)

    # 对元组进行重复去除
    for t in range(len(kact_args)):
        kact_args[t] = list(kact_args[t])
        kact_args[t].sort()
        kact_args[t] = tuple(kact_args[t])

    kact_args = list(set(kact_args))

    rest_point = used.rest_points

    return kact_args, rest_point