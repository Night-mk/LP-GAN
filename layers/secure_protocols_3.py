'''
    secure_protocols_2.py 构建Z2上的安全协议 
    直接计算的版本
'''

import random
import numpy as np
import math
import time
import secure_protocols as secp

# 只需要传递2bit长度的数据
# mul不存在offline，初始化时已有大量生成
def SecMul_z2(u1, v1, u2, v2, bit_length=1):
    input_shape = u1.shape
    a1,a2,b1,b2,c1,c2 = mul_generate_random_z2(input_shape)
    # S1
    alpha1 = u1 - a1
    beta1 = v1 - b1
    # S2
    alpha2 = u2 - a2
    beta2 = v2 - b2

    # S1
    S1_alpha = alpha1 + alpha2
    S1_beta = beta1 + beta2
    # f1 = np.mod(np.mod(c1 + b1 * S1_alpha, 2) + a1 * S1_beta, 2)
    f1 = (c1 + b1 * S1_alpha + a1 * S1_beta)&(2-1)
    # S2
    S2_alpha = alpha1 + alpha2
    S2_beta = beta1 + beta2
    # f2 = np.mod(np.mod(np.mod(c2 + b2 * S2_alpha, 2) + a2 * S2_beta, 2) + S2_alpha * S2_beta, 2)
    f2 = (c2 + b2 * S2_alpha + a2 * S2_beta + S2_alpha * S2_beta)&(2-1)

    return f1, f2

# 生成随机值
# 初始化时生成，不属于offline
def mul_generate_random_z2(shape):
    a = np.random.randint(0,2,shape)
    b = np.random.randint(0,2,shape)
    c = a * b 
    #1
    a1 = np.random.randint(0,2,shape)
    a2 = (a-a1)&(2-1)
    b1 = np.random.randint(0,2,shape)
    b2 = (b-b1)&(2-1)
    c1 = np.random.randint(0,2,shape)
    c2 = (c-c1)&(2-1)
    return a1,a2,b1,b2,c1,c2

def SecXor_z2(u1, v1, u2, v2):
    f1 = (u1+v1)&(2-1)
    f2 = (u2+v2)&(2-1)
    return f1, f2

def SecOr_z2(u1, v1, u2, v2):
    m1, m2 = SecMul_z2(u1, v1, u2, v2)
    f1 = (u1+v1-m1)&(2-1)
    f2 = (u2+v2-m2)&(2-1)
    return f1, f2

# 总共只需要传递2L bit的数据
def SecMSB_z2(u1, u2, bit_length=32):
    # u1 = [0,1,0,1,1] 是整数的二进制比特序列的秘密分享
    # u2 = [0,1,0,0,0]
    L = bit_length
    u1_shape = u1.shape
    u_ori_shape = list(u1_shape)[:-1]
    t_shape = u_ori_shape.copy()
    t_shape.append(L+1)

    t1 = np.zeros(t_shape).astype(np.int) # t1=[input_shape, L+1]
    t2 = np.zeros(t_shape).astype(np.int)
    f1 = np.zeros(u1_shape).astype(np.int)
    f2 = np.zeros(u1_shape).astype(np.int)

    # 初始化zeta
    zeta1 = np.zeros(u_ori_shape).astype(np.int)
    zeta2 = np.zeros(u_ori_shape).astype(np.int)

    # offline
    # 初始化t1, t2（t1+t2=1）均为Z_2
    t1[...,0] = np.random.randint(0,2)
    t2[...,0] = 1-t1[...,0]
    
    for i in range(0, L):
        t1[...,i+1], t2[...,i+1] = SecMul_z2(t1[...,i], 1-u1[...,i], t2[...,i], -u2[...,i]&(2-1))
    for i in range(0, L):
        # S1
        f1[...,i] = (t1[...,i]-t1[...,i+1])&(2-1)
        # S2
        f2[...,i] = (t2[...,i]-t2[...,i+1])&(2-1)
    zeta1 = np.sum(f1, axis=-1)&(2-1)
    zeta2 = np.sum(f2, axis=-1)&(2-1)

    return f1,zeta1,f2,zeta2
    
'''采用boolean sharing做比较'''
def SecCmp_z2(u1,v1,u2,v2, bit_length=32):
    L = bit_length
    # 将本地的值转为二进制数组
    # S1
    a = u1-v1
    a_bin = secp.DFC(a, L)
    a_shape = a.shape
    a_bin_shape = a_bin.shape
    # S2
    b = v2-u2
    b_bin = secp.DFC(b, L)

    c1 = np.zeros(a_bin_shape).astype(np.int)
    c2 = np.zeros(a_bin_shape).astype(np.int)
    e1 = np.zeros(a_bin_shape).astype(np.int)
    e2 = np.zeros(a_bin_shape).astype(np.int)
    xi1 = np.zeros(a_shape).astype(np.int)
    xi2 = np.zeros(a_shape).astype(np.int)


    c1, c2= SecXor_z2(a_bin, b_bin, np.zeros(a_bin_shape).astype(np.int), np.zeros(a_bin_shape).astype(np.int))

    d1,zeta1,d2,zeta2= SecMSB_z2(c1,c2,L)
    e1, e2= SecMul_z2(a_bin, d1, np.zeros(a_bin_shape).astype(np.int), d2)

    xi1 = np.sum(e1, axis=-1)&(2-1)
    xi2 = np.sum(e2, axis=-1)&(2-1)


    a_bin_child_shape = a_bin[...,0].shape
    iota1, iota2= SecOr_z2(a_bin[...,0], b_bin[...,0], np.zeros(a_bin_child_shape).astype(np.int), np.zeros(a_bin_child_shape).astype(np.int))
    nu1, nu2= SecXor_z2(iota1, xi1, iota2, xi2)
    f1, f2= SecMul_z2(nu1, zeta1, nu2, zeta2)

    return f1, f2

