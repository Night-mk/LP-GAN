'''
    secure_protocols_2.py 构建Z2上的安全协议 
    计时版本
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
    start_offline = time.time()
    # a1,a2,b1,b2,c1,c2 = mul_generate_random_z2(input_shape)
    a1,a2,b1,b2,c1,c2 = mul_generate_random_z2(1)
    end_offline = time.time()

    start_online = time.time()
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
    f1 = (c1 + b1 * S1_alpha + a1 * S1_beta)& (2-1)
    # S2
    S2_alpha = alpha1 + alpha2
    S2_beta = beta1 + beta2
    # f2 = np.mod(np.mod(np.mod(c2 + b2 * S2_alpha, 2) + a2 * S2_beta, 2) + S2_alpha * S2_beta, 2)
    f2 = (c2 + b2 * S2_alpha + a2 * S2_beta + S2_alpha * S2_beta)&(2-1)
    end_online = time.time()

    return f1, f2, (end_offline-start_offline)*1000, (end_online-start_online)*1000

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
    # print('a: ',a)
    # print('b: ',b)
    # print('c: ',c)
    return a1,a2,b1,b2,c1,c2

def SecXor_z2(u1, v1, u2, v2):
    start_online = time.time()
    f1 = (u1+v1)&(2-1)
    f2 = (u2+v2)&(2-1)
    end_online = time.time()
    return f1, f2, (end_online-start_online)*1000

def SecOr_z2(u1, v1, u2, v2):
    m1, m2, offline, online = SecMul_z2(u1, v1, u2, v2)
    start_online = time.time()
    f1 = (u1+v1-m1)&1
    f2 = (u2+v2-m2)&1
    end_online = time.time()
    return f1, f2, offline, online+(end_online-start_online)*1000

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
    start_offline = time.time()
    t1[...,0] = np.random.randint(0,2)
    t2[...,0] = 1-t1[...,0]
    end_offline = time.time()
    
    online = 0
    offline = 0
    for i in range(0, L):
        t1[...,i+1], t2[...,i+1], offline_mul, online_mul = SecMul_z2(t1[...,i], 1-u1[...,i], t2[...,i], -u2[...,i]&(2-1))
        offline+=offline_mul
        online+=online_mul
    # print('t: ', (t1+t2)&(2-1))
    start_online = time.time()
    for i in range(0, L):
        # S1
        f1[...,i] = (t1[...,i]-t1[...,i+1])&(2-1)
        # S2
        f2[...,i] = (t2[...,i]-t2[...,i+1])&(2-1)
    zeta1 = np.sum(f1, axis=-1)&(2-1)
    zeta2 = np.sum(f2, axis=-1)&(2-1)
    end_online = time.time()

    return f1,zeta1,f2,zeta2, offline+(end_offline-start_offline)*1000, online+(end_online-start_online)*1000
    

'''对0比较算法的优化'''
# if f1+f2 = 1, x<0
# if f1+f2 = 0, x>=0
def SecCmp_tozero(u1,v1,u2,v2, bit_length=32):
    L = bit_length
    # 将本地的值转为二进制数组
    # start_online = time.time()
    # S1
    a = u1-v1
    a_bin = secp.DFC(a, L)
    a_shape = a.shape
    a_bin_shape = a_bin.shape
    # S2
    b = u2-v2
    b_bin = secp.DFC(b, L)

    print('a_bin: ', a_bin)
    print('b_bin: ', b_bin)
    d1,zeta1,d2,zeta2, offline_msb, online_msb = SecMSB_z2(a_bin,b_bin,L)
    print('d1+d2: ',(d1+d2)&(2-1))
    a_bin_child_shape = a_bin[...,0].shape
    f1, f2, online_xor = SecXor_z2(d1[...,0], d2[...,0], np.zeros(a_bin_child_shape).astype(np.int), np.zeros(a_bin_child_shape).astype(np.int))

    return f1, f2


'''采用boolean sharing做比较'''
def SecCmp_z2(u1,v1,u2,v2, bit_length=32):
    L = bit_length
    # 将本地的值转为二进制数组
    start_dfc = time.time()
    # S1
    a = u1-v1
    a_bin = secp.DFC(a, L)
    a_shape = a.shape
    a_bin_shape = a_bin.shape
    # S2
    b = v2-u2
    b_bin = secp.DFC(b, L)
    end_dfc = time.time()

    c1 = np.zeros(a_bin_shape).astype(np.int)
    c2 = np.zeros(a_bin_shape).astype(np.int)
    e1 = np.zeros(a_bin_shape).astype(np.int)
    e2 = np.zeros(a_bin_shape).astype(np.int)
    xi1 = np.zeros(a_shape).astype(np.int)
    xi2 = np.zeros(a_shape).astype(np.int)

    # print('a: ', a)
    # print('b: ', b)
    # print('a_bin: ',a_bin)
    # print('b_bin: ',b_bin)
    
    # start_online = time.time()
    c1, c2, online_xor = SecXor_z2(a_bin, b_bin, np.zeros(a_bin_shape).astype(np.int), np.zeros(a_bin_shape).astype(np.int))
    # print('c_bin: ',np.mod(c1+c2, 2))
    d1,zeta1,d2,zeta2, offline_msb, online_msb = SecMSB_z2(c1,c2,L)
    # print('d_bin: ',np.mod(d1+d2, 2))
    # print('z_bin: ',np.mod(zeta1+zeta2, 2))
    e1, e2, offline_mul, online_mul = SecMul_z2(a_bin, d1, np.zeros(a_bin_shape).astype(np.int), d2)
    # print('e_bin: ',np.mod(e1+e2, 2))
    # print('e1: ',e1)
    # print('e2: ',e2)

    xi1 = np.sum(e1, axis=-1)%2
    xi2 = np.sum(e2, axis=-1)%2
    # print('xi1: ',xi1)
    # print('xi2: ',xi2)
    # print('xi: ',np.mod(xi1+xi2, 2))


    a_bin_child_shape = a_bin[...,0].shape
    # print('a_bin_child_shape: ',a_bin_child_shape)
    iota1, iota2, offline_or, online_or = SecOr_z2(a_bin[...,0], b_bin[...,0], np.zeros(a_bin_child_shape).astype(np.int), np.zeros(a_bin_child_shape).astype(np.int))
    # print('iota: ',np.mod(iota1+iota2, 2))
    nu1, nu2, online_xor1 = SecXor_z2(iota1, xi1, iota2, xi2)
    f1, f2, offline_mul1, online_mul1 = SecMul_z2(nu1, zeta1, nu2, zeta2)
    
    # end_online = time.time()
    # print('offline_msb: ', offline_msb)
    # print('offline_mul: ', offline_mul)
    # print('offline_mul1: ', offline_mul1)
    # print('offline_or: ', offline_or)
    
    # print('real offline: ', offline_msb+offline_mul+offline_mul1+offline_or+(end_offline-start_offline)*1000)
    # print('offline DFC: ', (end_offline-start_offline)*1000)
    # print('offline: ', offline_msb+offline_mul+offline_mul1+offline_or)
    # print('online: ', online_msb+online_mul+online_or+online_mul1+online_xor+online_xor1)
    # print('online: ', online_msb+online_mul+online_or+online_mul1)
    online_time_consume = online_msb+online_mul+online_or+online_mul1
    offline_time_consume = offline_msb+offline_mul+offline_mul1+offline_or
    dfc_time = (end_dfc-start_dfc)/60
    # offline_time_consume = offline_msb+offline_mul+offline_mul1+offline_or
    # print('online real: ', (end_online-start_online)*1000)
    # print('dfc: ', dfc_time)
    # online_time_consume = (end_online-start_online)*1000

    return f1, f2, offline_time_consume, online_time_consume, dfc_time
    

'''将Z2数域的数据转化到Z_2**(bit-length)数域'''
def SecFieldC(u1, u2, bit_length=32):
    # 生成2**bit_length数域上的随机数,r1+r2=0
    # low = -2**(bit_length//4)
    # high = 2**(bit_length//4)
    low = 0
    high = 2
    input_shape = u1.shape
    r1 = np.random.randint(low, high, input_shape)
    r2 = 0-r1

    m1, m2 = secp.SecMul_matrix(u1, u2, 0, 0, bit_length)
    f1 = r1+u1-2*m1
    f2 = r2+u2-2*m2
    return f1, f2


def test_secmul():
    shape = 1
    u = np.random.randint(0,2,shape)
    u1 = np.random.randint(0,2,shape)
    u2 = np.mod(u-u1, 2)

    v = np.random.randint(0,2,shape)
    v1 = np.random.randint(0,2,shape)
    v2 = np.mod(v-v1, 2)
    
    start_time = time.time()
    f1, f2, off, on = SecMul_z2(u1,v1,u2,v2)
    end_time = time.time()
    print('u: ',u, u1, u2)
    print('v: ',v, v1, v2)
    print('f: ',np.mod(f1+f2,2))
    print('time: ',end_time-start_time)
    # print('u1: ',u1)
    # print('u2: ',u2)

def test_seccmp_z2():
    batchsize = 1
    num = 1000
    bit_length = 32
    u = np.random.randn(batchsize,num)
    # v = np.random.randn(batchsize,num)
    # print('u: \n', u)
    # print('v: \n', v)
    # u1 = np.random.randn(batchsize,num)
    # u2 = u-u1
    # v1 = np.random.randn(batchsize,num)
    # v2 = v-v1
    v = np.zeros((batchsize,num))
    u1 = np.random.randn(batchsize,num)
    u2 = u-u1
    v1 = np.zeros((batchsize,num))
    v2 = v-v1

    # comp_index = np.zeros((batchsize, num))
    # for i, item in enumerate(u):
    #     if item>v[i]: comp_index[i]=1
    #     else: comp_index[i]=0

    f1, f2, offline_time, online_time = SecCmp_z2(u1, v1, u2, v2, bit_length)
    # print('f: ', np.mod(f1+f2, 2))
    # print('c: ',comp_index)
    # print('error: ',np.sum(np.mod(f1+f2, 2)-comp_index))
    return offline_time, online_time

def test_seccmp_tozero():
    batchsize = 1
    num = 1
    bit_length = 32
    u = np.random.randn(batchsize,num)
    v = np.zeros((batchsize,num))
    print('u: \n', u)
    print('v: \n', v)
    u1 = np.random.randn(batchsize,num)
    u2 = u-u1
    v1 = np.zeros((batchsize,num))
    v2 = v-v1

    comp_index = np.zeros((batchsize, num))
    for i, item in enumerate(u):
        if item>=v[i]: comp_index[i]=0
        else: comp_index[i]=1

    f1, f2 = SecCmp_tozero(u1, v1, u2, v2, bit_length)
    print('f: ', np.mod(f1+f2, 2))
    print('c: ',comp_index)
    # print('error: ',np.sum(np.mod(f1+f2, 2)-comp_index))
    # return offline_time, online_time

def test_field_conversion():
    num = 10
    u1 = np.random.randint(0,2,size=num)
    u2 = np.random.randint(0,2,size=num)
    # print('u1: ', u1)
    # print('u2: ', u2)
    # print('u1+u2: ', (u1+u2)&(2-1))
    f1, f2 = SecFieldC(u1, u2, 32)
    # print('f1: ', f1)
    # print('f2: ', f2)
    print('f1+f2: ', f1+f2)
    # print('err: \n', np.sum(f1+f2-(u1+u2)&(2-1)))

    
if __name__ == '__main__':
    test_secmul()
    '''测试比较安全算法的平均时间'''
    '''
    avg_time_offline = 0
    avg_time_online = 0
    round_num = 10
    for i in range(round_num):
        time_offline, time_online = test_seccmp_z2()
        avg_time_offline+=time_offline
        avg_time_online+=time_online
    print('avg_time_offline: ', avg_time_offline/round_num)
    print('avg_time_online: ', avg_time_online/round_num)
    print('avg_time_sum: ', avg_time_offline/round_num+avg_time_online/round_num)
    '''
    # test_seccmp_tozero()
    # test_field_conversion()