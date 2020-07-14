'''
    secure_protocols.py 构建LP-GAN使用的基础安全协议
'''

import random
import numpy as np
from numpy import ndarray
import math
import time

################################  线性计算  ################################

'''安全加法协议 SecAdd'''
def SecAdd(u1,v1,u2,v2):
	f1 = u1 + v1
	f2 = u2 + v2
	return f1,f2

'''安全乘法协议 SecMul'''
def SecMul(u1,v1,u2,v2, bit_length=32):
    a1,a2,b1,b2,c1,c2 = mul_generate_random()

    alpha1 = u1 - a1
    beta1 = v1 - b1
    alpha2 = u2 - a2
    beta2 = v2 - b2

    S1_alpha = alpha1 + alpha2
    S1_beta = beta1 + beta2
    f1 = c1 + b1 * S1_alpha + a1 * S1_beta

    S2_alpha = alpha1 + alpha2
    S2_beta = beta1 + beta2
    f2 = c2 + b2 * S2_alpha + a2 * S2_beta + S2_alpha * S2_beta

    return f1,f2

# 生成随机值
def mul_generate_random(shape, bit_length=32):
    # range_down = -2**(bit_length//4)
    # range_up = 2**(bit_length//4)
    range_down = 1
    range_up = 5
    a = np.random.randint(range_down,range_up,shape)
    b = np.random.randint(range_down,range_up,shape)
    c = a * b
    a1 = np.random.randint(range_down,range_up,shape)
    a2 = a - a1
    b1 = np.random.randint(range_down,range_up,shape)
    b2 = b - b1
    c1 = np.random.randint(range_down,range_up,shape)
    c2 = c - c1

    return a1,a2,b1,b2,c1,c2

'''安全乘法协议（对应元素相乘）SecMul_matrix'''
def SecMul_matrix(u1,v1,u2,v2, bit_length=32):
    input_shape = u1.shape
    a1,a2,b1,b2,c1,c2 = mul_generate_random(input_shape, bit_length)
    # print((a1+a2)*(b1+b2),",",(c1+c2))
    # start_online = time.time()
    # S1
    alpha1 = u1 - a1
    # print('alpha1: ', alpha1)
    beta1 = v1 - b1 
    # S2
    alpha2 = u2 - a2
    beta2 = v2 - b2
    # S1
    S1_alpha = alpha1 + alpha2
    S1_beta = beta1 + beta2
    # print('beta1,beta2: ',beta1, beta2)
    # f1 = (c1 + b1 * S1_alpha + a1 * S1_beta ) % (2**bit_length)
    f1 = c1 + b1 * S1_alpha + a1 * S1_beta
    # S2
    S2_alpha = alpha1 + alpha2
    S2_beta = beta1 + beta2
    # f2 = (c2 + b2 * S2_alpha + a2 * S2_beta + S2_alpha * S2_beta ) % (2**bit_length)
    f2 = c2 + b2 * S2_alpha + a2 * S2_beta + S2_alpha * S2_beta
    # end_online = time.time()

    # f1和f2的数量级检测，若超出2**(bit_length-2)，则将一方设置为1 [check时间还可以]
    # f1, f2 = check_mul_magnitude(f1, f2, bit_length)

    # return f1,f2,(end_online-start_online)*1000
    return f1,f2

def check_mul_magnitude(f1,f2, bit_length):
    f1_copy = f1.copy()
    f1_copy[f1>2**(bit_length-1)]=1
    f1_copy[f1<-2**(bit_length-1)]=1
    if not (f1==f1_copy).all():
        f2 = f1+f2-1
    f2_copy = f2.copy()
    f2_copy[f2>2**(bit_length-1)]=1
    f2_copy[f2<-2**(bit_length-1)]=1
    if not (f2==f2_copy).all(): # 判断是否修改过
        f1 = f1+f2-1
    return f1, f2


'''安全乘法协议（矩阵相乘）SecMul_dot'''
def SecMul_dot(u1,v1,u2,v2, bit_length=32):
    # print(u1.shape) # (5,25)
    # print(u1.shape[0]) # 5
    a1,a2,b1,b2,c1,c2 = mul_generate_random(1)
    # u_shape = u1.shape
    # v_shape = v1.shape
    # a1,a2,b1,b2,c1,c2 = mul_generate_random_dot(u_shape, v_shape, bit_length)
    # print('c1.shape: ',c1.shape)
    # print('a1.shape: ',a1.shape)
    # print('b1.shape: ',b1.shape)
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
    # print('S1_alpha.shape: ',S1_alpha.shape)
    # print('S1_alpha.shape: ',type(b1))
    # print('S1_beta.shape: ',S1_beta.shape)
    # print('S1_beta.shape: ',type(a1))
    f1 = c1*u1.shape[1] + matrix_add(b1*S1_alpha, a1*S1_beta) # c.shape=1
    # f1 = c1 + matrix_add(b1.T*S1_alpha, a1.T*S1_beta) # c.shape=(a dot b).shape
    # S2
    S2_alpha = alpha1 + alpha2
    S2_beta = beta1 + beta2
    f2 = c2*u1.shape[1] + matrix_add(b2*S2_alpha, a2*S2_beta) + np.dot(S2_alpha,S2_beta) # c.shape=1
    # f2 = c2 + matrix_add(b2.T*S2_alpha, a2.T*S2_beta) + np.dot(S2_alpha,S2_beta)
    end_online = time.time()
    print('time consume online: \n', (end_online-start_online)*1000)

    # f1和f2的数量级检测，若超出2**(bit_length-2)，则将一方设置为1 [check时间还可以]
    # f1, f2 = check_mul_magnitude(f1, f2, bit_length)

    # return f1,f2,(end_online-start_online)*1000
    return f1,f2

# 对两个二维矩阵按矩阵相乘的方式相加
def matrix_add(a, b):
    a_shape = a.shape
    b_shape = b.shape
    a_scale = np.tile(a, b_shape[1])
    b_scale = np.tile(b.T.reshape(-1), (a_shape[0],1))
    # print('a_scale.shape: ',a_scale.shape)
    # print('b_scale.shape: ',b_scale.shape)

    a_b_scale = a_scale+b_scale
    a_b_resize = np.zeros((a_shape[0], b_shape[1]))
    range_num = a_shape[1]*b_shape[1]
    for i in range(0, range_num, a_shape[1]):
        a_b_resize[:, i//a_shape[1]] = np.sum(a_b_scale[:, i:i+a_shape[1]], axis=1)
    return a_b_resize

def mul_generate_random_dot(u_shape, v_shape, bit_length):
    # range_down = -2**(bit_length//2)
    # range_up = 2**(bit_length//2)
    range_down = 0
    range_up = 2
    a = np.random.randint(range_down,range_up,u_shape)
    b = np.random.randint(range_down,range_up,v_shape)
    c = np.dot(a,b)
    a1 = np.random.randint(range_down,range_up,u_shape)
    a2 = a - a1
    b1 = np.random.randint(range_down,range_up,v_shape)
    b2 = b - b1
    c1 = np.random.randint(range_down,range_up,c.shape)
    c2 = c - c1

    return a1,a2,b1,b2,c1,c2
    

'''安全矩阵乘法 SecMul_dot_2，实现加法秘密共享对矩阵相乘，并生成足够数量的随机数对数据进行加密'''
def SecMul_dot_2(u1,v1,u2,v2, bit_length=32):
    u_shape = u1.shape
    v_shape = v1.shape
    a1,a2,b1,b2,c1,c2 = mul_generate_random_dot(u_shape, v_shape, bit_length)
    alpha1 = u1 - a1
    beta1 = v1 - b1 
    # S2
    alpha2 = u2 - a2
    beta2 = v2 - b2
    # S1
    S1_alpha = alpha1 + alpha2
    S1_beta = beta1 + beta2
    f1 = c1 + np.dot(u1, S1_beta) + np.dot(S1_alpha, v1)

    S2_alpha = alpha1 + alpha2
    S2_beta = beta1 + beta2
    f2 = c2 + np.dot(u2, S2_beta) + np.dot(S2_alpha, v2) - np.dot(S2_alpha,S2_beta)

    return f1, f2
    
def SecMul_dot_3(u1,v1,u2,v2, bit_length=32):
    u_shape = u1.shape
    v_shape = v1.shape
    a1,a2,b1,b2,c1,c2 = mul_generate_random_dot(u_shape, v_shape, bit_length)
    alpha1 = u1 - a1
    beta1 = v1 - b1 
    # S2
    alpha2 = u2 - a2
    beta2 = v2 - b2
    # S1
    S1_alpha = alpha1 + alpha2
    S1_beta = beta1 + beta2
    f1 = c1 + np.dot(a1, S1_beta) + np.dot(S1_alpha, b1)

    S2_alpha = alpha1 + alpha2
    S2_beta = beta1 + beta2
    f2 = c2 + np.dot(a2, S2_beta) + np.dot(S2_alpha, b2) + np.dot(S2_alpha,S2_beta)

    return f1, f2


################################  非线性计算  ################################
'''
    规定bit标准长度为32bit, 64bit
    安全比较算法需要将输入先转换成整数int32，再转换成二进制进行比较，可以特殊判断0（无需进行转换）
    由于机器学习的数据通常是浮点数，并且数据大小不会是个大数（都会Normalize到某个空间进行计算），所以可以将浮点数直接乘2^16或2^32（整数小数对半开）转换为大整数
    
    ## 注：安全比较算法不会影响计算精度！！因为比完了，只是确定哪些数可以保留，这种
'''

# 精度设置为小数点后2^(bit_length//2)位
def float2int(num, bit_length=32):
    return np.array(num*(2**(bit_length//2))).astype(np.int64)

'''整数转二进制（指定长度为bit_length位）'''
def int2bin(num, bit_length=32):
    # 传入的num一定是个整数ndarray
    # num的范围是[-2**31~2**31]
    bl = bit_length 
    # 获取绝对值和符号位
    s = np.zeros(num.shape).astype(np.int) #一定要转为int类型
    s[num<0] = 1
    # 函数向量化,可以使用np.frompyfunc对numpy的所有数据进行处理，并返回对应类型（ndarray）的数据 np.frompyfunc(func_x, 1, 1)
    # func_x指的是你要使用的函数，第二个参数为func中的参数个数，第三个参数为func中的返回值的个数
    # func_ = np.frompyfunc(numpy2bin, 3, 1)
    # num_bin = func_(num_abs, bl-1, s) # 返回了ndarray类型的数据！

    num_bin_shape = list(num.shape)
    num_bin_shape.append(bl)
    s = s.astype(np.str).reshape(-1)
    num = abs(num.reshape(-1))
    num_str = ''

    # start_extract=time.time()
    for index, num_i in enumerate(num):
        num_str += s[index]+np.binary_repr(num_i, bl-1)
    end_extract=time.time()
    num_bin = np.array(list(num_str)).astype(np.int).reshape(num_bin_shape)
    # num_bin = num_bin.astype(np.int).reshape(num_bin_shape)
    # end_reshape=time.time()
    # print('extract consume: ',(end_extract-start_extract)*1000)
    # print('extract+reshape consume: ',(end_reshape-start_extract)*1000)
    return num_bin

'''浮点数转为二进制表示(字符串)'''
def float2bin(num, bit_length=32):
    int_data = float2int(num, bit_length)
    # print('int: ', int_data)
    return int2bin(int_data, bit_length)

## 将字符串数组转换为numpy数组
# def str2arr(str_num, arr_shape):
#     str_num = str_num.reshape(-1)
#     str_all_num = ''
#     for i in str_num: 
#         str_all_num +=i # 字符串是'100...01'第0位是最高位
#     return np.array(list(str_all_num)).astype(np.int).reshape(arr_shape)


'''数据格式转换协议 Data Format Conversion DFC'''
# 将实数转换成二进制形式的序列
def DFC(num, bit_length=32):
    return float2bin(num, bit_length)

# 计算某个数n规划到[1/4,1/2]和[-1/4,-1/8]需要乘的2^p中的p
# 计算出来的是-p，所以要在SecRC里取小的数
def scaled_p(u):
    p=0
    num = abs(u)
    if u>0: # u1 in [1/4, 1/2)
        if num>=0.5:
            while num>=0.5:
                num /=2
                p-=1
        elif num<0.25:
            while num<0.25:
                num *=2
                p+=1
    # if u>0: # u1 in [1/2, 1)
    #     if num>=1:
    #         while num>=1:
    #             num /=2
    #             p-=1
    #     elif num<0.5:
    #         while num<0.5:
    #             num *=2
    #             p+=1
    if u<0:
        if num>0.25:
            while num>0.25:
                num /=2
                p-=1
        elif num<=0.125:
            while num<=0.125:
                num *=2
                p+=1
    return p

'''Secure Range Conversion SRC 安全区间转换协议'''
def SecRC(u1, u2, bit_length=32):
    # 判断输入u1内的数据是否大于0
    u_shape = u1.shape
    u1_line = u1.reshape(-1)
    u2_line = u2.reshape(-1)
    p = np.zeros(u1_line.shape)
    p1 = np.zeros(u1_line.shape)
    p2 = np.zeros(u2_line.shape)

    # S1, S2 p=max(p1,p2)
    for index, item in enumerate(u1_line):
        p1[index] = scaled_p(item)
        p2[index] = scaled_p(u2_line[index])
        if p1[index]<=p2[index]:
            p[index] = p1[index]
        else:
            p[index] = p2[index]

    # print('u1: \n',u1)
    # print('u1_line: \n',u1_line)

    p = p.reshape(u_shape)
    p1 = p1.reshape(u_shape)
    p2 = p2.reshape(u_shape)
    # print('2**-p: \n',2**(-p))
    
    m1 = u1*(2**p)
    m2 = u2*(2**p)
    return m1, m2, -p
    # S2
    # for index, item in enumerate(u2_line):
    #     p2[index] = scaled_p(item)

    # # 计算p=max(p1,p2)
    # for index, item in enumerate(p):
    #     if p1[index]>=p2[index]:
    #         item = p1[index]
    #     else:
    #         item = p2[index]
    
'''安全平方根算法 SSqrt'''
## shares太大(>10)会产生精度太低
def SSqrt(u1, u2, iter_num, inverse_required=False, bit_length=32, length_required=False):
    input_shape = u1.shape
    # 转换成bit-length长度的整数形式进行运算（整数长度：2**(bit-length//2),小数长度：2**(bit-length//2)）
    if length_required:
        u1 = np.floor_divide(u1*(2**(bit_length//2)), 1)
        u2 = np.floor_divide(u2*(2**(bit_length//2)), 1)
    ## 开始协议
    ## 重新分配shares
    u = u1+u2
    # u1 = np.ones(input_shape)
    u1 = u/3
    u2 = u-u1
    # S1 and S2
    m1, m2, p = SecRC(u1, u2, bit_length)
    # print('m1+m2 * 2**p', (m1+m2)*2**p)
    # print('m1', m1)
    # print('m1', m2)
    # print('p', p)
    # print('m1+m2 * 2**p', (m1+m2)*2**p)
    alpha = -0.8099868542
    beta = 1.787727479
    # S1
    y1 = alpha*m1+beta
    h1 = 0.5*y1
    # S2 
    y2 = alpha*m2
    h2 = 0.5*y2
    g1, g2 = SecMul_matrix(m1, y1, m2, y2, bit_length)

    i = 0
    while i < iter_num: # 迭代计算g,h
        r1, r2 = SecMul_matrix(g1, h1, g2, h2, bit_length)
        r1 = 1.5-r1
        r2 = -r2
        g1, g2 = SecMul_matrix(g1, r1, g2, r2, bit_length)
        h1, h2 = SecMul_matrix(h1, r1, h2, r2, bit_length)
        i +=1
        
    a = np.ones(input_shape)
    a[p%2!=0] = math.sqrt(2)
    # print('p: ', p)
    # print('a: ', a)
    p_ = np.floor_divide(p,2)
    # print('p_: ', p_)
    if inverse_required: # 判断是否要计算1/sqrt(u)
        f1 = 2/(a*(2**p_))*h1
        f2 = 2/(a*(2**p_))*h2
    else:
        f1 = a*(2**p_)*g1
        f2 = a*(2**p_)*g2
    
    # 将整数恢复成实数(如果需要)
    if length_required:
        f1 = f1/(2**(bit_length//2))
        f2 = f2/(2**(bit_length//2))

    return f1, f2


'''安全平方算法 SSq'''
def SSq(u1, u2, bit_length=32):
    m1, m2 = SecMul_matrix(2*u1, u2, 0, 0, bit_length) 
    # print('mul result: ',m1+m2)
    f1 = u1**2 + m1
    f2 = u2**2 + m2

    return f1, f2

# 将原始值scale到(1,2)之间，保证收敛， u=1+e^x（明文狗头orz）
def sec_scale(u1, u2):
    input_shape = u1.shape
    u1_line = u1.reshape(-1)
    u2_line = u2.reshape(-1)
    u = u1_line+u2_line # 1维
    # print('u: first', u)
    p = np.zeros(u1_line.shape)
    p1 = np.ones(u1_line.shape)
    eps = 1e-1
    acc_low = 1e-30
    for index, item in enumerate(u):
        # 精度不够要计算log(0)时，直接设置u=1e-16
        if u[index]<acc_low:
            u[index]=acc_low
            # break
        if u[index]>2:
            while u[index]>2:
                u[index] /=2
                p[index] += 1
        if u[index]<1 and u[index]>=acc_low:
            while u[index]<1:
                u[index] *=2
                p[index] -=1
        if u[index]>1.1:
            p1[index] = u[index]-eps
            u[index] /= u[index]-eps # 将数据u, scale到1附近

    u2 = (u-u1_line).reshape(input_shape)
    u1 = u1_line.reshape(input_shape)
    p = p.reshape(input_shape)
    p1 = p1.reshape(input_shape)
    return u1, u2, p, p1


'''安全log算法 SecLog 利用scale可以达到收敛[1<x<1.1]最佳'''
def SecLog(u1, u2, tau, bit_length=32):
    # print('log input: ',u1+u2)
    m1, m2, p, p1 = sec_scale(u1, u2)
    f1 = m1-1
    f2 = m2
    # 计算x,x^2,x^3
    z1 = m1-1
    z2 = m2
    iter_num = 1
    
    while iter_num<=tau:
        iter_num+=1
        # x_(n+1)=x_n*x
        z1, z2 = SecMul_matrix(z1, m1-1, z2, m2, bit_length)
        coefficient = ((-1)**(iter_num+1))*(1/iter_num)
        # print(coefficient)
        f1 = f1 + coefficient*z1
        f2 = f2 + coefficient*z2
    f1 = f1+np.log(2**p)+np.log(p1)
    return f1, f2

'''测速版删去scale'''
def SecLog_speed(u1, u2, tau, bit_length=32):
    # print('log input: ',u1+u2)
    f1 = u1-1
    f2 = u2
    # 计算x,x^2,x^3
    z1 = u1-1
    z2 = u2
    iter_num = 1
    
    while iter_num<=tau:
        iter_num+=1
        # x_(n+1)=x_n*x
        z1, z2 = SecMul_matrix(z1, u1-1, z2, u2, bit_length)
        coefficient = ((-1)**(iter_num+1))*(1/iter_num)
        # print(coefficient)
        f1 = f1 + coefficient*z1
        f2 = f2 + coefficient*z2
    # f1 = f1+np.log(2**p)+np.log(p1)
    return f1, f2

def SecLog_plaintxt(u, tau):
    x=u
    f=u.copy()
    iter_num = 1
    
    while iter_num<=tau:
        iter_num+=1
        # x_(n+1)=x_n*x
        x = x*u
        coefficient = ((-1)**(iter_num+1))*(1/iter_num)
        print('coe: ',coefficient)
        f += coefficient*x
        print('f_iter: ', f)
    return f

'''安全log算法，平方迭代加速版'''
def SecLog_square(u1, u2, tau, bit_length=32):
    # 对输入规约空间
    u1, u2, p = SecRC(u1, u2, bit_length)
    print('p: ',p)
    print('u1+u2: ',u1+u2)
    # 计算y=(x-1)/(x+1)
    tau_inv = 10
    initial_inv = 1e-2
    x1_inv, x2_inv = SecInv(u1+1, u2, tau_inv, initial_inv, bit_length)
    # print('inv_sec: ', x1_inv+x2_inv)
    # print('inv: ', 1/(u1+u2+1))
    y1, y2 = SecMul_matrix(u1-1, x1_inv, u2, x2_inv, bit_length)
    # print('y: ', (u1+u2-1)/(u1+u2+1))
    # print('y_sec: ', y1+y2)

    z1 = y1
    z2 = y2
    f1 = 0
    f2 = 1
    # 迭代计算（log=2y*(1+(1/3)y^2+(1/5)y^4+...)）
    iter_num = 0
    while iter_num<tau:
        iter_num+=1
        coefficient = 1/(2*iter_num-1)
        print('coef: ', coefficient)
        if iter_num==1:
            continue
        else:
            z1, z2 = SecMul_matrix(z1, z1, z2, z2, bit_length)
            f1 += coefficient*z1
            f2 += coefficient*z2
    
    f1, f2 = SecMul_matrix(2*y1, f1, 2*y2, f2, bit_length)
    return f1+np.log(2**p), f2
        

'''精度比较版'''
def SecLog_acc(u1, u2, tau, bit_length=32):
    f1 = u1
    f2 = u2
    # 计算x,x^2,x^3
    z1 = u1
    z2 = u2
    iter_num = 1
    acc = np.ones(u1.shape)*1e-10
    ori = np.log(u1+u2+1)
    
    while f1+f2-ori>acc:
        iter_num+=1
        # x_(n+1)=x_n*x
        z1, z2 = SecMul_matrix(z1, u1, z2, u2, bit_length)
        coefficient = ((-1)**(iter_num+1))*(1/iter_num)
        # print(coefficient)
        f1 = f1 + coefficient*z1
        f2 = f2 + coefficient*z2
    print('iter_num: ', iter_num)
    return f1, f2


'''安全倒数算法SecInv'''
def SecInv(u1, u2, tau, initial=1e-5, bit_length=32):
    # x0 = initial
    u = u1+u2
    u1 = u/3
    u2 = u-u1
    x0 = 1/(u1+u2)
    a1 = u1
    a2 = u2
    iter_num = 0
    g1 = (2-x0*a1)*x0
    g2 = -a2*x0*x0

    while iter_num<=tau:
        iter_num+=1
        y1, y2 = SecMul_matrix(a1, g1, a2, g2)
        y1 = 2-y1
        y2 = -y2
        g1, g2 = SecMul_matrix(g1, y1, g2, y2)
    f1 = g1
    f2 = g2
    return f1, f2

def SecInv_plaintxt(u, tau, bit_length=32):
    x0 = 1e-7
    iter_num = 0

    while iter_num<=tau:
        iter_num+=1
        x1 = x0*(2-u*x0)
        x0 = x1

    return x0


'''安全exp算法SecExp'''
def SecExp(u1, u2, bit_length=32):
    input_shape = u1.shape
    u = u1+u2
    # 重新分配，避免exp爆炸
    z1 = 1
    z2 = u-1
    k1 = np.exp(z1)
    k2 = np.exp(z2)
    f1, f2 = SecMul_matrix(k1, k2, np.zeros(input_shape), np.zeros(input_shape), bit_length)
    return f1, f2

def test_seclog():
    bit_length=64
    u = np.random.uniform(1,2**(bit_length//2), size=(5,5)).astype(np.float64)
    # u = 1.886*np.ones((1,1)).astype(np.float64)
    u1 = np.random.uniform(0,1, size=(5,5)).astype(np.float64)
    u2 = u-u1

    tau = 10
    f1, f2 = SecLog(u1, u2, tau, bit_length=bit_length)
    # f = SecLog_plaintxt(u-1, tau)
    # f1, f2 = SecLog_square(u1, u2, tau, bit_length=bit_length)
    # f1, f2 = SecLog_acc(u1, u2-1, 10, bit_length=bit_length)
    u_log = np.log(u)
    print('u: ', u)
    print('log_u: ', u_log)
    print('f1+f2: ', f1+f2)
    print('error: ',(f1+f2)-u_log)

def test_seclog_speed():
    bit_length = 64
    width = 512
    height = 100
    tau=10
    
    x_numpy = np.random.uniform(1,1000, size=(height,width)).astype(np.float64)
    x_numpy_1 = np.random.uniform(-1,1, size=(height,width)).astype(np.float64)
    x_numpy_2 = x_numpy - x_numpy_1

    test_num = 10
    time_avg = 0
    for i in range(test_num):
        start_time_sec = time.time()
        # out_1, out_2 = SecLog(x_numpy_1, x_numpy_2, tau, bit_length)
        out_1, out_2 = SecLog_speed(x_numpy_1, x_numpy_2, tau, bit_length)
        # out = np.log(x_numpy)
        # print('error sum: ', np.sum(out-out_1-out_2))
        # print('error sum: ', out-out_1-out_2)
        end_time_sec = time.time()
        # print('time: ', (end_time_sec-start_time_sec)*1000)
        # time_avg+=(end_time_sec-start_time_sec)*1000
        time_avg+=(end_time_sec-start_time_sec)*1000
    print('time avg sec: \n', time_avg/test_num)

def test_secinv():
    bit_length=64
    width = 10
    height = 1
    u = np.random.uniform(1,100, size=(height,width)).astype(np.float64)
    u1 = np.random.uniform(0,1, size=(height,width)).astype(np.float64)
    u2 = u-u1
    initial_num = 1e-3
    f1, f2 = SecInv(u1, u2, 10, initial_num, bit_length)
    u_inv_iter = SecInv_plaintxt(u, 10, bit_length)
    u_inv = 1/u
    print('u: ', u)
    # print('u_inv: ', u_inv)
    # print('u_inv_iter: ', u_inv_iter)
    # print('f1+f2: ', f1+f2)
    print('error: ',(f1+f2)-u_inv)


if __name__ == '__main__':
    # test_seclog()
    # test_secinv()
    test_seclog_speed()

