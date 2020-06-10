'''
    secure_protocols.py 构建LP-GAN使用的基础安全协议
'''

import random
import numpy as np
from numpy import ndarray
import math

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
def mul_generate_random():
    a = random.randint(10,20)
    b = random.randint(10,20)
    c = a * b 
    #1
    a1 = random.randint(5,a-5)
    a2 = a - a1
    b1 = random.randint(5,b-5)
    b2 = b - b1
    c1 = random.randint(5,c-5)
    c2 = c - c1

    return a1,a2,b1,b2,c1,c2

'''安全乘法协议（矩阵形式）SecMul_matrix'''
def SecMul_matrix(u1,v1,u2,v2, bit_length=32):
    a1,a2,b1,b2,c1,c2 = mul_generate_random()
    # print((a1+a2)*(b1+b2),",",(c1+c2))
    # S1
    alpha1 = u1 - a1
    beta1 = v1 - b1
    # S2
    alpha2 = u2 - a2
    beta2 = v2 - b2

    # S1
    S1_alpha = alpha1 + alpha2
    S1_beta = beta1 + beta2
    f1 = c1 + b1 * S1_alpha + a1 * S1_beta
    # S2
    S2_alpha = alpha1 + alpha2
    S2_beta = beta1 + beta2
    f2 = c2 + b2 * S2_alpha + a2 * S2_beta + S2_alpha * S2_beta

    # f1和f2的数量级检测，若超出2**(bit_length-2)，则将一方设置为1 [check时间还可以]
    f1, f2 = check_mul_magnitude(f1, f2, bit_length)

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

################################  非线性计算  ################################
'''
    安全或 SecOr（矩阵）
'''
def SecOr(u1,v1,u2,v2):
    m1,m2 = SecMul_matrix(-u1,v1,-u2,v2)
    f1 = u1+v1+m1
    f2 = u2+v2+m2
    return f1, f2

'''
    安全异或 SecXor（矩阵）
'''
def SecXor(u1,v1,u2,v2):
    m1,m2 = SecMul_matrix(-2*u1,v1,-2*u2,v2)
    f1 = u1+v1+m1
    f2 = u2+v2+m2
    return f1, f2


################################  安全比较算法  ################################
'''
    规定bit标准长度为32bit, 64bit
    安全比较算法需要将输入先转换成整数int32，再转换成二进制进行比较，可以特殊判断0（无需进行转换）
    由于机器学习的数据通常是浮点数，并且数据大小不会是个大数（都会Normalize到某个空间进行计算），所以可以将浮点数直接乘2^16或2^32（整数小数对半开）转换为大整数
    
    ## 注：安全比较算法不会影响计算精度！！因为比完了，只是确定哪些数可以保留，这种
'''

# 精度设置为小数点后2^(bit_length//2)位
def float2int(num, bit_length=32):
    return np.array(num*(2**(bit_length//2))).astype(np.int32)

# 使用np将整数转为指定长度的二进制数据（如果超出长度则不受width限制）
def numpy2bin(num, width, s):
    '''
        num: 正整数数据
        width: 转成二进制的位数 31,63
        s: 符号位
        返回值：整数的二进制字符串表示矩阵 i.e.['00010']
    '''
    limit_num = 2**width
    if num>limit_num: # 对溢出数据的处理,讲道理还不知道咋处理orz
        num = np.mod(num, limit_num)
        print('overflow')
    return str(s)+np.binary_repr(num, width)

'''整数转二进制（指定长度为bit_length位）'''
def int2bin(num, bit_length=32):
    # 传入的num一定是个整数ndarray
    # num的范围是[-2**31~2**31]
    bl = bit_length 
    # 获取绝对值和符号位
    s = np.zeros(num.shape).astype(np.int) #一定要转为int类型
    s[num<0] = 1
    num_abs = np.abs(num)
    # 函数向量化,可以使用np.frompyfunc对numpy的所有数据进行处理，并返回对应类型（ndarray）的数据 np.frompyfunc(func_x, 1, 1)
    # func_x指的是你要使用的函数，第二个参数为func中的参数个数，第三个参数为func中的返回值的个数
    func_ = np.frompyfunc(numpy2bin, 3, 1)
    num_bin = func_(num_abs, bl-1, s) # 返回了ndarray类型的数据！
    return num_bin

'''浮点数转为二进制表示'''
def float2bin(num, bit_length=32):
    int_data = float2int(num, bit_length)
    return int2bin(int_data, bit_length)
    
# 本地计算offline，初始化t1, t2
def initT1T2(t1, t2, bit_length=32):
    t1 = np.random.randint(-2**(bit_length//2), 2**(bit_length//2))
    t2 = 1-t1
    return t1, t2

# 将字符串数组转换为numpy数组
def str2arr(str_num, arr_shape):
    str_num = str_num.reshape(-1)
    str_all_num = ''
    for i in str_num: 
        str_all_num +=i # 字符串是'100...01'第0位是最高位
    return np.array(list(str_all_num)).astype(np.int).reshape(arr_shape)

'''
    安全最重要比特协议 SecMSB
'''
def SecMSB(u1, u2, bit_length=32):
    L = bit_length
    u1_shape = u1.shape
    print('u1_shape: ',u1_shape)
    u_ori_shape = list(u1_shape)[:-1]
    # t_shape = u_ori_shape.copy().append(L+1)
    t_shape = u_ori_shape.copy()
    t_shape.append(L+1)
    print('u_ori_shape: ',u_ori_shape)
    print('t_shape: ',t_shape)

    t1 = np.zeros(t_shape).astype(np.int) # t1=[input_shape, L+1]
    t2 = np.zeros(t_shape).astype(np.int)
    f1 = np.zeros(u1_shape).astype(np.int)
    f2 = np.zeros(u1_shape).astype(np.int)

    # 初始化zeta
    zeta1 = np.zeros(u_ori_shape).astype(np.int)
    zeta2 = np.zeros(u_ori_shape).astype(np.int)

    # offline
    # 初始化t1, t2（t1+t2=1） 采用切片方法
    t1[...,0] = np.random.randint(-2**(L//2), 2**(L//2))
    t2[...,0] = 1-t1[...,0]
     
    # online
    for i in range(0, L):
        # t1[i], t2[i] = SecMul(t1[i+1], 1-u1[i], t2[i+1], u2[i])
        t1[...,i+1], t2[...,i+1] = SecMul_matrix(t1[...,i], 1-u1[...,i], t2[...,i], -u2[...,i])
        # print('t1+t2: ',t1[...,i]+t2[...,i])

    for i in range(0, L):
        # S1
        # f1[i] = t1[i+1]-t1[i]
        f1[...,i] = t1[...,i]-t1[...,i+1]
        # S2
        # f2[i] = t2[i+1]-t2[i]
        f2[...,i] = t2[...,i]-t2[...,i+1]

    for i in range(0, L):
        # zeta1 += f1[i]
        zeta1 += f1[...,i]
        # zeta2 += f2[i]
        zeta2 += f2[...,i]

    return f1,zeta1,f2,zeta2

def DFC(num, bit_length=32):
    binary = float2bin(num, bit_length)
    L = bit_length-1
    return binary[:L]

'''
SecCmp
'''
def SecCmp(u1,v1,u2,v2, bit_length=32):
    L = bit_length
    c1 = ndarray((L,), int)
    c2 = ndarray((L,), int)
    e1 = ndarray((L,), int)
    e2 = ndarray((L,), int)
    xi1 = 0
    xi2 = 0
    # S1
    a = u1-v1
    a_bin = float2bin(a)

    # S2
    b = v2-u2
    b_bin = float2bin(b)
    print("a,b: ", a,b)

    print("a_bin:", a_bin)
    print("b_bin:", b_bin)

    for i in range(0, L):
        c1[i], c2[i] = SecXor(int(a_bin[i]),int(b_bin[i]), 0,0)
    print("c1,c2: ", c1+c2)
    
    d1,zeta1,d2,zeta2 = SecMSB(c1,c2,L)
    print("d,zeta: ", d1+d2,zeta1+zeta2)
    
    for i in range(0, L):
        e1[i], e2[i] = SecMul(int(a_bin[i]), d1[i], 0, d2[i])
        
    print("e: ", e1+e2)

    for i in range(0, L):
        xi1 += e1[i]
        xi2 += e2[i]
    
    # print(int(a_bin[L-1]),int(b_bin[]))
    iota1, iota2 = SecOr(int(a_bin[0]),int(b_bin[0]), 0, 0)
    print("iota:", iota1+iota2)
    nu1, nu2 = SecXor(iota1, xi1, iota2, xi2)
    print("nu:", nu1+nu2)
    f1, f2 = SecMul(nu1, zeta1, nu2, zeta2)

    return f1, f2

def test():
    '''float2Int'''
    u = np.random.randn(2,4,2)
    print('u: \n', u)
    print('u[]: \n',u[..., 0])
    u_bin = float2bin(u, 32)
    print('u_bin: \n', u_bin)


if __name__ == '__main__':
    test()

